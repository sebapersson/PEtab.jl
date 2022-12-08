# Handles piecewise functions that are to be redefined with ifelse statements in the model
# equations to allow MKT symbolic calculations.
# Calls goToBottomPiecewiseToEvent to handle multiple logical conditions.
function rewritePiecewiseToIfElse(ruleFormula, variable, modelDict, baseFunctions; retFormula::Bool=false)

    piecewiseStrings = getPiencewiseStr(ruleFormula)
    eqSyntaxDict = Dict() # Hold the Julia syntax for iffelse statements

    # If the rule variable is a part of the parameters list remove it
    if variable in keys(modelDict["parameters"])
        delete!(modelDict["parameters"], variable)
    end

    # Loop over each piecewise statement
    for i in eachindex(piecewiseStrings)

        piecewiseString = (piecewiseStrings[i])[11:end-1] # Extract everything inside piecewise

        args = splitBetween(piecewiseString, ',')
        vals = args[1:2:end]
        conds = args[2:2:end]

        # In case our variable is the sum of several piecewise bookeep each piecewise
        if length(piecewiseStrings) > 1
            varChange = variable * "Event" * string(i)
        else
            varChange = variable
        end

        if length(conds) > 1
            println("Warning : Breaking example with multile conds")
        end

        # Process the piecewise into ifelse statements
        cIndex, condition = 1, conds[1]

        # Check if we have nested piecewise within either the active or inactive value. If true, apply recursion
        # to reach bottom level of piecewise.
        if occursin("piecewise(", vals[cIndex])
            valActive = rewritePiecewiseToIfElse(vals[cIndex], "foo", modelDict, baseFunctions, retFormula=true)#[7:end]
            valActive = rewriteDerivatives(valActive, modelDict, baseFunctions)
        else
            valActive = rewriteDerivatives(vals[cIndex], modelDict, baseFunctions)
        end
        if occursin("piecewise(", vals[end])
            println("vals = ", vals[end])
            valInactive = rewritePiecewiseToIfElse(vals[end], "foo", modelDict, baseFunctions, retFormula=true)#[7:end]
            println("valInActive = ", valInactive)
            valInactive = rewriteDerivatives(valInactive, modelDict, baseFunctions)
        else
            valInactive = rewriteDerivatives(vals[end], modelDict, baseFunctions)
        end

        if condition[1:2] == "lt" || condition[1:2] == "gt" || condition[1:3] == "geq" || condition[1:3] == "leq"
            eqSyntaxDict[varChange] = simplePiecewiseToIfElse(condition, varChange, valActive, valInactive, modelDict, baseFunctions)
        elseif condition[1:3] == "and" || condition[1:2] == "if"
            eqSyntaxDict[varChange] = complexPiecewiseToIfElse(condition, variable, valActive, valInactive, modelDict, baseFunctions)
        else
            println("Error : Somehow we cannot process the piecewise expression")
        end
    end

    # Add the rule as equation into the model
    delete!(modelDict["inputFunctions"], "foo")
    strInput = variable * " ~ "
    formulaUse = deepcopy(ruleFormula)
    if length(piecewiseStrings) > 1
        for i in eachindex(piecewiseStrings)
            formulaUse = replace(formulaUse, piecewiseStrings[i] => eqSyntaxDict[variable * "Event" * string(i)])
        end
    else
        formulaUse = replace(formulaUse, piecewiseStrings[1] => eqSyntaxDict[variable])
    end
    if retFormula == false
        modelDict["inputFunctions"][variable] = strInput * rewriteDerivatives(formulaUse, modelDict, baseFunctions)
        return nothing
    else
        return formulaUse
    end
end


function getPiencewiseStr(strArg::AbstractString)::Array{String, 1}

    # Extract in a string the substrings captured by the piecewise
    iPiecewise = findall("piecewise(", strArg)
    nPiecewise = length(iPiecewise)
    piecewiseStr = fill("", nPiecewise)

    # Extract entire piecewise expression. Handles inner paranthesis, e.g
    # when we have "piecewise(0, lt(t - insulin_time_1, 0), 1)" it extracts
    # the full expression. Also does not extrat nested. Will not extract the innner
    # one for
    # piecewise(beta_0, lt(t, t_1), piecewise(beta_1, lt(t, t_2), beta_2 * (1 - beta_2_multiplier)))
    i, k = 1, 1
    while i <= nPiecewise
        iStart = iPiecewise[i][1]
        nInnerParanthesis = 0
        iEnd = iPiecewise[i][end]
        while true
            iEnd += 1
            if nInnerParanthesis == 0 && strArg[iEnd] == ')'
                break
            end

            if strArg[iEnd] == '('
                nInnerParanthesis += 1
            end

            if strArg[iEnd] == ')'
                nInnerParanthesis -= 1
            end
        end

        piecewiseStr[k] = strArg[iStart:iEnd]
        k += 1

        # Check the number of piecewise inside the piecewise to avoid counting nested ones
        nInnerPiecewise = length(findall("piecewise(", strArg[iStart:iEnd]))
        i += nInnerPiecewise
    end

    return piecewiseStr[piecewiseStr .!== ""]
end


function simplePiecewiseToIfElse(condition, variable, valActive, valInactive, dicts, baseFunctions)

    if "leq" == condition[1:3]
        strippedCondition = condition[5:end-1]
        inEqUse = " <= "
    elseif "lt" == condition[1:2]
        strippedCondition = condition[4:end-1]
        inEqUse = " < "
    elseif "geq" == condition[1:3]
        strippedCondition = condition[5:end-1]
        inEqUse = " >= "
    elseif "gt" == condition[1:2]
        strippedCondition = condition[4:end-1]
        inEqUse = " > "
    else
        println("Cannot recognize form of inequality")
    end

    parts = splitBetween(strippedCondition, ',')
    # Trigger of event
    expression = "ifelse(" * parts[1] * inEqUse * parts[2] * ", " * valActive * ", " * valInactive * ")"

    println("Expression = $expression")

    return expression
end


function complexPiecewiseToIfElse(condition, variable, valActive, valInactive, modelDict, baseFunctions)
    eventStr = recursionComplexPiecewise(condition, variable, modelDict, baseFunctions)
    return eventStr * " * (" * valActive * ") + (1 - " * eventStr *") * (" * valInactive * ")"
end


# As MTK does not support iffelse with multiple comparisons, e.g, a < b && c < d when we have nested piecewise
# with && and || statements the situation is more tricky when trying to create a reasonable expression.
function recursionComplexPiecewise(condition, variable, modelDict, baseFunctions)

    if "and" == condition[1:3]
        strippedCondition = condition[5:end-1]
        lPart, rPart = splitBetween(strippedCondition, ',')
        lPartExp = recursionComplexPiecewise(lPart, variable, modelDict, baseFunctions)
        rPartExp = recursionComplexPiecewise(rPart, variable, modelDict, baseFunctions)

        # An or statment can in a differentiable way here be encoded as sigmoid function
        return "(" * lPartExp * ") * (" * rPartExp * ")"

    elseif "if" == condition[1:2]
        strippedCondition = condition[4:end-1]
        lPart, rPart = splitBetween(strippedCondition, ',')
        lPartExp = recursionComplexPiecewise(lPart, variable, modelDict, baseFunctions)
        rPartExp = recursionComplexPiecewise(rPart, variable, modelDict, baseFunctions)

        return "1 / (exp(-" * lpartExp * "+" *  rPartExt *") + 1)"
    else
        return simplePiecewiseToIfElse(condition, variable, "1.0", "0.0", modelDict, baseFunctions)
    end
end


# Splits strings by a given delimiter, but only if the delimiter is not inside a function / parenthesis.
function splitBetween(stringToSplit, delimiter)
    parts = Vector{SubString{String}}(undef, length(stringToSplit))
    numParts = 0
    inParenthesis = 0
    startPart = 1
    endPart = 1
    for i in eachindex(stringToSplit)
        if stringToSplit[i] == '('
            inParenthesis += 1
        elseif stringToSplit[i] == ')'
            inParenthesis -= 1
        end
        if stringToSplit[i] == delimiter && inParenthesis == 0
            endPart = i-1
            numParts += 1
            parts[numParts] = stringToSplit[startPart:endPart]
            parts[numParts] = strip(parts[numParts])
            startPart = i+1
        end
    end
    numParts += 1
    parts[numParts] = stringToSplit[startPart:end]
    parts[numParts] = strip(parts[numParts])
    parts = parts[1:numParts]
end
