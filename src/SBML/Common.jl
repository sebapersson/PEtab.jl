# Handles piecewise functions that are to be redefined with ifelse statements in the model
# equations to allow MKT symbolic calculations.
# Calls goToBottomPiecewiseToEvent to handle multiple logical conditions.
function rewritePiecewiseToIfElse(ruleFormula, variable, modelDict, baseFunctions, modelSBML; retFormula::Bool=false)

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
            println("Warning : Potentially breaking example with multiple conditions")
        end

        # Process the piecewise into ifelse statements
        cIndex, condition = 1, conds[1]

        # Check if we have nested piecewise within either the active or inactive value. If true, apply recursion
        # to reach bottom level of piecewise.
        if occursin("piecewise(", vals[cIndex])
            valActive = rewritePiecewiseToIfElse(vals[cIndex], "foo", modelDict, baseFunctions, modelSBML, retFormula=true)#[7:end]
            valActive = rewriteDerivatives(valActive, modelDict, baseFunctions, modelSBML)
        else
            valActive = rewriteDerivatives(vals[cIndex], modelDict, baseFunctions, modelSBML)
        end
        if occursin("piecewise(", vals[end])
            valInactive = rewritePiecewiseToIfElse(vals[end], "foo", modelDict, baseFunctions, modelSBML, retFormula=true)#[7:end]
            valInactive = rewriteDerivatives(valInactive, modelDict, baseFunctions, modelSBML)
        else
            valInactive = rewriteDerivatives(vals[end], modelDict, baseFunctions, modelSBML)
        end

        if condition[1:2] == "lt" || condition[1:2] == "gt" || condition[1:3] == "geq" || condition[1:3] == "leq"
            eqSyntaxDict[varChange] = simplePiecewiseToIfElse(condition, varChange, valActive, valInactive, modelDict, baseFunctions)
        elseif condition[1:3] == "and" || condition[1:2] == "if"
            eqSyntaxDict[varChange] = complexPiecewiseToIfElse(condition, variable, valActive, valInactive, modelDict, baseFunctions)
        else
            @error "Somehow we cannot process the piecewise expression"
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
        modelDict["inputFunctions"][variable] = strInput * rewriteDerivatives(formulaUse, modelDict, baseFunctions, modelSBML)
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


# Check if time is present in a string (used for rewriting piecewise to event)
function checkForTime(str::String)
    strNoWhitespace = replace(str, " " => "")

    # In case we find time t
    iT = 0
    findT = false
    for i in eachindex(strNoWhitespace)
        if strNoWhitespace[i] == 't'
            if (i > 1 && i < length(strNoWhitespace)) && !isletter(strNoWhitespace[i-1]) && !isletter(strNoWhitespace[i+1])
                findT = true
                iT = i
                break
            elseif i == 1 && i < length(strNoWhitespace) && !isletter(strNoWhitespace[i+1])
                findT = true
                iT = i
                break
            elseif i == 1 && i == length(strNoWhitespace)
                findT = true
                iT = i
                break
            elseif 1 == length(strNoWhitespace)
                findT = true
                iT = i
                break
            elseif i == length(strNoWhitespace) && length(strNoWhitespace) > 1 && !isletter(strNoWhitespace[i-1])
                findT = true
                iT = i
                break
            end
        end
    end

    return findT
end


# If we identity time in an ifelse expression identify the sign of time to know whether or not the ifelse statement will
# or will not be triggered with time.
function checkSignTime(str::String)

    # Easy special case with single term
    strNoWhitespace = replace(str, " " => "")
    strNoWhitespace = replace(strNoWhitespace, "(" => "")
    strNoWhitespace = replace(strNoWhitespace, ")" => "")
    if strNoWhitespace == "t"
        return 1
    end

    terms = findTerms(strNoWhitespace)
    iTime = 0
    for i in eachindex(terms)
        iStart, iEnd = terms[i]
        if checkForTime(strNoWhitespace[iStart:iEnd]) == true
            iTime = i
            break
        end
    end
    iStart, iEnd = terms[iTime]
    strTime = strNoWhitespace[iStart:iEnd]
    signTime = findSignTerm(strTime)
    if iStart == 1
        signBefore = 1
    elseif strNoWhitespace[iStart-1] == '-'
        signBefore = -1
    else
        signBefore = 1
    end

    return signTime * signBefore
end


# Returns the end index for a paranthesis for a string, assuming that the string starts
# with a paranthesis
function findIParanthesis(str::String)
    numberNested = 0
    iEnd = 1
    for i in eachindex(str)
        if str[i] == '('
            numberNested += 1
        end
        if str[i] == ')'
            numberNested -= 1
        end
        if numberNested == 0
            iEnd = i
            break
        end
    end
    return iEnd
end


# For a mathemathical expression finds the terms
function findTerms(str::String)
    iTerm = Array{Tuple, 1}(undef, 0)
    i = 1
    while i < length(str)
        if str[i] == '-' || str[i] == '+' || isletter(str[i]) || isnumeric(str[i]) || str[i] == '('
            iEnd = 0
            iStart = str[i] ∈ ['-', '+'] ? i+1 : i
            j = iStart
            while j ≤ length(str)
                if str[j] == '+'
                    iEnd = j-1
                    i = j
                    break
                end
                if str[j] == '-'
                    if length(str) ≥ j+1 && (isnumeric(str[j+1]) || isletter(str[j+1]))
                        if j == iStart
                            j += 1
                            continue
                        elseif str[j-1] ∈ ['*', '/']
                            j += 1
                            continue
                        else
                            iEnd = j-1
                            i = j
                            break
                        end
                    else
                        iEnd = j-1
                        i = j
                        break
                    end
                end
                if str[j] == '('
                    j += (findIParanthesis(str[j:end]) - 1)
                    if j == length(str)
                        iEnd = j
                        i = j
                        break
                    end
                end
                j += 1
                if j ≥ length(str)
                    j = length(str)
                    iEnd = j
                    i = j
                    break
                end
            end
            iTerm = push!(iTerm, tuple(iStart, iEnd))
        end
    end
    return iTerm
end


# For a string like a*b/(c+d) identify sign of the product assuming all variables,
# e.g a, b, c, d... are positive.
function findSignTerm(str::String)
    # Identify each factor
    iFactor = Array{Tuple, 1}(undef, 0)
    i = 1
    while i ≤ length(str)
        iStart = i
        j = iStart
        iEnd = 0
        while j ≤ length(str)
            if str[j] ∈ ['*', '/']
                iEnd = j - 1
                i = j+1
                break
            end
            if length(str) == j
                iEnd = j
                i = j + 1
                break
            end
            if str[j] == '('
                j += (findIParanthesis(str[j:end]) - 1)
                iEnd = j
                if length(str) > j+1 && str[j+1] ∈ ['*', '/']
                    i = j+2
                else
                    i = j+1
                end
                break
            end
            j += 1
        end
        iFactor = push!(iFactor, tuple(iStart, iEnd))
        if iEnd == length(str)
            break
        end
    end

    signTerms = ones(length(iFactor))
    for i in eachindex(iFactor)

        iStart, iEnd = iFactor[i]

        if str[iStart] == '('
            signTerms[i] = getSignExpression(str[(iStart+1):(iEnd-1)])
        elseif '-' ∈ str[iStart:iEnd]
            signTerms[i] = -1
        else
            signTerms[i] = 1
        end
    end
    return prod(signTerms)
end


# Get the sign of a factor like "(a + b * (c + d)*-1) assuming all variables are
# positive. In case we cannot infer the sign Inf is returned. Employs recursion to
# handle paranthesis
function getSignExpression(str::String)

    iTerms = findTerms(str)
    signTerms = ones(Float64, length(iTerms)) * 100
    for i in eachindex(signTerms)

        # Get the sign before the term
        iStart, iEnd = iTerms[i]
        if iStart == 1
            signBeforeTerm = 1
        elseif str[iStart-1] == '-'
            signBeforeTerm = -1
        elseif str[iStart-1] == '+'
            signBeforeTerm = 1
        else
            println("Cannot infer sign before term")
        end

        valRet = findSignTerm(str[iStart:iEnd])
        signTerms[i] = signBeforeTerm * valRet

    end

    if all(i -> i == 1, signTerms)
        return 1
    elseif all(i -> i == -1, signTerms)
        return -1
    # In case all terms do not have the same sign and we thus cannot solve
    # without doubt the sign.
    else
        return Inf
    end
end
