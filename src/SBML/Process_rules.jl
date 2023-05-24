function getRuleFormula(rule)

    ruleFormula = rule[:getFormula]()
    ruleFormula = replaceWholeWord(ruleFormula, "time", "t")
    ruleFormula = removePowFunctions(ruleFormula)

    return ruleFormula
end


function processAssignmentRule!(modelDict::Dict, ruleFormula::String, ruleVariable::String, baseFunctions)

    # If piecewise occurs in the rule we are looking at a time-based event which is encoded as an
    # event into the model to ensure proper evaluation of the gradient.
    if occursin("piecewise(", ruleFormula)
        rewritePiecewiseToIfElse(ruleFormula, ruleVariable, modelDict, baseFunctions)

    # If the rule does not involve a piecewise expression simply encode it as a function which downsteram
    # is integrated into the equations.
    else
        # Extract the parameters and states which make out the rule, further check if rule consists of an
        # existing function (nesting).
        arguments, includesFunction = getArguments(ruleFormula, modelDict["modelRuleFunctions"], baseFunctions)
        if isempty(arguments)
            modelDict["parameters"][ruleVariable] = ruleFormula
        else
            if includesFunction == true
                ruleFormula = replaceWholeWordDict(ruleFormula, modelDict["modelRuleFunctions"])
            end
            modelDict["modelRuleFunctions"][ruleVariable] = [arguments, ruleFormula]

            # As we hard-code the rule variable into the equation remove it as state or model parameter.
            # TODO : Add this as expression into the model eq. and allow structurally simplify to act on it.
            if ruleVariable in keys(modelDict["states"])
                modelDict["states"] = delete!(modelDict["states"], ruleVariable)
            end
            if ruleVariable in keys(modelDict["parameters"])
                modelDict["parameters"] = delete!(modelDict["parameters"], ruleVariable)
            end
        end
    end

end


function processRateRule!(modelDict::Dict, ruleFormula::String, ruleVariable::String, baseFunctions)

    # Rewrite rule to function if there are not any piecewise, eles rewrite to formula with ifelse
    if occursin("piecewise(", ruleFormula)
        ruleFormula = rewritePiecewiseToIfElse(ruleFormula, ruleVariable, modelDict, baseFunctions, retFormula=true)
    else
        arguments, includesFunction = getArguments(ruleFormula, modelDict["modelRuleFunctions"], baseFunctions)
        if arguments != "" && includesFunction == true
            ruleFormula = replaceWholeWordDict(ruleFormula, modelDict["modelRuleFunctions"])
        end
    end

    # Add rate rule as part of model derivatives and remove from parameters dict if rate rule variable
    # is a parameter
    if ruleVariable in keys(modelDict["states"])
        modelDict["derivatives"][ruleVariable] = "D(" * ruleVariable * ") ~ " * ruleFormula

    elseif ruleVariable in keys(modelDict["nonConstantParameters"])
        modelDict["states"][ruleVariable] = modelDict["nonConstantParameters"][ruleVariable]
        delete!(modelDict["nonConstantParameters"], ruleVariable)
        modelDict["derivatives"][ruleVariable] = "D(" * ruleVariable * ") ~ " * ruleFormula

    elseif ruleVariable in keys(modelDict["parameters"])
        modelDict["states"][ruleVariable] = modelDict["parameters"][ruleVariable]
        delete!(modelDict["parameters"], ruleVariable)
        modelDict["derivatives"][ruleVariable] = "D(" * ruleVariable * ") ~ " * ruleFormula
    else
        println("Warning : Cannot find rate rule variable in either model states or parameters")
    end
end


# Rewrites time-dependent ifElse-statements to depend on a boolean variable. This makes it possible to treat piecewise
# as events, allowing us to properly handle discontinious. Does not rewrite ifElse if the activation criteria depends
# on a state.
function timeDependentIfElseToBool!(modelDict::Dict)

    # Rewrite piecewise using Boolean variables. Due to the abillity of piecewiese statements to be nested
    # recursion is needed.
    for key in keys(modelDict["inputFunctions"])
        formulaWithIfelse = modelDict["inputFunctions"][key]
        modelDict["inputFunctions"][key] = reWriteStringWithIfelseToBool(string(formulaWithIfelse), modelDict, key)
    end
end


function reWriteStringWithIfelseToBool(formulaWithIfelse::String, modelDict::Dict, key::String)::String

    formulaReplaced = formulaWithIfelse

    indexIfElse = getIndexPiecewise(formulaWithIfelse)
    if isempty(indexIfElse)
        return formulaReplaced
    end

    for i in eachindex(indexIfElse)

        ifelseFormula = formulaWithIfelse[indexIfElse[i]][8:end-1]
        activationRule, leftSide, rightSide = splitIfElse(ifelseFormula)

        # Find inequality
        iLt = findfirst(x -> x == '<', activationRule)
        iGt = findfirst(x -> x == '>', activationRule)
        if isnothing(iGt) && !isnothing(iLt)
            signUsed = "lt"
            if activationRule[iLt:(iLt+1)] == "<="
                splitBy = "<="
            else
                splitBy = "<"
            end
        elseif !isnothing(iGt) && isnothing(iLt)
            signUsed = "gt"
            if activationRule[iGt:(iGt+1)] == ">="
                splitBy = ">="
            else
                splitBy = ">"
            end
        else
            println("Error : Did not find criteria to split ifelse on")
        end
        lhsRule, rhsRule = split(activationRule, string(splitBy))

        # Identify which side of ifelse expression is activated with time
        timeRight = checkForTime(string(rhsRule))
        timeLeft = checkForTime(string(lhsRule))
        rewriteIfElse = true
        if timeLeft == false && timeLeft == false
            println("Have ifelse statements which does not contain time. Hence we do not rewrite as event, but rather keep it as an ifelse.")
            rewriteIfElse = false
            continue
        elseif timeLeft == true
            signTime = checkSignTime(string(lhsRule))
            if (signTime == 1 && signUsed == "lt") || (signTime == -1 && signUsed == "gt")
                sideActivatedWithTime = "right"
            elseif (signTime == 1 && signUsed == "gt") || (signTime == -1 && signUsed == "lt")
                sideActivatedWithTime = "left"
            end
        elseif timeRight == true
            signTime = checkSignTime(string(rhsRule))
            if (signTime == 1 && signUsed == "lt") || (signTime == -1 && signUsed == "gt")
                sideActivatedWithTime = "left"
            elseif (signTime == 1 && signUsed == "gt") || (signTime == -1 && signUsed == "lt")
                sideActivatedWithTime = "right"
            end
        end

        # In case of nested ifelse rewrite left-hand and right-hand side
        leftSide = reWriteStringWithIfelseToBool(string(leftSide), modelDict, key)
        rightSide = reWriteStringWithIfelseToBool(string(rightSide), modelDict, key)

        if rewriteIfElse == true
            j = 1
            local varName = ""
            while true
                varName = string(key) * "_bool" * string(j)
                if varName ∉ keys(modelDict["boolVariables"])
                    break
                end
                j += 1
            end
            activatedWithTime = sideActivatedWithTime == "left" ? leftSide : rightSide
            deActivatedWithTime = sideActivatedWithTime == "left" ? rightSide : leftSide
            formulaInModel = "((1 - " * varName * ")*" * "(" * deActivatedWithTime *") + " * varName * "*(" * activatedWithTime * "))"
            modelDict["parameters"][varName] = "0.0"
            formulaReplaced = replace(formulaReplaced, formulaWithIfelse[indexIfElse[i]] => formulaInModel)
            modelDict["boolVariables"][varName] = [activationRule, sideActivatedWithTime]
        end
    end

    return formulaReplaced
end


# Here we assume we receive the arguments to ifelse(a ≤ 1 , b, c) on the form
# a ≤ 1, b, c and our goal is to return tuple(a ≤ 1, b, c)
function splitIfElse(str::String)
    paranthesisLevel = 0
    split, i = 1, 1
    firstSet, secondSet, thirdSet = 1, 1, 1
    while i < length(str)

        if str[i] == '('
            paranthesisLevel += 1
        elseif str[i] == ')'
            paranthesisLevel -= 1
        end

        if str[i] == ',' && paranthesisLevel == 0
            if split == 1
                firstSet = 1:(i-1)
                split += 1
            elseif split == 2
                secondSet = (firstSet[end]+2):(i-1)
                thirdSet = (secondSet[end]+2):length(str)
                break
            end
        end
        i += 1
    end
    return str[firstSet], str[secondSet], str[thirdSet]
end


function getIndexPiecewise(str::String)

    ret = Array{Any, 1}(undef, 0)

    iStart, iEnd = 0, 0
    i = 1
    while i < length(str)

        if !(length(str) > i+6)
            break
        end

        if str[i:(i+5)] == "ifelse"
            iStart = i
            paranthesisLevel = 1
            for j in (i+7):length(str)
                if str[j] == '('
                    paranthesisLevel += 1
                elseif str[j] == ')'
                    paranthesisLevel -= 1
                end
                if paranthesisLevel == 0
                    iEnd = j
                    break
                end
            end
            ret = push!(ret, collect(iStart:iEnd))
            i = iEnd + 1
            continue
        end
        i += 1
    end

    return ret
end
