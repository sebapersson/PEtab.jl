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
    
    elseif ruleVariable in keys(parameterDict)
        modelDict["states"][ruleVariable] = parameterDict[ruleVariable]
        delete!(parameterDict, ruleVariable)
        modelDict["derivatives"][ruleVariable] = "D(" * ruleVariable * ") ~ " * ruleFormula
    else
        println("Warning : Cannot find rate rule variable in either model states or parameters")
    end
end