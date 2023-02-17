
"""
    petabFormulaToJulia(formula::String, stateNames, paramData::ParametersInfo, namesParamDyn::Vector{String}, namesNonDynParam::Vector{String})::String
    
    Translate a peTab formula (e.g for observable or for sd-parameter) into Julia syntax and output the result 
    as a string.
"""
function petabFormulaToJulia(formula::String, 
                             modelStateNames::Vector{String}, 
                             parameterInfo::ParametersInfo, 
                             θ_dynamicNames::Vector{String}, 
                             θ_nonDynamicNames::Vector{String})::String

    # Characters directly translate to Julia and characters that also are assumed to terminate a word (e.g state 
    # and parameter)
    charDirectlyTranslate = ['(', ')', '+', '-', '/', '*', '^'] 
    lenFormula = length(formula)
    
    i, juliaFormula = 1, ""
    while i <= lenFormula
        # In case character i of the string can be translated directly 
        if formula[i] in charDirectlyTranslate
            juliaFormula *= formula[i] * " "
            i += 1

        # In case character i cannot be translated directly (is part of a word)
        else
            # Get word (e.g param, state, math-operation or number)
            word, iNew = getWord(formula, i, charDirectlyTranslate)
            # Translate word to Julia syntax 
            juliaFormula *= wordToJuliaSyntax(word, modelStateNames, parameterInfo, θ_dynamicNames, θ_nonDynamicNames)
            i = iNew

            # Special case where we have multiplication
            if isNumber(word) && i <= lenFormula && isletter(formula[i])
                juliaFormula *= "* "
            end
        end
    end

    return juliaFormula
end


"""
    getWord(str::String, iStart::Int, charTerminate::Vector{Char})

    In a string starting from position iStart extract the next "word", which is the longest 
    concurent occurance of characters that are not in the character list with word termination 
    characters. Returns the word and iEnd (the position where the word ends).
    
    For example, if charListTerm = ['(', ')', '+', '-', '/', '*', '^'] abc123 is 
    considered a word but not abc123*.  
"""
function getWord(str::String, iStart::Int, charTerminate::Vector{Char})
    
    wordStr = ""
    iEnd = iStart

    # If the first character is a numberic the termination occurs when 
    # the first non-numeric character (not digit or dot .) is reached. 
    startIsNumeric = isnumeric(str[iEnd])

    while iEnd <= length(str)

        # In case str[iEnd] is not a termination character we need to be careful with numbers 
        # so that we handle sciencetific notations correctly, e.g we do not consider 
        # 1.2e-3 to be two words [1.2e, 3] but rather a single workd.
        if !(str[iEnd] in charTerminate) 

            # Parase sciencetific notation for number 
            if startIsNumeric == true && str[iEnd] == 'e'
                if length(str) > iEnd && (str[iEnd+1] == '-' || isnumeric(str[iEnd+1]))
                    if str[iEnd+1] == '-'
                        iEnd += 2
                        wordStr *= "e-"
                    else
                        iEnd += 1
                        wordStr *= "e"
                    end

                else
                    break 
                end
            end

            if startIsNumeric == true && !(isnumeric(str[iEnd]) || str[iEnd] == '.')
                break
            end
            wordStr *= str[iEnd]
        else
            break 
        end
        iEnd += 1
    end
    # Remove all spaces from the word
    wordStr = replace(wordStr, " " => "")
    return wordStr, iEnd
end


""""
    wordToJuliaSyntax(wordTranslate::String, 
                      stateNames,
                      paramData::ParametersInfo, 
                      namesParamDyn::Vector{String})::String

    Translate a word (state, parameter, math-expression or number) into Julia syntax 
    when building Ymod, U0 and Sd functions.

"""
function wordToJuliaSyntax(wordTranslate::String, 
                           modelStateNames::Vector{String},
                           parameterInfo::ParametersInfo, 
                           θ_dynamicNames::Vector{String}, 
                           θ_nonDynamicNames::Vector{String})::String

    # List of mathemathical operations that are accpeted and will be translated 
    # into Julia syntax (t is assumed to be time)
    listOperations = ["exp", "sin", "cos", "t"]

    wordJuliaSyntax = ""    
    # If wordTranslate is a constant parameter 
    if wordTranslate ∈ string.(parameterInfo.parameterId) && wordTranslate ∉ θ_dynamicNames && wordTranslate ∉ θ_nonDynamicNames
        # Constant parameters get a _C appended to tell them apart 
        wordJuliaSyntax *= wordTranslate * "_C"
    end

    if wordTranslate ∈ θ_dynamicNames
        wordJuliaSyntax *= wordTranslate
    end

    if wordTranslate ∈ θ_nonDynamicNames
        wordJuliaSyntax *= wordTranslate
    end

    if wordTranslate ∈ modelStateNames
        wordJuliaSyntax *= wordTranslate
    end

    if isNumber(wordTranslate)
        wordJuliaSyntax *= wordTranslate
    end

    if wordTranslate in listOperations
        wordJuliaSyntax *= listOperations[wordTranslate .== listOperations]
    end

    if length(wordTranslate) >= 19 && wordTranslate[1:19] == "observableParameter"
        wordJuliaSyntax *= wordTranslate
    end

    if length(wordTranslate) >= 14 && wordTranslate[1:14] == "noiseParameter"
        wordJuliaSyntax *= wordTranslate
    end

    if isempty(wordTranslate)
        println("Warning : When creating observation function $wordTranslate could not be processed")
    end

    wordJuliaSyntax *= " "

    return wordJuliaSyntax
end


"""
    getObservableParametersStr(formula::String)::String

    Helper function to extract all observableParameter in the observableFormula in the PEtab-file. 
"""
function getObservableParametersStr(formula::String)::String
    
    # Find all words on the form observableParameter
    _observableParameters = sort(unique([ match.match for match in eachmatch(r"observableParameter[0-9]_\w+", formula)]))
    observableParameters = ""
    for i in eachindex(_observableParameters)
        if i != length(_observableParameters) 
            observableParameters *= _observableParameters[i] * ", "
        else
            observableParameters *= _observableParameters[i] 
        end
    end

    return observableParameters
end


"""
    getNoiseParametersStr(formula::String)::String

    Helper function to extract all the noiseParameter in noiseParameter formula in the PEtab file.
"""
function getNoiseParametersStr(formula::String)::String
    
    # Find all words on the form observableParameter
    _noiseParameters = [ match.match for match in eachmatch(r"noiseParameter[0-9]_\w+", formula) ]
    noiseParameters = ""
    for i in eachindex(_noiseParameters)
        if i != length(_noiseParameters) 
            noiseParameters *= _noiseParameters[i] * ", "
        else
            noiseParameters *= _noiseParameters[i] 
        end
    end

    return noiseParameters
end


"""
    replaceVariablesWithArrayIndex(formula,stateNames,parameterNames,namesNonDynParam,paramData)::String

    Replaces any state or parameter from formula with their corresponding index in the ODE system 
    Symbolics can return strings without multiplication sign, e.g. 100.0STAT5 instead of 100.0*STAT5 
    so replaceWholeWord cannot be used here
"""
function replaceVariablesWithArrayIndex(formula::String, 
                                        modelStateNames::Vector{String},
                                        parameterInfo::ParametersInfo, 
                                        pNames::Vector{String},
                                        θ_nonDynamicNames::Vector{String};
                                        pODEProblem::Bool=false)::String
    
    for (i, stateName) in pairs(modelStateNames)
        formula = replaceWholeWordWithNumberPrefix(formula, stateName, "u["*string(i)*"]")
    end

    if pODEProblem == true
        for (i, pName) in pairs(pNames)
            formula = replaceWholeWordWithNumberPrefix(formula, pName, "pODEProblem["*string(i)*"]")
        end
    else
        for (i, pName) in pairs(pNames)
            formula = replaceWholeWordWithNumberPrefix(formula, pName, "θ_dynamic["*string(i)*"]")
        end
    end

    for (i, θ_nonDynamicName) in pairs(θ_nonDynamicNames)
        formula = replaceWholeWordWithNumberPrefix(formula, θ_nonDynamicName, "θ_nonDynamic["*string(i)*"]")
    end
    
    for i in eachindex(parameterInfo.parameterId)
        if parameterInfo.estimate[i] == false
            formula = replaceWholeWordWithNumberPrefix(formula, string(parameterInfo.parameterId[i]) * "_C", "parameterInfo.nominalValue[" * string(i) *"]")
        end
    end

    return formula
end


"""
    replaceExplicitVariableWithRule(formula::String, SBMLDict::Dict)::String

    Replace the explicit rule variable with the explicit rule
"""
function replaceExplicitVariableWithRule(formula::String, SBMLDict::Dict)::String
    for (key,value) in SBMLDict["modelRuleFunctions"]            
        formula = replaceWholeWord(formula, key, "(" * value[2] * ")")
    end
    return formula
end


"""
    replaceWholeWordWithNumberPrefix(formula, from, to)::String
    
    Replaces variables that can be prefixed with numbers, e.g., 
    replaceWholeWordWithNumberPrefix("4STAT5 + 100.0STAT5 + RE*STAT5 + STAT5","STAT5","u[1]") gives
    4u[1] + 100.0u[1] + RE*u[1] + u[1]   
"""
function replaceWholeWordWithNumberPrefix(oldString, replaceFrom, replaceTo)
    replaceFromRegex = Regex("\\b(\\d+\\.?\\d*+)*(" * replaceFrom * ")\\b")
    replaceToRegex = SubstitutionString("\\1" * replaceTo )
    sleep(0.001)
    newString = replace(oldString, replaceFromRegex => replaceToRegex)
    return newString
end
