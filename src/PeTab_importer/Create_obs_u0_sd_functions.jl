# TODO: Refactor yMod and SD functions to avoid redundant code 


"""
    createFileYmodSdU0(modelName::String, 
                       dirModel::String, 
                       odeSys::ODESystem, 
                       stateMap,
                       modelDict)

    For a PeTab model with name modelName with all PeTab-files in dirModel and associated 
    ModellingToolkit ODESystem (with its stateMap) build a file containing a functions for 
    i) computing the observable model value (yMod) ii) compute the initial value u0 (by using the 
    stateMap) and iii) computing the standard error (sd) for each observableFormula in the 
    observables PeTab file.

    Note - The produced Julia file will go via the JIT-compiler.
"""
function createFileYmodSdU0(modelName::String, 
                            dirModel::String, 
                            odeSys::ODESystem, 
                            stateMap,
                            modelDict)
                            
    parameterNames = parameters(odeSys)
    stateNames = states(odeSys)

    # Read data on experimental conditions and parameter values 
    experimentalConditionsFile, measurementDataFile, parameterBoundsFile, observablesDataFile = readDataFiles(dirModel, readObs=true)
    paramData = processParameterData(parameterBoundsFile) # Model data in convient structure 
    measurementData = processMeasurementData(measurementDataFile, observablesDataFile) 
    
    # Indices for mapping parameter-estimation vector to dynamic, observable and sd parameters correctly when calculating cost
    paramIndices = getIndicesParam(paramData, measurementData, odeSys, experimentalConditionsFile)
    
    createYmodFunction(modelName, dirModel, stateNames, paramData, paramIndices.namesDynParam, paramIndices.namesNonDynParam, observablesDataFile, modelDict)
    println("Done with Ymod function")
    println("")
    
    createU0Function(modelName, dirModel, paramData, string.(parameterNames), stateMap, inPlace=true)
    println("Done with u0 function in-place")
    println("")

    createU0Function(modelName, dirModel, paramData, string.(parameterNames), stateMap, inPlace=false)
    println("Done with u0 function not in-place")
    println("")

    createSdFunction(modelName, dirModel, paramData, stateNames, paramIndices.namesDynParam, paramIndices.namesNonDynParam, observablesDataFile)
    println("Done with sd function")
    println("")
end


"""
    createYmodFunction(modelName::String, 
                       dirModel::String, 
                       stateNames, 
                       paramData::ParamData, 
                       namesParamDyn::Array{String, 1}, 
                       observablesData::DataFrame,
                       modelDict)

    For modelName create a function for computing yMod by translating the observablesData
    PeTab-file into Julia syntax. 

    To correctly create the function the state-names, names of dynamic parameters to estiamte 
    (namesDynParam) and PeTab parameter-file (to get constant parameters) data are needed. 

    The modelDict is used to define explicit rules to the Ymod file.
"""
function createYmodFunction(modelName::String, 
                            dirModel::String, 
                            stateNames, 
                            paramData::ParamData, 
                            namesParamDyn::Array{String, 1}, 
                            namesNonDynParam::Array{String, 1},
                            observablesData::DataFrame,
                            modelDict)

    io = open(dirModel * "/" * modelName * "ObsSdU0.jl", "w")
    
    write(io, "function evalYmod(u, t, dynPar, obsPar, nonDynParam, paramData, observableId, mapObsParam) \n")

    # Extract names of model states and write to file 
    stateNamesShort = replace.(string.(stateNames), "(t)" => "")
    stateStr = "\n\t"
    for i in eachindex(stateNamesShort)
        stateStr *= stateNamesShort[i] * ", "
    end
    stateStr = stateStr[1:end-2]
    stateStr *= "= u \n"
    write(io, stateStr)

    # Extract name of dynamic parameter and write to file
    paramDynStr = "\t"
    for i in eachindex(namesParamDyn)
        paramDynStr *= namesParamDyn[i] * ", "
    end
    paramDynStr = paramDynStr[1:end-2]
    paramDynStr *= " = dynPar \n"
    write(io, paramDynStr)

    # Extract name of non-dynamic parameter and write to file
    if !isempty(namesNonDynParam)
        paramNonDynStr = "\t"
        for i in eachindex(namesNonDynParam)
            paramNonDynStr *= namesNonDynParam[i] * ", "
        end
        paramNonDynStr = paramNonDynStr[1:end-2]
        paramNonDynStr *= " = nonDynParam \n"
        write(io, paramNonDynStr)
    end

    # Extract constant parameters. To avoid cluttering the function only constant parameters that are used to 
    # to compute yMod are written to file.
    paramConstStr = ""
    for i in eachindex(paramData.parameterID)
        if paramData.shouldEst[i] == false
            paramConstStr *= "\t" * paramData.parameterID[i] * "_C = paramData.paramVal[" * string(i) *"] \n" 
        end
    end
    paramConstStr *= "\n"
    write(io, paramConstStr)

    # Create namesExplicitRules, if length(modelDict["modelRuleFunctions"]) = 0 this becomes a String[].
    namesExplicitRules = Array{String, 1}(undef,length(modelDict["modelRuleFunctions"]))
    # Extract the keys from the modelRuleFunctions
    for (index, key) in enumerate(keys(modelDict["modelRuleFunctions"]))
        namesExplicitRules[index] = key
    end
    
    # Write the formula of each observable to file
    observableIDs = String.(observablesData[!, "observableId"])
    strObserveble = ""
    for i in eachindex(observableIDs)
        # Each observebleID falls below its own if-statement 
        strObserveble *= "\tif observableId == " * "\"" * observableIDs[i] * "\"" * " \n"
        tmpFormula = filter(x -> !isspace(x), String(observablesData[i, "observableFormula"]))

        # Extract observable parameters 
        obsParam = getObsParamStr(tmpFormula)
        if !isempty(obsParam)
            strObserveble *= "\t\t" * obsParam * " = getObsOrSdParam(obsPar, mapObsParam)\n" 
        end 

        # Translate the formula for the observable to Julia syntax 
        juliaFormula = peTabFormulaToJulia(tmpFormula, stateNames, paramData, namesParamDyn, namesNonDynParam, namesExplicitRules)
        strObserveble *= "\t\t" * "return " * juliaFormula * "\n"
        strObserveble *= "\tend\n\n"
    end

    # Extracts the explicit rules that have keys among the observables.
    # Writes them before the observable functions.
    explicitRules = ""
    for (key,value) in modelDict["modelRuleFunctions"]
        if occursin(" " * key * " ", strObserveble)
            explicitRules *= "\t" * key * " = " * value[2] * "\n"
        end
    end
    if length(explicitRules)>0
        explicitRules *= "\n"
        write(io, explicitRules)
    end

    write(io, strObserveble)

    strClose = "end"
    write(io, strClose)
    close(io)
end


"""
    createU0Function(modelName::String, 
                         dirModel::String, 
                         paramData::ParamData, 
                         namesParameter::Array{String, 1}, 
                         stateMap)

    For modelName create a function for computing initial value by translating the stateMap 
    into Julia syntax.

    To correctly create the function the name of all parameters, paramData (to get constant parameters)
    are required.
"""
function createU0Function(modelName::String, 
                          dirModel::String, 
                          paramData::ParamData, 
                          namesParameter::Array{String, 1}, 
                          stateMap;
                          inPlace::Bool=true)

    io = open(dirModel * "/" * modelName * "ObsSdU0.jl", "a")

    if inPlace == true
        write(io, "\n\nfunction evalU0!(u0Vec, paramVec) \n\n")
    else
        write(io, "\n\nfunction evalU0(paramVec) \n\n")
    end

    # Extract all model parameters (constant and none-constant) and write to file 
    paramDynStr = "\t"
    for i in eachindex(namesParameter)
        paramDynStr *= namesParameter[i] * ", "
    end
    paramDynStr = paramDynStr[1:end-2]
    paramDynStr *= " = paramVec \n\n"
    write(io, paramDynStr)    

    # Write the formula for each initial condition to file 
    stateNames = [replace.(string.(stateMap[i].first), "(t)" => "") for i in eachindex(stateMap)]
    stateExpWrite = ""
    for i in eachindex(stateMap)
        stateName = stateNames[i]
        stateExp = replace(string(stateMap[i].second), " " => "")
        stateFormula = peTabFormulaToJulia(stateExp, stateNames, paramData, namesParameter, String[], String[])
        stateExpWrite *= "\t" * stateName * " = " * stateFormula * "\n"
    end
    write(io, stateExpWrite * "\n")

    # Ensure the states in correct order are written to u0 
    # In place version where we mutate stateVec 
    if inPlace == true
        stateStr = "\tu0Vec .= "
        for i in eachindex(stateNames)
            stateStr *= stateNames[i] * ", "
        end
        stateStr = stateStr[1:end-2]
        write(io, stateStr)

    # Where we return the entire initial value vector 
    elseif inPlace == false
        stateStr = "\t return ["
        for i in eachindex(stateNames)
            stateStr *= stateNames[i] * ", "
        end
        stateStr = stateStr[1:end-2]
        stateStr *= "]"
        write(io, stateStr)
    end

    strClose = "\nend"
    write(io, strClose)
    close(io)
end


"""
    createSdFunction(modelName::String, 
                          dirModel::String, 
                          paramData::ParamData, 
                          stateNames, 
                          namesParamDyn::Array{String, 1}, 
                          observablesData::DataFrame)

    For modelName create a function for computing the standard deviation by translating the observablesData
    PeTab-file into Julia syntax. 

    To correctly create the function the state-names, names of dynamic parameters to estiamte 
    (namesDynParam) and PeTab parameter-file (to get constant parameters) data are needed. 
"""
function createSdFunction(modelName::String, 
                          dirModel::String, 
                          paramData::ParamData, 
                          stateNames, 
                          namesParamDyn::Array{String, 1}, 
                          namesNonDynParam::Array{String, 1},
                          observablesData::DataFrame)


    io = open(dirModel * "/" * modelName * "ObsSdU0.jl", "a")

    write(io, "\n\nfunction evalSd!(u, t, sdPar, dynPar, nonDynParam, paramData, observableId, mapSdParam) \n")

    # Extract names of model states and write to file 
    stateNamesShort = replace.(string.(stateNames), "(t)" => "")
    stateStr = "\n\t"
    for i in eachindex(stateNamesShort)
        stateStr *= stateNamesShort[i] * ", "
    end
    stateStr = stateStr[1:end-2]
    stateStr *= "= u \n"
    write(io, stateStr)

    # Extract name of dynamic parameter and write to file
    paramDynStr = "\t"
    for i in eachindex(namesParamDyn)
        paramDynStr *= namesParamDyn[i] * ", "
    end
    paramDynStr = paramDynStr[1:end-2]
    paramDynStr *= " = dynPar \n"
    write(io, paramDynStr)

    # Extract name of non-dynamic parameter and write to file
    if !isempty(namesNonDynParam)
        paramNonDynStr = "\t"
        for i in eachindex(namesNonDynParam)
            paramNonDynStr *= namesNonDynParam[i] * ", "
        end
        paramNonDynStr = paramNonDynStr[1:end-2]
        paramNonDynStr *= " = nonDynParam \n"
        write(io, paramNonDynStr)
    end

    # Extract constant parameters. To avoid cluttering the function only constant parameters that are used to 
    # to compute yMod are written to file.
    paramConstStr = ""
    for i in eachindex(paramData.parameterID)
        if paramData.shouldEst[i] == false
            paramConstStr *= "\t" * paramData.parameterID[i] * "_C = paramData.paramVal[" * string(i) *"] \n" 
        end
    end
    paramConstStr *= "\n"
    write(io, paramConstStr)

    # Write the formula for standard deviations to file
    observableIDs = String.(observablesData[!, "observableId"])
    strObserveble = ""
    for i in eachindex(observableIDs)
        # Each observebleID falls below its own if-statement 
        strObserveble *= "\tif observableId == " * "\"" * observableIDs[i] * "\"" * " \n"
        tmpFormula = filter(x -> !isspace(x), String(observablesData[i, "noiseFormula"]))

        # Extract noise parameters 
        noiseParam = getNoiseParamStr(tmpFormula)
        if !isempty(noiseParam)
            strObserveble *= "\t\t" * noiseParam * " = getObsOrSdParam(sdPar, mapSdParam)\n" 
        end 

        juliaFormula = peTabFormulaToJulia(tmpFormula, stateNames, paramData, namesParamDyn, namesNonDynParam, String[])
        strObserveble *= "\t\t" * "return " * juliaFormula * "\n"
        strObserveble *= "\tend\n\n"
    end
    write(io, strObserveble)

    strClose = "end"
    write(io, strClose)
    close(io)
end


"""
    peTabFormulaToJulia(formula::String, stateNames, paramData::ParamData, namesParamDyn::Array{String, 1}, namesNonDynParam::Array{String, 1}, namesExplicitRules::Array{String, 1})::String
    Translate a peTab formula (e.g for observable or for sd-parameter) into Julia syntax and output the result 
    as a string.

    State-names, namesParamDyn and paramData are all required to correctly identify states and parameters in the formula.
    namesExplicitRules is optional and is only set if there are any explicit rules in the SBML-file.
"""
function peTabFormulaToJulia(formula::String, stateNames, paramData::ParamData, namesParamDyn::Array{String, 1}, namesNonDynParam::Array{String, 1}, namesExplicitRules::Array{String, 1})::String

    # Characters directly translate to Julia and characters that also are assumed to terminate a word (e.g state and parameter)
    charDirectTranslate = ['(', ')', '+', '-', '/', '*', '^'] 
    lenFormula = length(formula)
    formulaJulia = ""
    i = 1
    while i <= lenFormula
        # In case character i of the string can be translated directly 
        if formula[i] in charDirectTranslate
            formulaJulia *= formula[i] * " "
            i += 1

        # In case character i cannot be translated directly (is part of a word)
        else
            # Get word (e.g param, state, math-operation or number)
            word, iNew = getWord(formula, i, charDirectTranslate)
            # Translate word to Julia syntax 
            formulaJulia *= wordToJuliaSyntax(word, stateNames, paramData, namesParamDyn, namesNonDynParam, namesExplicitRules)
            i = iNew

            # Special case where we have multiplication
            if isNumber(word) && i <= lenFormula && isletter(formula[i])
                formulaJulia *= "* "
            end
        end
    end

    return formulaJulia
end


"""
    getWord(str::String, iStart, charListTerm)

    In a string starting from position iStart extract the next "word", which is the longest 
    concurent occurance of characters that are not in the character list with word termination 
    characters. Returns the word and iEnd (the position where the word ends).
    
    For example, if charListTerm = ['(', ')', '+', '-', '/', '*', '^'] abc123 is 
    considered a word but not abc123*.  
"""
function getWord(str::String, iStart::Int, charListTerm::Array{Char, 1})
    
    wordStr = ""
    iEnd = iStart

    # If the first character is a numberic the termination occurs when 
    # the first non-numeric character (not digit or dot .) is reached. 
    isNumericStart = isnumeric(str[iEnd])

    while iEnd <= length(str)
        if !(str[iEnd] in charListTerm) 

            # Parase sciencetific notation for number 
            if isNumericStart == true && str[iEnd] == 'e'
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

            if isNumericStart == true && !(isnumeric(str[iEnd]) || str[iEnd] == '.')
                break
            end
            wordStr *= str[iEnd]
        else
            break 
        end
        iEnd += 1
    end

    return wordStr, iEnd
end


""""
    wordToJuliaSyntax(wordTranslate::String, 
                           stateNames,
                           paramData::ParamData, 
                           namesParamDyn::Array{String, 1},
                           namesExplicitRules::Array{String, 1})::String

    Translate a word (state, parameter, math-expression or number) into Julia syntax 
    when building Ymod, U0 and Sd functions.
    namesExplicitRules is optional and is only set if there are any explicit rules in the SBML-file.

"""
function wordToJuliaSyntax(wordTranslate::String, 
                           stateNames,
                           paramData::ParamData, 
                           namesParamDyn::Array{String, 1}, 
                           namesNonDynParam::Array{String, 1},
                           namesExplicitRules::Array{String, 1})::String

    # List of mathemathical operations that are accpeted and will be translated 
    # into Julia syntax (t is assumed to be time)
    listOperations = ["exp", "sin", "cos", "t"]

    stateNamesStr = replace.(string.(stateNames), "(t)" => "")
    wordJuliaSyntax = ""
    
    # If wordTranslate is a constant parameter (is not paramDyn - list of parameter to estimate)
    if wordTranslate in paramData.parameterID && !(wordTranslate in namesParamDyn) && !(wordTranslate in namesNonDynParam)
        # Constant parameters get a _C appended to tell them apart 
        wordJuliaSyntax *= wordTranslate * "_C"
    end

    # If wordTranslate is a dynamic parameters 
    if wordTranslate in namesParamDyn
        wordJuliaSyntax *= wordTranslate
    end

    if wordTranslate in namesNonDynParam
        wordJuliaSyntax *= wordTranslate
    end

    if wordTranslate in namesExplicitRules
        wordJuliaSyntax *= wordTranslate
    end

    if wordTranslate in stateNamesStr
        wordJuliaSyntax *= wordTranslate
    end

    if isNumber(wordTranslate)
        wordJuliaSyntax *= wordTranslate
    end

    if wordTranslate in listOperations
        wordJuliaSyntax *= listOperations[wordTranslate .== listOperations]
    end

    # If word is one of the observable parameters 
    if length(wordTranslate) >= 19 && wordTranslate[1:19] == "observableParameter"
        wordJuliaSyntax *= wordTranslate
    end

    # If word is one of the noise parameters 
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
    getObsParamStr(measurmentFormula::String)::String

    Helper function to extract all observableParameter in the observableFormula in the PeTab-file. 
"""
function getObsParamStr(measurmentFormula::String)::String
    
    # Find all words on the form observableParameter
    obsWords = sort(unique([ match.match for match in eachmatch(r"observableParameter[0-9]_\w+", measurmentFormula) ]))
    obsWordStr = ""
    for i in eachindex(obsWords)
        if i != length(obsWords) 
            obsWordStr *= obsWords[i] * ", "
        else
            obsWordStr *= obsWords[i] 
        end
    end

    return obsWordStr
end


"""
    getNoiseParamStr(sdFormula::String)::String

    Helper function to extract all the noiseParameter in noiseParameter formula in the PeTab file.
"""
function getNoiseParamStr(sdFormula::String)::String
    
    # Find all words on the form observableParameter
    sdWords = [ match.match for match in eachmatch(r"noiseParameter[0-9]_\w+", sdFormula) ]
    sdWordStr = ""
    for i in eachindex(sdWords)
        if i != length(sdWords) 
            sdWordStr *= sdWords[i] * ", "
        else
            sdWordStr *= sdWords[i] 
        end
    end

    return sdWordStr
end
