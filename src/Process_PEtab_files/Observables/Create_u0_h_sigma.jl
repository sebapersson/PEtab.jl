# TODO: Refactor yMod and SD functions to avoid redundant code


"""
    create_σ_h_u0_File(modelName::String,
                       dirModel::String,
                       odeSystem::ODESystem,
                       stateMap,
                       SBMLDict::Dict;
                       verbose::Bool=false)

    For a PeTab model with name modelName with all PeTab-files in dirModel and associated
    ModellingToolkit ODESystem (with its stateMap) build a file containing a functions for
    i) computing the observable model value (h) ii) compute the initial value u0 (by using the
    stateMap) and iii) computing the standard error (σ) for each observableFormula in the
    observables PeTab file.

    Note - The produced Julia file will go via the JIT-compiler. The SBML-dict is needed as
    sometimes variables are encoded via explicit-SBML rules.
"""
function create_σ_h_u0_File(modelName::String,
                            pathYAMl::String,
                            dirJulia::String, 
                            odeSystem::ODESystem,
                            stateMap,
                            SBMLDict::Dict;
                            verbose::Bool=true,
                            jlFile::Bool=false)

    pODEProblemNames = string.(parameters(odeSystem))
    modelStateNames = replace.(string.(states(odeSystem)), "(t)" => "")

    experimentalConditions, measurementsData, parametersData, observablesData = readPEtabFiles(pathYAMl, jlFile=jlFile)
    parameterInfo = processParameters(parametersData)
    measurementInfo = processMeasurements(measurementsData, observablesData)

    # Indices for keeping track of parameters in θ
    θ_indices = computeIndicesθ(parameterInfo, measurementInfo, odeSystem, experimentalConditions)

    create_h_Function(modelName, dirJulia, modelStateNames, parameterInfo, string.(θ_indices.θ_dynamicNames),
                      string.(θ_indices.θ_nonDynamicNames), observablesData, SBMLDict)
    verbose == true && @printf("Done with h function\n")

    create_u0_Function(modelName, dirJulia, parameterInfo, pODEProblemNames, stateMap, inPlace=true)
    verbose == true && @printf("Done with u0 in-place function\n")

    create_u0_Function(modelName, dirJulia, parameterInfo, pODEProblemNames, stateMap, inPlace=false)
    verbose == true && @printf("Done with u0 not in-place function\n")

    create_σ_Function(modelName, dirJulia, parameterInfo, modelStateNames, string.(θ_indices.θ_dynamicNames), string.(θ_indices.θ_nonDynamicNames), observablesData, SBMLDict)
    verbose == true && @printf("Done with σ function\n")
end


"""
    create_h_Function(modelName::String,
                      dirModel::String,
                      modelStateNames::Vector{String},
                      paramData::ParametersInfo,
                      namesParamDyn::Vector{String},
                      namesNonDynParam::Vector{String},
                      observablesData::DataFrame,
                      SBMLDict::Dict)

    For modelName create a function for computing yMod by translating the observablesData
    PeTab-file into Julia syntax.
"""
function create_h_Function(modelName::String,
                           dirModel::String,
                           modelStateNames::Vector{String},
                           parameterInfo::ParametersInfo,
                           θ_dynamicNames::Vector{String},
                           θ_nonDynamicNames::Vector{String},
                           observablesData::DataFrame,
                           SBMLDict::Dict)

    io = open(dirModel * "/" * modelName * "_h_sd_u0.jl", "w")
    modelStateStr, θ_dynamicStr, θ_nonDynamicStr, constantParametersStr = createTopOfFunction_h(modelStateNames, parameterInfo,
                                                                                                θ_dynamicNames, θ_nonDynamicNames)

    # Write the formula for each observable in Julia syntax
    observableIds = string.(observablesData[!, "observableId"])
    observableStr = ""
    for i in eachindex(observableIds)

        _formula = filter(x -> !isspace(x), string(observablesData[i, "observableFormula"]))
        observableParameters = getObservableParametersStr(_formula)
        observableStr *= "\tif observableId == " * ":" * observableIds[i] * " \n"
        if !isempty(observableParameters)
            observableStr *= "\t\t" * observableParameters * " = getObsOrSdParam(θ_observable, parameterMap)\n"
        end

        formula = replaceExplicitVariableWithRule(_formula, SBMLDict)

        # Translate the formula for the observable to Julia syntax
        _juliaFormula = petabFormulaToJulia(formula, modelStateNames, parameterInfo, θ_dynamicNames, θ_nonDynamicNames)
        juliaFormula = replaceVariablesWithArrayIndex(_juliaFormula, modelStateNames, parameterInfo, θ_dynamicNames, θ_nonDynamicNames)
        observableStr *= "\t\t" * "return " * juliaFormula * "\n" * "\tend\n\n"
    end

    # Create h function
    write(io, modelStateStr)
    write(io, θ_dynamicStr)
    write(io, θ_nonDynamicStr)
    write(io, constantParametersStr)
    write(io, "\n")
    write(io, "function compute_h(u::AbstractVector, t::Real, θ_dynamic::AbstractVector, θ_observable::AbstractVector,
                   θ_nonDynamic::AbstractVector, parameterInfo::ParametersInfo, observableId::Symbol,
                      parameterMap::θObsOrSdParameterMap)::Real \n")
    write(io, observableStr)
    write(io, "end\n\n")
    close(io)
end


"""
    createTopOfFunction_h(modelStateNames::Vector{String},
                          paramData::ParametersInfo,
                          namesParamODEProb::Vector{String},
                          namesNonDynParam::Vector{String})

    Extracts all variables needed for the observable h function.
"""
function createTopOfFunction_h(modelStateNames::Vector{String},
                               parameterInfo::ParametersInfo,
                               θ_dynamicNames::Vector{String},
                               θ_nonDynamicNames::Vector{String})

    modelStateStr = "#"
    for i in eachindex(modelStateNames)
        modelStateStr *= "u[" * string(i) * "] = " * modelStateNames[i] * ", "
    end
    modelStateStr = modelStateStr[1:end-2] # Remove last non needed ", "
    modelStateStr *= "\n"

    θ_dynamicStr = "#"
    for i in eachindex(θ_dynamicNames)
        θ_dynamicStr *= "θ_dynamicNames[" * string(i) * "] = " * θ_dynamicNames[i] * ", "
    end
    θ_dynamicStr = θ_dynamicStr[1:end-2] # Remove last non needed ", "
    θ_dynamicStr *= "\n"

    θ_nonDynamicStr = "#"
    if !isempty(θ_nonDynamicNames)
        for i in eachindex(θ_nonDynamicNames)
            θ_nonDynamicStr *= "θ_nonDynamic[" * string(i)* "] = " * θ_nonDynamicNames[i] * ", "
        end
        θ_nonDynamicStr = θ_nonDynamicStr[1:end-2] # Remove last non needed ", "
        θ_nonDynamicStr *= "\n"
    end

    constantParametersStr = ""
    for i in eachindex(parameterInfo.parameterId)
        if parameterInfo.estimate[i] == false
            constantParametersStr *= "#parameterInfo.nominalValue[" * string(i) *"] = " * string(parameterInfo.parameterId[i]) * "_C \n"
        end
    end
    constantParametersStr *= "\n"

    return modelStateStr, θ_dynamicStr, θ_nonDynamicStr, constantParametersStr
end


"""
    create_u0_Function(modelName::String,
                       dirModel::String,
                       parameterInfo::ParametersInfo,
                       pODEProblemNames::Vector{String},
                       stateMap;
                       inPlace::Bool=true)

    For modelName create a function for computing initial value by translating the stateMap
    into Julia syntax.

    To correctly create the function the name of all parameters, paramData (to get constant parameters)
    are required.
"""
function create_u0_Function(modelName::String,
                            dirModel::String,
                            parameterInfo::ParametersInfo,
                            pODEProblemNames::Vector{String},
                            stateMap;
                            inPlace::Bool=true)

    io = open(dirModel * "/" * modelName * "_h_sd_u0.jl", "a")

    if inPlace == true
        write(io, "function compute_u0!(u0::AbstractVector, pODEProblem::AbstractVector) \n\n")
    else
        write(io, "function compute_u0(pODEProblem::AbstractVector)::AbstractVector \n\n")
    end

    # Write named list of parameter to file
    pODEProblemStr = "\t#"
    for i in eachindex(pODEProblemNames)
        pODEProblemStr *= "pODEProblem[" * string(i) * "] = " * pODEProblemNames[i] * ", "
    end
    pODEProblemStr = pODEProblemStr[1:end-2]
    pODEProblemStr *= "\n\n"
    write(io, pODEProblemStr)

    # Write the formula for each initial condition to file
    modelStateNames = [replace.(string.(stateMap[i].first), "(t)" => "") for i in eachindex(stateMap)]
    modelStateStr = ""
    for i in eachindex(stateMap)
        stateName = modelStateNames[i]
        _stateExpression = replace(string(stateMap[i].second), " " => "")
        stateFormula = petabFormulaToJulia(_stateExpression, modelStateNames, parameterInfo, pODEProblemNames, String[])
        for i in eachindex(pODEProblemNames)
            stateFormula = replaceWholeWord(stateFormula, pODEProblemNames[i], "pODEProblem["*string(i)*"]")
        end
        modelStateStr *= "\t" * stateName * " = " * stateFormula * "\n"
    end
    write(io, modelStateStr * "\n")

    # Ensure the states in correct order are written to u0
    if inPlace == true
        modelStateStr = "\tu0 .= "
        for i in eachindex(modelStateNames)
            modelStateStr *= modelStateNames[i] * ", "
        end
        modelStateStr = modelStateStr[1:end-2]
        write(io, modelStateStr)

    # Where we return the entire initial value vector
    elseif inPlace == false
        modelStateStr = "\t return ["
        for i in eachindex(modelStateNames)
            modelStateStr *= modelStateNames[i] * ", "
        end
        modelStateStr = modelStateStr[1:end-2]
        modelStateStr *= "]"
        write(io, modelStateStr)
    end

    write(io, "\nend\n\n")
    close(io)
end


"""
    create_σ_Function(modelName::String,
                      dirModel::String,
                      parameterInfo::ParametersInfo,
                      modelStateNames::Vector{String},
                      θ_dynamicNames::Vector{String},
                      θ_nonDynamicNames::Vector{String},
                      observablesData::DataFrame,
                      SBMLDict::Dict)

    For modelName create a function for computing the standard deviation σ by translating the observablesData
    PeTab-file into Julia syntax.
"""
function create_σ_Function(modelName::String,
                           dirModel::String,
                           parameterInfo::ParametersInfo,
                           modelStateNames::Vector{String},
                           θ_dynamicNames::Vector{String},
                           θ_nonDynamicNames::Vector{String},
                           observablesData::DataFrame,
                           SBMLDict::Dict)

    io = open(dirModel * "/" * modelName * "_h_sd_u0.jl", "a")

    # Write the formula for standard deviations to file
    observableIds = string.(observablesData[!, "observableId"])
    observableStr = ""
    for i in eachindex(observableIds)

        _formula = filter(x -> !isspace(x), string(observablesData[i, "noiseFormula"]))
        noiseParameters = getNoiseParametersStr(_formula)
        observableStr *= "\tif observableId == " * ":" * observableIds[i] * " \n"
        if !isempty(noiseParameters)
            observableStr *= "\t\t" * noiseParameters * " = getObsOrSdParam(θ_sd, parameterMap)\n"
        end

        formula = replaceExplicitVariableWithRule(_formula, SBMLDict)

        # Translate the formula for the observable to Julia syntax
        _juliaFormula = petabFormulaToJulia(formula, modelStateNames, parameterInfo, θ_dynamicNames, θ_nonDynamicNames)
        juliaFormula = replaceVariablesWithArrayIndex(_juliaFormula, modelStateNames, parameterInfo, θ_dynamicNames, θ_nonDynamicNames)
        observableStr *= "\t\t" * "return " * juliaFormula * "\n" * "\tend\n\n"
    end

    write(io, "function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, θ_dynamic::AbstractVector, θ_nonDynamic::AbstractVector,
                   parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap)::Real \n")
    write(io, observableStr)
    write(io, "end")
    close(io)
end
