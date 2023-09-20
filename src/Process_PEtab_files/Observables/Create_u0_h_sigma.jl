# TODO: Refactor yMod and SD functions to avoid redundant code


"""
    create_σ_h_u0_File(model_name::String,
                       dir_model::String,
                       odeSystem::ODESystem,
                       state_map,
                       SBMLDict::Dict;
                       verbose::Bool=false)

    For a PeTab model with name model_name with all PeTab-files in dir_model and associated
    ModellingToolkit ODESystem (with its state_map) build a file containing a functions for
    i) computing the observable model value (h) ii) compute the initial value u0 (by using the
    state_map) and iii) computing the standard error (σ) for each observableFormula in the
    observables PeTab file.

    Note - The produced Julia file will go via the JIT-compiler. The SBML-dict is needed as
    sometimes variables are encoded via explicit-SBML rules.
"""
function create_σ_h_u0_File(model_name::String,
                            pathYAMl::String,
                            dir_julia::String,
                            odeSystem::ODESystem,
                            parameter_map,
                            state_map,
                            SBMLDict::Dict;
                            jlFile::Bool=false,
                            custom_parameter_values::Union{Nothing, Dict}=nothing, 
                            write_to_file::Bool=true)

    pODEProblemNames = string.(parameters(odeSystem))
    modelStateNames = replace.(string.(states(odeSystem)), "(t)" => "")

    experimentalConditions, measurementsData, parametersData, observablesData = readPEtabFiles(pathYAMl, jlFile=jlFile)
    parameterInfo = processParameters(parametersData, custom_parameter_values=custom_parameter_values)
    measurementInfo = processMeasurements(measurementsData, observablesData)

    # Indices for keeping track of parameters in θ
    θ_indices = computeIndicesθ(parameterInfo, measurementInfo, odeSystem, parameter_map, state_map, experimentalConditions)

    hStr = create_h_Function(model_name, dir_julia, modelStateNames, parameterInfo, pODEProblemNames,
                             string.(θ_indices.θ_nonDynamicNames), observablesData, SBMLDict, write_to_file)

    u0!Str = create_u0_Function(model_name, dir_julia, parameterInfo, pODEProblemNames, state_map, write_to_file, SBMLDict, inPlace=true)

    u0Str = create_u0_Function(model_name, dir_julia, parameterInfo, pODEProblemNames, state_map, write_to_file, SBMLDict, inPlace=false)

    σStr = create_σ_Function(model_name, dir_julia, parameterInfo, modelStateNames, pODEProblemNames, string.(θ_indices.θ_nonDynamicNames), observablesData, SBMLDict, write_to_file)

    return hStr, u0!Str, u0Str, σStr
end
"""
    When parsed from Julia input.
"""
function create_σ_h_u0_File(model_name::String,
                            system,
                            experimentalConditions::CSV.File,
                            measurementsData::CSV.File,
                            parametersData::CSV.File,
                            observablesData::CSV.File,
                            state_map)::NTuple{4, String}

    pODEProblemNames = string.(parameters(system))
    modelStateNames = replace.(string.(states(system)), "(t)" => "")
    parameter_map = [p => 0.0 for p in parameters(system)]

    parameterInfo = PEtab.processParameters(parametersData)
    measurementInfo = PEtab.processMeasurements(measurementsData, observablesData)

    # Indices for keeping track of parameters in θ
    θ_indices = PEtab.computeIndicesθ(parameterInfo, measurementInfo, system, parameter_map, state_map, experimentalConditions)

    # Dummary variables to keep PEtab importer happy even as we are not providing any PEtab files
    SBMLDict = Dict(); SBMLDict["assignmentRulesStates"] = Dict()

    hStr = PEtab.create_h_Function(model_name, @__DIR__, modelStateNames, parameterInfo, pODEProblemNames,
                                   string.(θ_indices.θ_nonDynamicNames), observablesData, SBMLDict, false)
    u0!Str = PEtab.create_u0_Function(model_name, @__DIR__, parameterInfo, pODEProblemNames, state_map, false,
                                      SBMLDict, inPlace=true)
    u0Str = PEtab.create_u0_Function(model_name, @__DIR__, parameterInfo, pODEProblemNames, state_map, false,
                                     SBMLDict, inPlace=false)
    σStr = PEtab.create_σ_Function(model_name, @__DIR__, parameterInfo, modelStateNames, pODEProblemNames,
                                   string.(θ_indices.θ_nonDynamicNames), observablesData, SBMLDict, false)

    return hStr, u0!Str, u0Str, σStr
end


"""
    create_h_Function(model_name::String,
                      dir_model::String,
                      modelStateNames::Vector{String},
                      paramData::ParametersInfo,
                      namesParamDyn::Vector{String},
                      namesNonDynParam::Vector{String},
                      observablesData::CSV.File,
                      SBMLDict::Dict)

    For model_name create a function for computing yMod by translating the observablesData
    PeTab-file into Julia syntax.
"""
function create_h_Function(model_name::String,
                           dir_model::String,
                           modelStateNames::Vector{String},
                           parameterInfo::ParametersInfo,
                           pODEProblemNames::Vector{String},
                           θ_nonDynamicNames::Vector{String},
                           observablesData::CSV.File,
                           SBMLDict::Dict, 
                           write_to_file::Bool)

    io = IOBuffer()
    pathSave = joinpath(dir_model, model_name * "_h_sd_u0.jl")
    modelStateStr, θ_dynamicStr, θ_nonDynamicStr, constantParametersStr = createTopOfFunction_h(modelStateNames, parameterInfo,
                                                                                                pODEProblemNames, θ_nonDynamicNames)

    # Write the formula for each observable in Julia syntax
    observableIds = string.(observablesData[:observableId])
    observableStr = ""
    for i in eachindex(observableIds)

        _formula = filter(x -> !isspace(x), string(observablesData[:observableFormula][i]))
        observableParameters = getObservableParametersStr(_formula)
        observableStr *= "\tif observableId === " * ":" * observableIds[i] * " \n"
        if !isempty(observableParameters)
            observableStr *= "\t\t" * observableParameters * " = getObsOrSdParam(θ_observable, parameter_map)\n"
        end

        formula = replaceExplicitVariableWithRule(_formula, SBMLDict)

        # Translate the formula for the observable to Julia syntax
        _juliaFormula = petabFormulaToJulia(formula, modelStateNames, parameterInfo, pODEProblemNames, θ_nonDynamicNames)
        juliaFormula = replaceVariablesWithArrayIndex(_juliaFormula, modelStateNames, parameterInfo, pODEProblemNames, θ_nonDynamicNames, pODEProblem=true)
        observableStr *= "\t\t" * "return " * juliaFormula * "\n" * "\tend\n\n"
    end

    # Create h function
    if write_to_file == true
        write(io, modelStateStr)
        write(io, θ_dynamicStr)
        write(io, θ_nonDynamicStr)
        write(io, constantParametersStr)
        write(io, "\n")
    end
    write(io, "function compute_h(u::AbstractVector, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                   θ_nonDynamic::AbstractVector, parameterInfo::ParametersInfo, observableId::Symbol,
                      parameter_map::θObsOrSdParameterMap)::Real \n")
    write(io, observableStr)
    write(io, "end")
    hStr = String(take!(io))
    if write_to_file == true
        strWrite = hStr * "\n\n"
        open(pathSave, "w") do f
            write(f, strWrite)
        end
    end
    close(io)
    return hStr    
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
                               pODEProblemNames::Vector{String},
                               θ_nonDynamicNames::Vector{String})

    modelStateStr = "#"
    for i in eachindex(modelStateNames)
        modelStateStr *= "u[" * string(i) * "] = " * modelStateNames[i] * ", "
    end
    modelStateStr = modelStateStr[1:end-2] # Remove last non needed ", "
    modelStateStr *= "\n"

    θ_dynamicStr = "#"
    for i in eachindex(pODEProblemNames)
        θ_dynamicStr *= "pODEProblemNames[" * string(i) * "] = " * pODEProblemNames[i] * ", "
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
    create_u0_Function(model_name::String,
                       dir_model::String,
                       parameterInfo::ParametersInfo,
                       pODEProblemNames::Vector{String},
                       state_map, 
                       SBMLDict;
                       inPlace::Bool=true)

    For model_name create a function for computing initial value by translating the state_map
    into Julia syntax.

    To correctly create the function the name of all parameters, paramData (to get constant parameters)
    are required.
"""
function create_u0_Function(model_name::String,
                            dir_model::String,
                            parameterInfo::ParametersInfo,
                            pODEProblemNames::Vector{String},
                            state_map,
                            write_to_file::Bool, 
                            SBMLDict;
                            inPlace::Bool=true)

    pathSave = joinpath(dir_model, model_name * "_h_sd_u0.jl")                            
    io = IOBuffer()

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

    write(io, "\tt = 0.0 # u at time zero\n\n")

    # Write the formula for each initial condition to file
    _modelStateNames = [replace.(string.(state_map[i].first), "(t)" => "") for i in eachindex(state_map)]
    modelStateNames = filter(x -> x ∉ string.(keys(SBMLDict["assignmentRulesStates"])), _modelStateNames)
    modelStateStr = ""
    for i in eachindex(modelStateNames)
        stateName = modelStateNames[i]
        _stateExpression = replace(string(state_map[i].second), " " => "")
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

    write(io, "\nend")
    u0Str = String(take!(io))
    if write_to_file == true
        strWrite = u0Str * "\n\n"
        open(pathSave, "a") do f
            write(f, strWrite)
        end
    end    
    close(io)

    return u0Str
end


"""
    create_σ_Function(model_name::String,
                      dir_model::String,
                      parameterInfo::ParametersInfo,
                      modelStateNames::Vector{String},
                      pODEProblemNames::Vector{String},
                      θ_nonDynamicNames::Vector{String},
                      observablesData::CSV.File,
                      SBMLDict::Dict)

    For model_name create a function for computing the standard deviation σ by translating the observablesData
    PeTab-file into Julia syntax.
"""
function create_σ_Function(model_name::String,
                           dir_model::String,
                           parameterInfo::ParametersInfo,
                           modelStateNames::Vector{String},
                           pODEProblemNames::Vector{String},
                           θ_nonDynamicNames::Vector{String},
                           observablesData::CSV.File,
                           SBMLDict::Dict, 
                           write_to_file::Bool)

    pathSave = joinpath(dir_model, model_name * "_h_sd_u0.jl")
    io = IOBuffer()

    # Write the formula for standard deviations to file
    observableIds = string.(observablesData[:observableId])
    observableStr = ""
    for i in eachindex(observableIds)

        _formula = filter(x -> !isspace(x), string(observablesData[:noiseFormula][i]))
        noiseParameters = getNoiseParametersStr(_formula)
        observableStr *= "\tif observableId === " * ":" * observableIds[i] * " \n"
        if !isempty(noiseParameters)
            observableStr *= "\t\t" * noiseParameters * " = getObsOrSdParam(θ_sd, parameter_map)\n"
        end

        formula = replaceExplicitVariableWithRule(_formula, SBMLDict)

        # Translate the formula for the observable to Julia syntax
        _juliaFormula = petabFormulaToJulia(formula, modelStateNames, parameterInfo, pODEProblemNames, θ_nonDynamicNames)
        juliaFormula = replaceVariablesWithArrayIndex(_juliaFormula, modelStateNames, parameterInfo, pODEProblemNames, θ_nonDynamicNames, pODEProblem=true)
        observableStr *= "\t\t" * "return " * juliaFormula * "\n" * "\tend\n\n"
    end

    write(io, "function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                   parameterInfo::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap)::Real \n")
    write(io, observableStr)
    write(io, "\nend")
    σStr = String(take!(io))
    if write_to_file == true
        strWrite = σStr * "\n\n"
        open(pathSave, "a") do f
            write(f, strWrite)
        end
    end    
    close(io)

    return σStr
end
