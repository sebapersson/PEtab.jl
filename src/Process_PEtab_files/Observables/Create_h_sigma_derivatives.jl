
"""
    createFileDYmodSdU0(modelName::String,
                       dirModel::String,
                       odeSys::ODESystem,
                       stateMap,
                       SBMLDict::Dict)

    For a PeTab model with name modelName with all PeTab-files in dirModel and associated
    ModellingToolkit ODESystem (with its stateMap) build a file containing a functions for
    i) computing the observable model value (yMod) ii) compute the initial value u0 (by using the
    stateMap) and iii) computing the standard error (sd) for each observableFormula in the
    observables PeTab file.
    Note - The produced Julia file will go via the JIT-compiler.
"""
function createDerivative_σ_h_File(modelName::String,
                                   pathYAMl::String,
                                   dirJulia::String,
                                   odeSystem::ODESystem,
                                   parameterMap,
                                   stateMap,
                                   SBMLDict::Dict;
                                   jlFile::Bool=false,
                                   customParameterValues::Union{Nothing, Dict}=nothing, 
                                   writeToFile::Bool=true)

    pODEProblemNames = string.(parameters(odeSystem))
    modelStateNames = replace.(string.(states(odeSystem)), "(t)" => "")

    experimentalConditions, measurementsData, parametersData, observablesData = readPEtabFiles(pathYAMl, jlFile = jlFile)
    parameterInfo = processParameters(parametersData, customParameterValues=customParameterValues)
    measurementInfo = processMeasurements(measurementsData, observablesData)

    # Indices for keeping track of parameters in θ
    θ_indices = computeIndicesθ(parameterInfo, measurementInfo, odeSystem, parameterMap, stateMap, experimentalConditions)

    ∂h∂uStr, ∂h∂pStr = create∂h∂_Function(modelName, dirJulia, modelStateNames, parameterInfo, pODEProblemNames, string.(θ_indices.θ_nonDynamicNames), observablesData, SBMLDict, writeToFile)
    ∂σ∂uStr, ∂σ∂pStr = create∂σ∂_Function(modelName, dirJulia, parameterInfo, modelStateNames, pODEProblemNames, string.(θ_indices.θ_nonDynamicNames), observablesData, SBMLDict, writeToFile)
    
    return ∂h∂uStr, ∂h∂pStr, ∂σ∂uStr, ∂σ∂pStr
end


"""
    create∂h∂_Function(modelName::String,
                       dirModel::String,
                       modelStateNames::Vector{String},
                       parameterInfo::ParametersInfo,
                       pODEProblemNames::Vector{String},
                       θ_nonDynamicNames::Vector{String},
                       observablesData::CSV.File,
                       SBMLDict::Dict)

    For modelName create using Symbolics function for computing ∂h/∂u and ∂h/∂p where
    u = modelStates and p = pODEProblem (parameters for ODE problem)
"""
function create∂h∂_Function(modelName::String,
                            dirModel::String,
                            modelStateNames::Vector{String},
                            parameterInfo::ParametersInfo,
                            pODEProblemNames::Vector{String},
                            θ_nonDynamicNames::Vector{String},
                            observablesData::CSV.File,
                            SBMLDict::Dict, 
                            writeToFile::Bool)

    pathSave = joinpath(dirModel, modelName * "_D_h_sd.jl")
    io1 = IOBuffer()
    io2 = IOBuffer()

    modelStateStr, pODEProblemStr, θ_nonDynamicStr = createTopOf∂h∂_Function(modelStateNames, pODEProblemNames,
                                                                             θ_nonDynamicNames, observablesData)

    # Store the formula of each observable in string
    observableIds = string.(observablesData[:observableId])
    pObservebleStr = ""
    uObservebleStr = ""
    for i in eachindex(observableIds)

        # Each observebleID falls below its own if-statement
        pObservebleStr *= "\tif observableId == " * ":" * observableIds[i] * "" * " \n"
        uObservebleStr *= "\tif observableId == " * ":" * observableIds[i] * "" * " \n"

        _formula = filter(x -> !isspace(x), string(observablesData[:observableFormula][i]))
        formula = replaceExplicitVariableWithRule(_formula, SBMLDict)
        juliaFormula = petabFormulaToJulia(formula, modelStateNames, parameterInfo, pODEProblemNames, θ_nonDynamicNames)

        enterObservable = true # Only extract observable parameter once
        for iState in eachindex(modelStateNames)
            if occursin(Regex("\\b" * modelStateNames[iState] * "\\b"), juliaFormula)
                # Extract observable parameters
                observableParameters = getObservableParametersStr(formula)
                if !isempty(observableParameters) && enterObservable == true
                    uObservebleStr *= "\t\t" * observableParameters * " = getObsOrSdParam(θ_observable, parameterMap)\n"
                    enterObservable = false
                end

                juliaFormulaSymbolic =  eval(Meta.parse(juliaFormula))
                uiSymbolic = eval(Meta.parse(modelStateNames[iState]))
                _∂h∂ui = string(Symbolics.derivative(juliaFormulaSymbolic, uiSymbolic; simplify=true))
                ∂h∂ui = replaceVariablesWithArrayIndex(_∂h∂ui, modelStateNames, parameterInfo, pODEProblemNames, θ_nonDynamicNames, pODEProblem=true)

                uObservebleStr *= "\t\tout[" * string(iState) * "] = " * ∂h∂ui * "\n"
            end
        end

        enterObservable = true # Only extract observable parameter once
        for ip in eachindex(pODEProblemNames)
            if occursin(Regex("\\b" * pODEProblemNames[ip] * "\\b"), juliaFormula)
                # Extract observable parameters
                observableParameters = getObservableParametersStr(formula)
                if !isempty(observableParameters) && enterObservable == true
                    pObservebleStr *= "\t\t" * observableParameters * " = getObsOrSdParam(θ_observable, parameterMap)\n"
                    enterObservable = false
                end

                juliaFormulaSymbolic =  eval(Meta.parse(juliaFormula))
                piSymbolic = eval(Meta.parse(pODEProblemNames[ip]))
                _∂h∂pi = string(Symbolics.derivative(juliaFormulaSymbolic, piSymbolic; simplify=true))
                ∂h∂pi = replaceVariablesWithArrayIndex(_∂h∂pi, modelStateNames, parameterInfo, pODEProblemNames, θ_nonDynamicNames, pODEProblem=true)
                pObservebleStr *= "\t\tout[" * string(ip) * "] = " * ∂h∂pi * "\n"
            end
        end

        uObservebleStr *= "\t\t" * "return nothing\n" * "\tend\n\n"
        pObservebleStr *= "\t\t" * "return nothing\n" * "\tend\n\n"
    end


    if writeToFile == true
        write(io1, modelStateStr)
        write(io1, pODEProblemStr)
        write(io1, θ_nonDynamicStr)
        write(io1, "\n")
    end
    write(io1, "function compute_∂h∂u!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) \n")
    write(io1, uObservebleStr)
    write(io1, "end")
    ∂h∂uStr = String(take!(io1))
    if writeToFile
        strWrite = ∂h∂uStr * "\n\n"
        open(pathSave, "w") do f
            write(f, strWrite)
        end
    end

    write(io2, "function compute_∂h∂p!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) \n")
    write(io2, pObservebleStr)
    write(io2, "end")
    ∂h∂pStr = String(take!(io2))
    if writeToFile
        strWrite = ∂h∂pStr * "\n\n"
        open(pathSave, "a") do f
            write(f, strWrite)
        end
    end
    close(io1)
    close(io2)

    return ∂h∂uStr, ∂h∂pStr
end


"""
    createTopOf∂h∂_Function(modelStateNames::Vector{String},
                            pODEProblemNames::Vector{String},
                            θ_nonDynamicNames::Vector{String},
                            observablesData::CSV.File)

    Extracts all variables needed for the functions and add them as variables for Symbolics.
"""
function createTopOf∂h∂_Function(modelStateNames::Vector{String},
                                 pODEProblemNames::Vector{String},
                                 θ_nonDynamicNames::Vector{String},
                                 observablesData::CSV.File)

    # We formulate the string in a format accepatble for symbolics so that we later
    # can differentative the observable function with respect to the h formula
    variablesStr = "@variables "

    modelStateStr = "#"
    for i in eachindex(modelStateNames)
        modelStateStr *= "u[" * string(i) * "] = " * modelStateNames[i] * ", "
        variablesStr *= modelStateNames[i] * ", "
    end
    modelStateStr = modelStateStr[1:end-2] # Remove last non needed ", "
    modelStateStr *= "\n"

    # Extract name of dynamic parameter
    pODEProblemStr = "#"
    for i in eachindex(pODEProblemNames)
        pODEProblemStr *= "pODEProblem[" * string(i) * "] = " * pODEProblemNames[i] * ", "
        variablesStr *= pODEProblemNames[i] * ", "
    end
    pODEProblemStr = pODEProblemStr[1:end-2]
    pODEProblemStr *= "\n"

    θ_nonDynamicStr = "#"
    if !isempty(θ_nonDynamicNames)
        for i in eachindex(θ_nonDynamicNames)
            θ_nonDynamicStr *= "θ_nonDynamic[" * string(i)* "] = " * θ_nonDynamicNames[i] * ", "
            variablesStr *= θ_nonDynamicNames[i] * ", "
        end
        θ_nonDynamicStr = θ_nonDynamicStr[1:end-2] # Remove last non needed ", "
        θ_nonDynamicStr *= "\n"
    end

    # Extracts all observable- and noise-parameters to add them to the symbolics variable string
    observableIds = string.(observablesData[:observableId])
    for i in eachindex(observableIds)

        # Extract observable parameters
        _formula = filter(x -> !isspace(x), string(observablesData[:observableFormula][i]))
        observableParameters = getObservableParametersStr(_formula)
        if !isempty(observableParameters)
            variablesStr *= observableParameters * ", "
        end

        # Extract noise parameters
        _formula = filter(x -> !isspace(x), string(observablesData[:noiseFormula][i]))
        noiseParameters = getNoiseParametersStr(_formula)
        if !isempty(noiseParameters)
            variablesStr *= noiseParameters * ", "
        end
    end

    # Remove last "," and Run @variables ... string to bring symbolic parameters into the scope
    variablesStr = variablesStr[1:end-2]
    eval(Meta.parse(variablesStr))

    return modelStateStr, pODEProblemStr, θ_nonDynamicStr
end


"""
    create∂σ∂_Function(modelName::String,
                            dirModel::String,
                            parameterInfo::ParametersInfo,
                            modelStateNames::Vector{String},
                            pODEProblemNames::Vector{String},
                            θ_nonDynamicNames::Vector{String},
                            observablesData::CSV.File,
                            SBMLDict::Dict)

    For modelName create a function for computing the standard deviation by translating the observablesData
"""
function create∂σ∂_Function(modelName::String,
                            dirModel::String,
                            parameterInfo::ParametersInfo,
                            modelStateNames::Vector{String},
                            pODEProblemNames::Vector{String},
                            θ_nonDynamicNames::Vector{String},
                            observablesData::CSV.File,
                            SBMLDict::Dict, 
                            writeToFile::Bool)

    pathSave = joinpath(dirModel, modelName * "_D_h_sd.jl")
    io1 = IOBuffer()
    io2 = IOBuffer()

    observableIds = string.(observablesData[:observableId])
    pObservebleStr = ""
    uObservebleStr = ""
    for i in eachindex(observableIds)

        # Each observebleID falls below its own if-statement
        pObservebleStr *= "\tif observableId == " * ":" * observableIds[i] * "" * " \n"
        uObservebleStr *= "\tif observableId == " * ":" * observableIds[i] * "" * " \n"

        _formula = filter(x -> !isspace(x), string(observablesData[:noiseFormula][i]))
        formula = replaceExplicitVariableWithRule(_formula, SBMLDict)
        juliaFormula = petabFormulaToJulia(formula, modelStateNames, parameterInfo, pODEProblemNames, θ_nonDynamicNames)

        enterObservable = true
        for iState in eachindex(modelStateNames)
            if occursin(Regex("\\b" * modelStateNames[iState] * "\\b"), juliaFormula)

                noiseParameters = getNoiseParametersStr(formula)
                if !isempty(noiseParameters) && enterObservable == true
                    uObservebleStr *= "\t\t" * noiseParameters * " = getObsOrSdParam(θ_sd, parameterMap)\n"
                    enterObservable = false
                end

                juliaFormulaSymbolic =  eval(Meta.parse(juliaFormula))
                uiSymbolic = eval(Meta.parse(modelStateNames[iState]))
                _∂σ∂ui = string(Symbolics.derivative(juliaFormulaSymbolic, uiSymbolic; simplify=true))
                ∂σ∂ui = replaceVariablesWithArrayIndex(_∂σ∂ui, modelStateNames, parameterInfo, pODEProblemNames, θ_nonDynamicNames, pODEProblem=true)
                uObservebleStr *= "\t\tout[" * string(iState) * "] = " * ∂σ∂ui * "\n"
            end
        end

        enterObservable = true
        for ip in eachindex(pODEProblemNames)
            if occursin(Regex("\\b" * pODEProblemNames[ip] * "\\b"), juliaFormula)

                noiseParameters = getNoiseParametersStr(formula)
                if !isempty(noiseParameters) && enterObservable == true
                    pObservebleStr *= "\t\t" * noiseParameters * " = getObsOrSdParam(θ_sd, parameterMap)\n"
                    enterObservable = false
                end

                juliaFormulaSymbolic =  eval(Meta.parse(juliaFormula))
                piSymbolic = eval(Meta.parse(pODEProblemNames[ip]))
                _∂σ∂pi = string(Symbolics.derivative(juliaFormulaSymbolic, piSymbolic; simplify=true))
                ∂σ∂pi = replaceVariablesWithArrayIndex(_∂σ∂pi, modelStateNames, parameterInfo, pODEProblemNames, θ_nonDynamicNames, pODEProblem=true)
                pObservebleStr *= "\t\tout[" * string(ip) * "] = " * ∂σ∂pi * "\n"
            end
        end

        uObservebleStr *= "\t\t" * "return nothing\n" * "\tend\n\n"
        pObservebleStr *= "\t\t" * "return nothing\n" * "\tend\n\n"
    end

    write(io1, "function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) \n")
    write(io1, uObservebleStr)
    write(io1, "end")
    ∂σ∂σuStr = String(take!(io1))
    if writeToFile == true
        strWrite = ∂σ∂σuStr * "\n\n"
        open(pathSave, "a") do f
            write(f, strWrite)
        end
    end

    write(io2, "function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) \n")
    write(io2, pObservebleStr)
    write(io2, "end")
    ∂σ∂σpStr = String(take!(io2))
    if writeToFile == true
        strWrite = ∂σ∂σpStr * "\n\n"
        open(pathSave, "a") do f
            write(f, strWrite)
        end
    end

    close(io1)
    close(io2)
    return ∂σ∂σuStr, ∂σ∂σpStr
end
