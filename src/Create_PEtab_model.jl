"""
    readPEtabModel(pathYAML::String;
                   forceBuildJuliaFiles::Bool=false,
                   verbose::Bool=true,
                   ifElseToEvent::Bool=true)::PEtabModel

    Parses a PEtab specified problem with yaml-file at `pathYAML` into a Julia accessible format. 

    When parsing a PEtab problem several things happens under the hood;
    1) The SBML file is translated into ModelingToolkit.jl format (e.g allow symbolic computations of the ODE-model 
       Jacobian). Piecewise and model events are written into DifferentialEquations.jl callbacks.
    2) The observable PEtab-table is translated into Julia-file with functions for computing the observable (h), 
       noise parameter (σ) and initial values (u0). 
    3) To be able to compute gradients via adjoint sensitivity analysis and/or forward sensitivity equations gradients
       of h and σ are computed symbolically with respect to the ODE-models states (u) and parameters (odeProblem.p).
    All these functions are created automatically and stored files under petabModel.dirJulia. To save time 
    `forceBuildJlFiles=false` by default such that the Julia files are not rebuilt in case they already exist.

    In the future we plan to allow the user to provide a Julia file directly instead of a SBML file.

    See also: [`PEtabModel`]
"""
function readPEtabModel(pathYAML::String;
                        forceBuildJuliaFiles::Bool=false,
                        verbose::Bool=true,
                        ifElseToEvent::Bool=true,
                        jlFile=false)::PEtabModel

    pathSBML, pathParameters, pathConditions, pathObservables, pathMeasurements, dirJulia, dirModel, modelName = readPEtabYamlFile(pathYAML, jlFile=jlFile)

    if jlFile == false

        pathModelJlFile = joinpath(dirJulia, modelName * ".jl")

        if !isfile(pathModelJlFile) && forceBuildJuliaFiles == false
            verbose == true && @printf("Julia model file does not exist, will build it\n")
            modelDict = XmlToModellingToolkit(pathSBML, pathModelJlFile, modelName, ifElseToEvent=ifElseToEvent)

        elseif isfile(pathModelJlFile) && forceBuildJuliaFiles == false
            verbose == true && @printf("Julia model file exists at %s - will not rebuild\n", pathModelJlFile)

        elseif forceBuildJuliaFiles == true
            verbose == true && @printf("By user option will rebuild Julia model file\n")
            isfile(pathModelJlFile) == true && rm(pathModelJlFile)
            modelDict = XmlToModellingToolkit(pathSBML, pathModelJlFile, modelName, ifElseToEvent=ifElseToEvent)
        end

    else
        jlDir = joinpath(dirModel, "Julia_model_files")
        modelDict, pathModelJlFile = JLToModellingToolkit(modelName, jlDir, ifElseToEvent=ifElseToEvent)
    end

    # Load model ODE-system
    @assert isfile(pathModelJlFile)
    _getODESystem = @RuntimeGeneratedFunction(Meta.parse(getFunctionsAsString(pathModelJlFile, 1)[1]))
    _odeSystem, stateMap, parameterMap = _getODESystem("https://xkcd.com/303/") # Argument needed by @RuntimeGeneratedFunction
    odeSystem = structural_simplify(_odeSystem)
    # TODO : Make these to strings here to save conversions
    parameterNames = parameters(odeSystem)
    stateNames = states(odeSystem)

    # Build functions for observables, sd and u0 if does not exist and include
    path_u0_h_sigma = joinpath(dirJulia, modelName * "_h_sd_u0.jl")
    path_D_h_sd = joinpath(dirJulia, modelName * "_D_h_sd.jl")
    if !isfile(path_u0_h_sigma) || !isfile(path_D_h_sd) || forceBuildJuliaFiles == true
        verbose && forceBuildJuliaFiles == false && @printf("File for h, u0 and σ does not exist will build it\n")
        verbose && forceBuildJuliaFiles == true && @printf("By user option will rebuild h, σ and u0\n")

        if !@isdefined(modelDict)
            modelDict = XmlToModellingToolkit(pathSBML, pathModelJlFile, modelName, writeToFile=false, ifElseToEvent=ifElseToEvent)
        end
        create_σ_h_u0_File(modelName, pathYAML, dirJulia, odeSystem, stateMap, modelDict, verbose=verbose, jlFile=jlFile)
        createDerivative_σ_h_File(modelName, pathYAML, dirJulia, odeSystem, modelDict, verbose=verbose, jlFile=jlFile)
    else
        verbose == true && @printf("File for h, u0 and σ exists will not rebuild it\n")
    end
    @assert isfile(path_u0_h_sigma)
    h_u0_σ_Functions = getFunctionsAsString(path_u0_h_sigma, 4)
    compute_h = @RuntimeGeneratedFunction(Meta.parse(h_u0_σ_Functions[1]))
    compute_u0! = @RuntimeGeneratedFunction(Meta.parse(h_u0_σ_Functions[2]))
    compute_u0 = @RuntimeGeneratedFunction(Meta.parse(h_u0_σ_Functions[3]))
    compute_σ = @RuntimeGeneratedFunction(Meta.parse(h_u0_σ_Functions[4]))
    @assert isfile(path_D_h_sd)
    ∂_h_σ_Functions = getFunctionsAsString(path_D_h_sd, 4)
    compute_∂h∂u! = @RuntimeGeneratedFunction(Meta.parse(∂_h_σ_Functions[1]))
    compute_∂h∂p! = @RuntimeGeneratedFunction(Meta.parse(∂_h_σ_Functions[2]))
    compute_∂σ∂σu! = @RuntimeGeneratedFunction(Meta.parse(∂_h_σ_Functions[3]))
    compute_∂σ∂σp! = @RuntimeGeneratedFunction(Meta.parse(∂_h_σ_Functions[4]))
    
    pathCallback = joinpath(dirJulia, modelName * "_callbacks.jl")
    if !isfile(pathCallback) || forceBuildJuliaFiles == true
        verbose && forceBuildJuliaFiles == false && @printf("File for callback does not exist will build it\n")
        verbose && forceBuildJuliaFiles == true && @printf("By user option will rebuild callback file\n")

        if !@isdefined(modelDict)
            modelDict = XmlToModellingToolkit(pathSBML, pathModelJlFile, modelName, writeToFile=false, ifElseToEvent=ifElseToEvent)
        end
        createCallbacksForTimeDepedentPiecewise(odeSystem, modelDict, modelName, pathYAML, dirJulia, jlFile = jlFile)
    end
    @assert isfile(pathCallback)
    strGetCallbacks = getFunctionsAsString(pathCallback, 2)
    getCallbackFunction = @RuntimeGeneratedFunction(Meta.parse(strGetCallbacks[1]))
    cbSet, checkCbActive, convertTspan = getCallbackFunction("https://xkcd.com/2694/") # Argument needed by @RuntimeGeneratedFunction
    computeTstops = @RuntimeGeneratedFunction(Meta.parse(strGetCallbacks[2]))

    petabModel = PEtabModel(modelName,
                            compute_h,
                            compute_u0!,
                            compute_u0,
                            compute_σ,
                            compute_∂h∂u!,
                            compute_∂σ∂σu!,
                            compute_∂h∂p!,
                            compute_∂σ∂σp!,
                            computeTstops,
                            convertTspan,
                            odeSystem,
                            parameterMap,
                            stateMap,
                            parameterNames,
                            stateNames,
                            dirModel,
                            dirJulia,
                            pathMeasurements,
                            pathConditions,
                            pathObservables,
                            pathParameters,
                            pathSBML,
                            pathYAML,
                            cbSet,
                            checkCbActive)

    return petabModel
end


# For reading the run-time generated PEtab-related functions which via Meta.parse are passed 
# on to @RuntimeGeneratedFunction to build the PEtab related functions without world-problems.
function getFunctionsAsString(filePath::AbstractString, nFunctions::Int64)::Vector{String}

    fStart, fEnd = zeros(Int64, nFunctions), zeros(Int64, nFunctions)
    iFunction = 1
    inFunction::Bool = false
    nLines = open(filePath, "r") do f countlines(f) end
    bodyStr = Vector{String}(undef, nLines)

    f = open(filePath, "r")
    for (iLine, line) in pairs(readlines(f))

        if length(line) ≥ 8 && line[1:8] == "function"
            fStart[iFunction] = iLine
            inFunction = true
        end

        if length(line) ≥ 3 && line[1:3] == "end"
            fEnd[iFunction] = iLine
            inFunction = false
            iFunction += 1
        end

        bodyStr[iLine] = string(line)
    end
    close(f)

    out = Vector{String}(undef, nFunctions)
    for i in eachindex(out)

        # Runtime generated functions requrie at least on function argument input, hence if missing we 
        # add a foo argument 
        if bodyStr[fStart[i]][end-1:end] == "()"
            bodyStr[fStart[i]] = bodyStr[fStart[i]][1:end-2] * "(foo)"
        end

        out[i] = prod([bodyStr[j] * '\n' for j in fStart[i]:fEnd[i]])
    end
    return out
end

import Base.show
function show(io::IO, a::PEtabModel)

    modelName = a.modelName
    numberOfODEStates = length(a.stateNames)
    numberOfODEParameters = length(a.parameterNames)
    
    @printf("PEtabModel for model %s where the ODE-system has %d states and %d parameters.\nJulia-model files can be found at %s",            
            modelName, numberOfODEStates, numberOfODEParameters, a.dirJulia)
end