"""
    readPEtabModel(pathYAML::String;
                   forceBuildJuliaFiles::Bool=false,
                   verbose::Bool=true,
                   ifElseToEvent::Bool=true)::PEtabModel

Parses a PEtab specified problem with yaml-file at `pathYAML` into a Julia accessible format. 

When parsing a PEtab problem several things happens under the hood;
1) The SBML file is translated into ModelingToolkit.jl format (e.g allow symbolic computations of the ODE-model Jacobian). Piecewise and model events are further written into DifferentialEquations.jl callbacks.
2) The observable PEtab-table is translated into Julia-file with functions for computing the observable (h), noise parameter (σ) and initial values (u0). 
3) To allow gradients via adjoint sensitivity analysis and/or forward sensitivity equations the gradients of h and σ are computed symbolically with respect to the ODE-models states (u) and parameters (odeProblem.p).
All this happens automatically, and resulting files are stored under petabModel.dirJulia. To save time `forceBuildJlFiles=false` meaning that Julia files are not rebuilt in case the already exist.

In the future we plan to allow the user to also provide a Julia file instead of a SBML file.
"""
function readPEtabModel(pathYAML::String;
                        forceBuildJuliaFiles::Bool=false,
                        verbose::Bool=true,
                        ifElseToEvent::Bool=true,
                        jlFile=false)::PEtabModel

    pathSBML, pathParameters, pathConditions, pathObservables, pathMeasurements, dirJulia, dirModel, modelName = readPEtabYamlFile(pathYAML, jlFile=jlFile)

    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && @printf(" Building PEtabModel for %s\n", modelName)

    if jlFile == false

        pathModelJlFile = joinpath(dirJulia, modelName * ".jl")

        if !isfile(pathModelJlFile) 
            if verbose == true 
                printstyled("[ Info:", color=123, bold=true)
                print(" Building Julia model file as it does not exist ...")
            end
            bBuild = @elapsed modelDict = XmlToModellingToolkit(pathSBML, pathModelJlFile, modelName, ifElseToEvent=ifElseToEvent)
            @printf(" done. Time = %.1es\n", bBuild)

        elseif isfile(pathModelJlFile) && forceBuildJuliaFiles == false && verbose == true
            printstyled("[ Info:", color=123, bold=true)
            print(" Julia model file exists and will not be rebuilt\n")

        elseif forceBuildJuliaFiles == true
            if verbose == true 
                printstyled("[ Info:", color=123, bold=true)
                print(" By user option rebuilds Julia model file ...")
            end
            isfile(pathModelJlFile) == true && rm(pathModelJlFile)
            bBuild = @elapsed modelDict = XmlToModellingToolkit(pathSBML, pathModelJlFile, modelName, ifElseToEvent=ifElseToEvent)
            @printf(" done. Time = %.1es\n", bBuild)
        end

    else
        jlDir = joinpath(dirModel, "Julia_model_files")
        modelDict, pathModelJlFile = JLToModellingToolkit(modelName, jlDir, ifElseToEvent=ifElseToEvent)
    end

    # Load model ODE-system
    @assert isfile(pathModelJlFile)
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Symbolically processes ODE-system ...")
    bTake = @elapsed begin
        _getODESystem = @RuntimeGeneratedFunction(Meta.parse(getFunctionsAsString(pathModelJlFile, 1)[1]))
        _odeSystem, stateMap, parameterMap = _getODESystem("https://xkcd.com/303/") # Argument needed by @RuntimeGeneratedFunction
        odeSystem = structural_simplify(_odeSystem)
        # TODO : Make these to strings here to save conversions
        parameterNames = parameters(odeSystem)
        stateNames = states(odeSystem)
    end
    verbose == true && @printf(" done. Time = %.1es\n", bTake)

    # Build functions for observables, sd and u0 if does not exist and include
    path_u0_h_sigma = joinpath(dirJulia, modelName * "_h_sd_u0.jl")
    path_D_h_sd = joinpath(dirJulia, modelName * "_D_h_sd.jl")
    if !isfile(path_u0_h_sigma) || forceBuildJuliaFiles == true
        if verbose == true && !isfile(path_u0_h_sigma)
            printstyled("[ Info:", color=123, bold=true)
            print(" Building u0, h σ file as it does not exist ...")
        elseif verbose == true
            printstyled("[ Info:", color=123, bold=true)
            print(" By user option rebuilds u0, h σ file ...")
        end
        if !@isdefined(modelDict)
            modelDict = XmlToModellingToolkit(pathSBML, pathModelJlFile, modelName, writeToFile=false, ifElseToEvent=ifElseToEvent)
        end
        bBuild = @elapsed create_σ_h_u0_File(modelName, pathYAML, dirJulia, odeSystem, stateMap, modelDict, jlFile=jlFile)
        verbose == true && @printf(" done. Time = %.1es\n", bBuild)
    elseif verbose == true
        printstyled("[ Info:", color=123, bold=true)
        print(" u0, h and σ file exists and will not be rebuilt\n")
    end
    
    if !isfile(path_D_h_sd) || forceBuildJuliaFiles == true
        if verbose == true && !isfile(path_D_h_sd)
            printstyled("[ Info:", color=123, bold=true)
            print(" Building ∂h∂p, ∂h∂u, ∂σ∂p and ∂σ∂u file as it does not exist ...")
        elseif verbose == true
            printstyled("[ Info:", color=123, bold=true)
            print(" By user option rebuilds ∂h∂p, ∂h∂u, ∂σ∂p and ∂σ∂u file ...")
        end
        if !@isdefined(modelDict)
            modelDict = XmlToModellingToolkit(pathSBML, pathModelJlFile, modelName, writeToFile=false, ifElseToEvent=ifElseToEvent)
        end
        bBuild = @elapsed createDerivative_σ_h_File(modelName, pathYAML, dirJulia, odeSystem, modelDict, jlFile=jlFile)
        verbose == true && @printf(" done. Time = %.1es\n", bBuild)
    elseif verbose == true
        printstyled("[ Info:", color=123, bold=true)
        print(" ∂h∂p, ∂h∂u, ∂σ∂p and ∂σ∂u file exists and will not be rebuilt\n")
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
        if verbose == true && !isfile(pathCallback)
            printstyled("[ Info:", color=123, bold=true)
            print(" Building callback file as it does not exist ...")
        elseif verbose == true
            printstyled("[ Info:", color=123, bold=true)
            print(" By user option rebuilds callback file ...")
        end
        if !@isdefined(modelDict)
            modelDict = XmlToModellingToolkit(pathSBML, pathModelJlFile, modelName, writeToFile=false, ifElseToEvent=ifElseToEvent)
        end
        bBuild = @elapsed createCallbacksForTimeDepedentPiecewise(odeSystem, modelDict, modelName, pathYAML, dirJulia, jlFile = jlFile)
        verbose == true && @printf(" done. Time = %.1es\n", bBuild)

    elseif verbose == true
        printstyled("[ Info:", color=123, bold=true)
        print(" Callback file exists and will not be rebuilt\n")
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

        if length(line) ≥ 3 && line[1] != '#' && line[1:3] == "end"
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

    modelName = @sprintf("%s", a.modelName)
    numberOfODEStates = @sprintf("%d", length(a.stateNames))
    numberOfODEParameters = @sprintf("%d",  length(a.parameterNames))
    
    printstyled("PEtabModel", color=116)
    print(" for model ")
    printstyled(modelName, color=116)
    print(". ODE-system has ")
    printstyled(numberOfODEStates * " states", color=116)
    print(" and ")
    printstyled(numberOfODEParameters * " parameters.", color=116)
    @printf("\nGenerated Julia files are at %s", a.dirJulia)
end