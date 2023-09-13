function savePartialResults(pathSaveRes::String,
                            pathSaveParameters::String,
                            pathSaveTrace::Union{String, Nothing},
                            res::PEtabOptimisationResult,
                            θ_estNames::Vector{Symbol},
                            i::Int64)::Nothing

    dfSaveRes = DataFrame(fmin=res.fBest,
                          alg=String(res.alg),
                          n_iterations = res.nIterations,
                          run_time = res.runTime,
                          converged=string(res.converged),
                          Start_guess=i)
    dfSaveParameters = DataFrame(Matrix(res.xBest'), θ_estNames)
    dfSaveParameters[!, "Start_guess"] = [i]
    CSV.write(pathSaveRes, dfSaveRes, append=isfile(pathSaveRes))
    CSV.write(pathSaveParameters, dfSaveParameters, append=isfile(pathSaveParameters))

    if !isnothing(pathSaveTrace)
        dfSaveTrace = DataFrame(Matrix(reduce(vcat, res.xTrace')), θ_estNames)
        dfSaveTrace[!, "f_trace"] = res.fTrace
        dfSaveTrace[!, "Start_guess"] = repeat([i], length(res.fTrace))
        CSV.write(pathSaveTrace, dfSaveTrace, append=isfile(pathSaveTrace))
    end
    return nothing
end


function generateStartGuesses(petabProblem::PEtabODEProblem,
                              samplingMethod::T,
                              nMultiStarts::Int64)::Matrix{Float64} where T <: QuasiMonteCarlo.SamplingAlgorithm

    # Nothing prevents the user from sending in a parameter vector with zero parameters
    if length(petabProblem.lowerBounds) == 0
        return nothing
    end

    # Return a random number sampled from uniform distribution
    if nMultiStarts == 1
        return [rand() * (petabProblem.upperBounds[i] - petabProblem.lowerBounds[i]) + petabProblem.lowerBounds[i] for i in eachindex(petabProblem.lowerBounds)]
    end

    startGuesses = Matrix{Float64}(undef, length(petabProblem.lowerBounds), nMultiStarts)
    foundStarts = 0
    while true
        _samples = QuasiMonteCarlo.sample(nMultiStarts - foundStarts, petabProblem.lowerBounds, petabProblem.upperBounds, samplingMethod)
        for i in size(_samples)[2]
            _p = _samples[:, i]
            _cost = petabProblem.computeCost(_p)
            if !isinf(_cost)
                foundStarts += 1
                startGuesses[:, foundStarts] .= _p
            end
        end
        if foundStarts == nMultiStarts
            break
        end
    end

    return startGuesses
end


function _multistartModelCallibration(petabProblem::PEtabODEProblem, 
                                      alg, 
                                      nMultiStarts, 
                                      dirSave, 
                                      samplingMethod, 
                                      options, 
                                      saveTrace::Bool)::PEtabMultistartOptimisationResult

    if isnothing(dirSave)
        pathSavex0, pathSaveRes, pathSaveTrace = nothing, nothing, nothing
    else
        !isdir(dirSave) && mkpath(dirSave)
        _i = 1
        while true
            pathSavex0 = joinpath(dirSave, "Start_guesses" * string(_i) * ".csv")
            if !isfile(pathSavex0)
                break
            end
            _i += 1
        end
        pathSavex0 = joinpath(dirSave, "Start_guesses" * string(_i) * ".csv")
        pathSaveRes = joinpath(dirSave, "Optimisation_results" * string(_i) * ".csv")
        pathSaveParameters = joinpath(dirSave, "Best_parameters" * string(_i) * ".csv")
        if saveTrace == true
            pathSaveTrace = joinpath(dirSave, "Trace" * string(_i) * ".csv")
        else
            pathSaveTrace = nothing
        end
    end

    startGuesses = generateStartGuesses(petabProblem, samplingMethod, nMultiStarts)
    if !isnothing(pathSavex0)
        startGuessesDf = DataFrame(Matrix(startGuesses)', petabProblem.θ_estNames)
        startGuessesDf[!, "Start_guess"] = 1:size(startGuessesDf)[1]
        CSV.write(pathSavex0, startGuessesDf)
    end

    _res = Vector{PEtabOptimisationResult}(undef, nMultiStarts)
    for i in 1:nMultiStarts
        _p0 = startGuesses[:, i]
        _res[i] = callibrateModel(petabProblem, _p0, alg, saveTrace=saveTrace, options=options)
        if !isnothing(pathSaveRes)
            savePartialResults(pathSaveRes, pathSaveParameters, pathSaveTrace, _res[i], petabProblem.θ_estNames, i)
        end
    end

    resBest = _res[argmin([_res[i].fBest for i in eachindex(_res)])]
    fMin = resBest.fBest
    xMin = resBest.xBest
    samplingMethodStr = string(samplingMethod)[1:findfirst(x -> x == '(', string(samplingMethod))][1:end-1]
    results = PEtabMultistartOptimisationResult(xMin, 
                                                fMin, 
                                                nMultiStarts, 
                                                resBest.alg,
                                                samplingMethodStr, 
                                                dirSave,
                                                _res)
    return results
end      