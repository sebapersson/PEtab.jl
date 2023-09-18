"""
    calibrateModel(petabProblem::PEtabODEProblem,
                   p0::Vector{Float64},
                   alg;
                   saveTrace::Bool=false,
                   options=algOptions)::PEtabOptimisationResult

Parameter estimate a model for a PEtabODEProblem using an optimization algorithm `alg` and an initial guess `p0`.

The optimization algorithm `alg` can be one of the following:
- [Optim](https://julianlsolvers.github.io/Optim.jl/stable/) LBFGS, BFGS, or IPNewton methods
- [IpoptOptimiser](https://coin-or.github.io/Ipopt/) interior-point optimizer
- [Fides](https://github.com/fides-dev/fides) Newton trust region method

Each algorithm accepts specific optimizer options in the format of the respective package. For a
comprehensive list of available options, please refer to the main documentation.

If you want the optimizer to return parameter and objective trace information, set `saveTrace=true`.
Results are returned as a `PEtabOptimisationResult`, which includes the following information: minimum
parameter values found (`xMin`), smallest objective value (`fMin`), number of iterations, runtime, whether
the optimizer converged, and optionally, the trace.

!!! note
    To use Optim optimizers, you must load Optim with `using Optim`. To use Ipopt, you must load Ipopt with `using Ipopt`. 
    To use Fides, load PyCall with `using PyCall` and ensure Fides is installed (see documentation for setup).

## Examples
```julia
# Perform parameter estimation using Optim's IPNewton with a given initial guess
using Optim
res = calibrateModel(petabProblem, p0, Optim.IPNewton();
                     options=Optim.Options(iterations = 1000))
```
```julia
# Perform parameter estimation using Fides with a given initial guess
using PyCall
res = calibrateModel(petabProblem, p0, Fides(nothing);
                     options=py"{'maxiter' : 1000}"o)
```
```julia
# Perform parameter estimation using Ipopt and save the trace
using Ipopt
res = calibrateModel(petabProblem, p0, IpoptOptimiser(false);
                     options=IpoptOptions(max_iter = 1000), 
                     saveTrace=true)
```
"""
function calibrateModel end


"""
    calibrateModelMultistart(petabProblem::PEtabODEProblem,
                             alg,
                             nMultiStarts::Signed,
                             dirSave::Union{Nothing, String};
                             samplingMethod=QuasiMonteCarlo.LatinHypercubeSample(),
                             options=algOptions,
                             seed=nothing,
                             saveTrace::Bool=false)::PEtabMultistartOptimisationResult

Perform multistart optimization for a PEtabODEProblem using the algorithm `alg`.

The optimization algorithm `alg` can be one of the following:
- [Optim](https://julianlsolvers.github.io/Optim.jl/stable/) LBFGS, BFGS, or IPNewton methods
- [IpoptOptimiser](https://coin-or.github.io/Ipopt/) interior-point optimizer
- [Fides](https://github.com/fides-dev/fides) Newton trust region method

For each algorithm, optimizer options can be provided in the format of the respective package.
For a comprehensive list of available options, please refer to the main documentation. If you want the optimizer
to return parameter and objective trace information, set `saveTrace=true`.

Multistart optimization involves generating multiple starting points for optimization runs. These starting points
are generated using the specified `samplingMethod` from [QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl),
with the default being LatinHypercubeSample, a method that typically produces better results than random sampling.
For reproducibility, you can set a random number generator seed using the `seed` parameter.

If `dirSave` is provided as `nothing`, results are not written to disk. Otherwise, if a directory path is provided,
results are written to disk. Writing results to disk is recommended in case the optimization process is terminated
after a number of optimization runs.

The results are returned as a `PEtabMultistartOptimisationResult`, which stores the best-found minima (`xMin`),
smallest objective value (`fMin`), as well as optimization results for each run.

!!! note
    To use Optim optimizers, you must load Optim with `using Optim`. To use Ipopt, you must load Ipopt with `using Ipopt`. 
    To use Fides, load PyCall with `using PyCall` and ensure Fides is installed (see documentation for setup).

## Examples
```julia
# Perform 100 optimization runs using Optim's IPNewton, save results in dirSave
using Optim
dirSave = joinpath(@__DIR__, "Results")
res = calibrateModelMultistart(petabProblem, Optim.IPNewton(), 100, dirSave;
                               options=Optim.Options(iterations = 1000))
```
```julia
# Perform 100 optimization runs using Fides, save results in dirSave
using PyCall
dirSave = joinpath(@__DIR__, "Results")
res = calibrateModelMultistart(petabProblem, Fides(nothing), 100, dirSave;
                               options=py"{'maxiter' : 1000}"o)
```
```julia
# Perform 100 optimization runs using Ipopt, save results in dirSave. For each 
# run save the trace 
using Ipopt
dirSave = joinpath(@__DIR__, "Results")
res = calibrateModelMultistart(petabProblem, IpoptOptimiser(false), 100, dirSave;
                               options=IpoptOptions(max_iter = 1000), 
                               saveTrace=true)
```
"""
function calibrateModelMultistart end


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
                              nMultiStarts::Int64;
                              verbose::Bool=false)::Matrix{Float64} where T <: QuasiMonteCarlo.SamplingAlgorithm

    verbose == true && @info "Generating start-guesses"

    # Nothing prevents the user from sending in a parameter vector with zero parameters
    if length(petabProblem.lowerBounds) == 0
        return nothing
    end

    startGuesses = Matrix{Float64}(undef, length(petabProblem.lowerBounds), nMultiStarts)
    foundStarts = 0
    while true
        # QuasiMonteCarlo is deterministic, so for sufficiently few start-guesses we can end up in a never ending
        # loop. To sidestep this if less than 10 starts are left numbers are generated from the uniform distribution
        if nMultiStarts - foundStarts > 10
            _samples = QuasiMonteCarlo.sample(nMultiStarts - foundStarts, petabProblem.lowerBounds, petabProblem.upperBounds, samplingMethod)
        else
            _samples = Matrix{Float64}(undef, length(petabProblem.lowerBounds), nMultiStarts - foundStarts)
            for i in 1:(nMultiStarts - foundStarts)
                _samples[:, i] .= [rand() * (petabProblem.upperBounds[j] - petabProblem.lowerBounds[j]) + petabProblem.lowerBounds[j] for j in eachindex(petabProblem.lowerBounds)]
            end
        end

        for i in 1:size(_samples)[2]
            _p = _samples[:, i]
            _cost = petabProblem.computeCost(_p)
            if !isinf(_cost)
                foundStarts += 1
                startGuesses[:, foundStarts] .= _p
            end
        end
        verbose == true && @printf("Found %d of %d multistarts\n", foundStarts, nMultiStarts)
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
        _res[i] = calibrateModel(petabProblem, _p0, alg, saveTrace=saveTrace, options=options)
        if !isnothing(pathSaveRes)
            savePartialResults(pathSaveRes, pathSaveParameters, pathSaveTrace, _res[i], petabProblem.θ_estNames, i)
        end
    end

    resBest = _res[argmin([_res[i].fMin for i in eachindex(_res)])]
    fMin = resBest.fMin
    xMin = resBest.xMin
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