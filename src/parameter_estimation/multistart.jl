"""
    calibrate_multistart(prob::PEtabODEProblem,
                               alg,
                               nmultistarts::Signed,
                               dirsave::Union{Nothing, String};
                               sampling_method=QuasiMonteCarlo.LatinHypercubeSample(),
                               sample_prior::Bool=true,
                               options=options,
                               seed=nothing,
                               save_trace::Bool=false)::PEtabMultistartResult

Perform multistart optimization for a PEtabODEProblem using the algorithm `alg`.

The optimization algorithm `alg` can be one of the following:
- [Optim](https://julianlsolvers.github.io/Optim.jl/stable/) LBFGS, BFGS, or IPNewton methods
- [IpoptOptimiser](https://coin-or.github.io/Ipopt/) interior-point optimizer
- [Fides](https://github.com/fides-dev/fides) Newton trust region method

For each algorithm, optimizer options can be provided in the format of the respective package.
For a comprehensive list of available options, please refer to the main documentation. If you want the optimizer
to return parameter and objective trace information, set `save_trace=true`.

Multistart optimization involves generating multiple starting points for optimization runs. These starting points
are generated using the specified `sampling_method` from [QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl),
with the default being LatinHypercubeSample, a method that typically produces better results than random sampling.
If `sample_prior=true` (default), for parameters with priors samples are taken from the prior distribution, where the
distribution is clipped/truncated by the parameter's lower- and upper bound. For reproducibility, you can set a random
number generator seed using the `seed` parameter.

If `dirsave` is provided as `nothing`, results are not written to disk. Otherwise, if a directory path is provided,
results are written to disk. Writing results to disk is recommended in case the optimization process is terminated
after a number of optimization runs.

The results are returned as a `PEtabMultistartResult`, which stores the best-found minima (`xmin`),
smallest objective value (`fmin`), as well as optimization results for each run.

!!! note
    To use Optim optimizers, you must load Optim with `using Optim`. To use Ipopt, you
    must load Ipopt with `using Ipopt`. To use Fides, load PyCall with `using PyCall` and
    ensure Fides is installed (see documentation for setup).

## Examples
```julia
# Perform 100 optimization runs using Optim's IPNewton, save results in dirsave
using Optim
dirsave = joinpath(@__DIR__, "Results")
res = calibrate_multistart(prob, Optim.IPNewton(), 100, dirsave;
                           options=Optim.Options(iterations = 1000))
```
```julia
# Perform 100 optimization runs using Fides, save results in dirsave
using PyCall
dirsave = joinpath(@__DIR__, "Results")
res = calibrate_multistart(prob, Fides(nothing), 100, dirsave;
                               options=py"{'maxiter' : 1000}"o)
```
```julia
# Perform 100 optimization runs using Ipopt, save results in dirsave. For each
# run save the trace
using Ipopt
dirsave = joinpath(@__DIR__, "Results")
res = calibrate_multistart(prob, IpoptOptimiser(false), 100, dirsave;
                               options=IpoptOptions(max_iter = 1000),
                               save_trace=true)
```
```julia
# Perform 100 optimization runs using Optimization with IPNewton, save results in dirsave.
using Optimization
using OptimizationOptimJL
prob = PEtab.OptimizationProblem(prob, interior_point_alg=true)
res = calibrate_multistart(prob, IPNewton(), 100, dirsave;
                                 reltol=1e-8)
```
"""
function calibrate_multistart end

function _calibrate_multistart(prob::PEtabODEProblem, alg, nmultistarts, dirsave,
                               sampling_method, options, sample_prior::Bool,
                               save_trace::Bool)::PEtabMultistartResult
    if isnothing(dirsave)
        path_x0, path_res, path_trace, path_x = nothing, nothing, nothing, nothing
    else
        !isdir(dirsave) && mkpath(dirsave)
        i = 1
        while true
            path_x0 = joinpath(dirsave, "Start_guesses_$i.csv")
            !isfile(path_x0) && break
            i += 1
        end
        path_x0 = joinpath(dirsave, "startguesses" * string(i) * ".csv")
        path_res = joinpath(dirsave, "results" * string(i) * ".csv")
        path_x = joinpath(dirsave, "xmins" * string(i) * ".csv")
        if save_trace == true
            path_trace = joinpath(dirsave, "trace" * string(i) * ".csv")
        else
            path_trace = nothing
        end
    end

    xstarts = get_startguesses(prob, nmultistarts; sampling_method = sampling_method,
                               sample_prior = sample_prior)
    if !isnothing(path_x0)
        xnames = propertynames(xstarts[1]) |> collect
        xstarts_df = DataFrame(vcat(reduce(vcat, xstarts')), xnames)
        xstarts_df[!, "startguess"] = 1:nrow(xstarts_df)
        CSV.write(path_x0, xstarts_df)
    end

    runs = Vector{PEtabOptimisationResult}(undef, nmultistarts)
    for i in 1:nmultistarts
        if !isempty(xstarts[i])
            xstart = xstarts[i]
            runs[i] = calibrate(prob, xstart, alg; save_trace = save_trace,
                                options = options)
        # This happens when there are now parameter to estimate, edge case that can
        # appear in for example petab-select
        else
            runs[i] = PEtabOptimisationResult(:alg, Vector{Vector{Float64}}(undef, 0),
                                              Vector{Float64}(undef, 0), 0,
                                              prob.nllh(Float64[]), Float64[],
                                              Float64[], true, 0.0, nothing)
        end
        isnothing(path_res) && continue
        # It is best practive to save results after each multistart, as sometimes running
        # multistarts can take very long time
        _save_multistart_results(path_res, path_x, path_trace, runs[i], i)
    end

    bestrun = runs[argmin([isnan(r.fmin) ? Inf : r.fmin for r in runs])]
    fmin = bestrun.fmin
    xmin = bestrun.xmin
    # Will fix with regex when things are running
    sampling_method_str = string(sampling_method)[1:findfirst(x -> x == '(',
                                                              string(sampling_method))][1:(end - 1)]
    return PEtabMultistartResult(xmin, fmin, nmultistarts, bestrun.alg, sampling_method_str,
                                 dirsave, runs)
end

function _save_multistart_results(path_res::String, path_x::String,
                                  path_trace::Union{String, Nothing},
                                  res::PEtabOptimisationResult, i::Int64)::Nothing
    xnames = propertynames(res.xmin) |> collect
    res_df = DataFrame(fmin = res.fmin, alg = res.alg, runtime = res.runtime,
                       niterations = res.niterations, converged = res.converged,
                       startguess = i)
    x_df = DataFrame(Matrix(res.xmin'), xnames)
    x_df[!, "startguess"] = [i]
    CSV.write(path_res, res_df, append = isfile(path_res))
    CSV.write(path_x, x_df, append = isfile(path_x))
    if !isnothing(path_trace) && !isnothing(res.ftrace) && !isempty(res.ftrace)
        trace_df = DataFrame(Matrix(reduce(vcat, res.xtrace')), xnames)
        trace_df[!, "ftrace"] = res.ftrace
        trace_df[!, "startguess"] = repeat([i], length(res.ftrace))
        CSV.write(path_trace, trace_df, append = isfile(path_trace))
    end
    return nothing
end
