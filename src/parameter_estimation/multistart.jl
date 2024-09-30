"""
    calibrate_multistart(prob::PEtabODEProblem, alg, nmultistarts::Integer;
                         dirsave=nothing, kwargs...)::PEtabMultistartResult

Perform `nmultistarts` parameter estimation runs from randomly sampled starting points using
the optimization algorithm `alg` to estimate the unknown model parameters in `prob`.

 A list of available and recommended optimization algorithms (`alg`) can be found in the
package documentation and [`calibrate`](@ref) documentation. If `dirsave` is provided,
intermediate results for each run are saved in `dirsave`. It is **strongly** recommended to
provide `dirsave` for larger models, as parameter estimation can take hours (or even days!),
and without `dirsave`, all intermediate results will be lost if something goes wrong.

Different ways to visualize the parameter estimation result can be found in the
documentation.

See also [`PEtabMultistartResult`](@ref), [`get_startguesses`](@ref), and [`calibrate`](@ref).

## Keyword Arguments
- `sampling_method = LatinHypercubeSample()`: Method for sampling a diverse (spread out) set
   of starting points. See the documentation for [`get_startguesses`](@ref), which is the
   function used for sampling.
- `sample_prior::Bool = true`: See the documentation for [`get_startguesses`](@ref).
- `seed = nothing`: Seed used when generating starting points.
- `options = DEFAULT_OPTIONS`: Configurable options for `alg`. See the documentation for
    [`calibrate`](@ref).
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
            path_x0 = joinpath(dirsave, "startguesses$i.csv")
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
        if !isempty(xstarts)
            xstart = xstarts[i]
            runs[i] = calibrate(prob, xstart, alg; save_trace = save_trace,
                                options = options)
            # This happens when there are now parameter to estimate, edge case that can
            # appear in for example petab-select
        else
            xstart, xmin = ComponentArray{Float64}(), ComponentArray{Float64}()
            xtrace, ftrace = Vector{Vector{Float64}}(undef, 0), Vector{Float64}(undef, 0)
            fmin = prob.nllh(xstart)
            runs[i] = PEtabOptimisationResult(xmin, fmin, xstart, :alg, 0, 0.0, xtrace,
                                              ftrace, true, nothing)
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
    return PEtabMultistartResult(xmin, fmin, bestrun.alg, nmultistarts, sampling_method_str,
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
