"""
    calibrate_multistart(prob::PEtabODEProblem, alg, nmultistarts::Integer; nprocs = 1,
                         dirsave=nothing, kwargs...)::PEtabMultistartResult

Perform `nmultistarts` parameter estimation runs from randomly sampled starting points using
the optimization algorithm `alg` to estimate the unknown model parameters in `prob`.

A list of available and recommended optimisation algorithms (`alg`) can be found in the
package documentation and in the [`calibrate`](@ref) documentation. If `nprocs > 1`, the
parameter estimation runs are performed in parallel using the `pmap` function from
Distributed.jl with `nprocs` processes. If parameter estimation on a single process
(`nprocs = 1`) takes longer than 5 minutes, we **strongly** recommend setting `nprocs > 1`,
because since multi-start parameter estimation is inherently parallel, performing it in
parallel can greatly reduce runtime. Note that `nprocs` should not be larger than the number
of cores on the computer.

If `dirsave` is provided, intermediate results for each run are saved in `dirsave`. It is
**strongly** recommended to provide `dirsave` for larger models, as parameter estimation can
take hours (or even days!),and without `dirsave`, all intermediate results will be lost if
something goes wrong.

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
                               save_trace::Bool, nprocs::Int64)::PEtabMultistartResult
    paths_save = Dict{Symbol, String}()
    if !isnothing(dirsave)
        !isdir(dirsave) && mkpath(dirsave)
        i = 1
        while true
            path_x0 = joinpath(dirsave, "startguesses$i.csv")
            !isfile(path_x0) && break
            i += 1
        end
        paths_save[:x0] = joinpath(dirsave, "startguesses" * string(i) * ".csv")
        paths_save[:res] = joinpath(dirsave, "results" * string(i) * ".csv")
        paths_save[:xmin] = joinpath(dirsave, "xmins" * string(i) * ".csv")
        if save_trace == true
            paths_save[:trace] = joinpath(dirsave, "trace" * string(i) * ".csv")
        end
    end

    xstarts = get_startguesses(prob, nmultistarts; sampling_method = sampling_method,
                               sample_prior = sample_prior)
    if !isempty(paths_save)
        xnames = propertynames(xstarts[1]) |> collect
        xstarts_df = DataFrame(vcat(reduce(vcat, xstarts')), xnames)
        xstarts_df[!, "startguess"] = 1:nrow(xstarts_df)
        CSV.write(paths_save[:x0], xstarts_df)
    end

    # In case nprocs > 1 a mutex is needed to prevent multiple processes from writing
    # intermediate results to file at the same time
    mutex = Distributed.RemoteChannel(()->Channel{Bool}(1))
    Distributed.put!(mutex, true)
    # Setup processes to run parameter estimation on. If nproc > 1 then the main proc is
    # not doing any work, hence the plus + 1. Second, to run the code, PEtab must be loaded
    # on each relevant process
    _nprocs = nprocs == 1 ? nprocs - 1 : nprocs
    pids = _create_workers(_nprocs)
    _load_packages_workers(pids)
    _xstarts = [(x.second, x.first) for x in pairs(xstarts)]
    _calibrate_procs = x -> _calibrate_startguess(x[1], x[2], prob, alg, save_trace,
                                                  options, paths_save, mutex)
    runs = Distributed.pmap(_calibrate_procs, _xstarts)
    _remove_workers(pids)

    bestrun = runs[argmin([isnan(r.fmin) ? Inf : r.fmin for r in runs])]
    fmin = bestrun.fmin
    xmin = bestrun.xmin
    # Will fix with regex when things are running
    sampling_method_str = string(sampling_method)[1:findfirst(x -> x == '(',
                                                              string(sampling_method))][1:(end - 1)]
    return PEtabMultistartResult(xmin, fmin, bestrun.alg, nmultistarts, sampling_method_str,
                                 dirsave, runs)
end

function _calibrate_startguess(xstart, i, prob::PEtabODEProblem, alg, save_trace::Bool, options, paths_save, mutex)
    if !isempty(xstart)
        res = calibrate(prob, xstart, alg; save_trace = save_trace, options = options)
    # This happens when there are now parameter to estimate, edge case that can
    # appear in for example petab-select
    else
        xstart, xmin = ComponentArray{Float64}(), ComponentArray{Float64}()
        xtrace, ftrace = Vector{Vector{Float64}}(undef, 0), Vector{Float64}(undef, 0)
        fmin = prob.nllh(xstart)
        res = PEtabOptimisationResult(xmin, fmin, xstart, :alg, 0, 0.0, xtrace, ftrace,
                                      true, nothing)
    end
    if !isempty(paths_save)
        Distributed.take!(mutex)
        try
            _save_multistart_results(paths_save, res, i)
        finally
            Distributed.put!(mutex, true)
        end
    end
    return res
end

function _save_multistart_results(paths_save::Dict{Symbol, String},
                                  res::PEtabOptimisationResult, i::Int64)::Nothing
    xnames = propertynames(res.xmin) |> collect
    res_df = DataFrame(fmin = res.fmin, alg = res.alg, runtime = res.runtime,
                       niterations = res.niterations, converged = res.converged,
                       startguess = i)
    x_df = DataFrame(Matrix(res.xmin'), xnames)
    x_df[!, "startguess"] = [i]
    CSV.write(paths_save[:res], res_df, append = isfile(paths_save[:res]))
    CSV.write(paths_save[:xmin], x_df, append = isfile(paths_save[:xmin]))
    if haskey(paths_save, :trace) && !isnothing(res.ftrace) && !isempty(res.ftrace)
        trace_df = DataFrame(Matrix(reduce(vcat, res.xtrace')), xnames)
        trace_df[!, "ftrace"] = res.ftrace
        trace_df[!, "startguess"] = repeat([i], length(res.ftrace))
        CSV.write(paths_save[:trace], trace_df, append = isfile(paths_save[:trace]))
    end
    return nothing
end

function _load_packages_workers(workers::Vector{Int64})::Nothing
    isempty(workers) && return nothing
    names_imported = names(Main, imported=true)
    @eval @everywhere $workers eval(:(using PEtab))
    if :Optim in names_imported
        @eval @everywhere $workers eval(:(using Optim))
    end
    if :Ipopt in names_imported
        @eval @everywhere $workers eval(:(using Ipopt))
    end
    if :PyCall in names_imported
        @eval @everywhere $workers eval(:(using PyCall))
    end
    return nothing
end

function _create_workers(n::Int64)::Vector{Int64}
    return Distributed.addprocs(n)
end

function _remove_workers(pids::Vector{Int64})::Nothing
    Distributed.rmprocs(pids)
    return nothing
end
