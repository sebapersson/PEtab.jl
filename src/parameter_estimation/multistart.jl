const _SUPPORTED_PACKAGES = [:Fides, :Ipopt, :Optim, :Optimisers]

"""
    calibrate_multistart([rng::AbstractRng], prob::PEtabODEProblem, alg, nmultistarts::Integer;
                         nprocs = 1, dirsave=nothing, kwargs...)::PEtabMultistartResult

Perform `nmultistarts` parameter estimation runs from randomly sampled starting points using
the optimization algorithm `alg` to estimate the unknown model parameters in `prob`.

A list of available and recommended optimisation algorithms (`alg`) can be found in the
package documentation and in the [`calibrate`](@ref) documentation.

As with [`get_startguesses`](@ref), the `rng` controlling the generation of starting points
is optional; if omitted, `Random.default_rng()` is used. For reproducible starting points,
pass a seeded `rng` (e.g., `MersenneTwister(42)`).

If `nprocs > 1`, the parameter estimation runs are performed in parallel using the
[`pmap`](https://docs.julialang.org/en/v1/stdlib/Distributed/#Distributed.pmap) function
from Distributed.jl with `nprocs` processes. If parameter estimation on a single process
(`nprocs = 1`) takes longer than 5 minutes, we **strongly** recommend setting `nprocs > 1`,
as this can greatly reduce runtime. Note that `nprocs` should not be larger than the number
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
   function used for generating starting points.
- `sample_prior::Bool = true`: See the documentation for [`get_startguesses`](@ref).
- `options = DEFAULT_OPTIONS`: Configurable options for `alg`. See the documentation for
    [`calibrate`](@ref).
- `show_progress::Bool = false`: Whether to display a progress bar tracking how many
   multistart runs have completed.
"""
function calibrate_multistart(
        prob::PEtabODEProblem, alg, nmultistarts; kwargs...
    )::PEtab.PEtabMultistartResult
    rng = Random.default_rng()

    return calibrate_multistart(
        rng, prob, alg, nmultistarts; kwargs...
    )
end

function _calibrate_multistart(
        rng::Random.AbstractRNG, prob::PEtabODEProblem, alg, nmultistarts::Signed,
        dirsave::Union{Nothing, AbstractString}, sampling_method::SamplingAlgorithm, options,
        sample_prior::Bool, save_trace::Bool, nprocs::Signed, show_progress::Bool,
        init_weight::Union{Nothing, Function}, init_bias::Union{Function, Nothing},
    )::PEtabMultistartResult

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

    xstarts = get_startguesses(
        rng, prob, nmultistarts; sampling_method = sampling_method,
        sample_prior = sample_prior, init_weight = init_weight, init_bias = init_bias
    )
    if !isempty(paths_save)
        x_names = ComponentArrays.labels(xstarts[1])
        xstarts_df = DataFrame(vcat(reduce(vcat, xstarts')), x_names)
        xstarts_df[!, "startguess"] = 1:DataFrames.nrow(xstarts_df)
        CSV.write(paths_save[:x0], xstarts_df)
    end

    # In case nprocs > 1 a mutex is needed to prevent multiple processes from writing
    # intermediate results to file at the same time
    mutex = Distributed.RemoteChannel(() -> Channel{Bool}(1))
    Distributed.put!(mutex, true)
    # Setup processes to run parameter estimation on. If nproc > 1 then the main proc is
    # not doing any work, hence the plus + 1. Second, to run the code, PEtab must be loaded
    # on each relevant process
    _nprocs = nprocs == 1 ? nprocs - 1 : nprocs
    pids = _create_workers(_nprocs)
    _load_packages_workers(pids)
    _xstarts = [(x.second, x.first) for x in pairs(xstarts)]
    _calibrate_procs = x -> _calibrate_startguess(
        x[1], x[2], prob, alg, save_trace,
        options, paths_save, mutex
    )
    # Progress-meter logging
    p = ProgressMeter.Progress(
        length(_xstarts); enabled = show_progress, desc = "Multistart runs: ",
        showspeed = true, dt = 1.0
    )
    runs = ProgressMeter.progress_pmap(_calibrate_procs, _xstarts; progress = p)
    _remove_workers(pids)

    bestrun = runs[argmin([isnan(r.fmin) ? Inf : r.fmin for r in runs])]
    fmin = bestrun.fmin
    xmin = bestrun.xmin
    # Will fix with regex when things are running
    sampling_method_str = string(sampling_method)[
        1:findfirst(x -> x == '(', string(sampling_method)),
    ][1:(end - 1)]
    return PEtabMultistartResult(
        xmin, fmin, bestrun.alg, nmultistarts, sampling_method_str, dirsave, runs
    )
end

function _calibrate_startguess(
        xstart, i, prob::PEtabODEProblem, alg, save_trace::Bool, options, paths_save, mutex
    )
    if !isempty(xstart)
        res = calibrate(prob, xstart, alg; save_trace = save_trace, options = options)
        # This happens when there are now parameter to estimate, edge case that can
        # appear in for example petab-select
    else
        xstart, xmin = ComponentVector{Float64}(), ComponentVector{Float64}()
        xtrace, ftrace = Vector{Vector{Float64}}(undef, 0), Vector{Float64}(undef, 0)
        fmin = prob.nllh(xstart)
        res = PEtabOptimisationResult(
            xmin, fmin, xstart, :alg, 0, 0.0, xtrace, ftrace, true, nothing
        )
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

function _save_multistart_results(
        paths_save::Dict{Symbol, String}, res::PEtabOptimisationResult, i::Int64
    )::Nothing
    x_names = ComponentArrays.labels(res.xmin)
    x_min_vals = collect(res.xmin)
    res_df = DataFrame(
        fmin = res.fmin, alg = res.alg, runtime = res.runtime,
        niterations = res.niterations, converged = res.converged, startguess = i
    )
    x_df = DataFrame(Matrix(x_min_vals'), x_names)
    x_df[!, "startguess"] = [i]
    CSV.write(paths_save[:res], res_df, append = isfile(paths_save[:res]))
    CSV.write(paths_save[:xmin], x_df, append = isfile(paths_save[:xmin]))
    if haskey(paths_save, :trace) && !isnothing(res.ftrace) && !isempty(res.ftrace)
        trace_df = DataFrame(Matrix(reduce(vcat, res.xtrace')), x_names)
        trace_df[!, "ftrace"] = res.ftrace
        trace_df[!, "startguess"] = repeat([i], length(res.ftrace))
        CSV.write(paths_save[:trace], trace_df, append = isfile(paths_save[:trace]))
    end
    return nothing
end

function _load_packages_workers(workers::Vector{Int64})::Nothing
    isempty(workers) && return nothing
    loaded = Set(Symbol(k.name) for k in keys(Base.loaded_modules))
    @eval Distributed.@everywhere $workers Base.eval(Main, :(using PEtab))
    for pkg in filter(in(loaded), _SUPPORTED_PACKAGES)
        ex = :(using $(pkg))
        @eval Distributed.@everywhere $workers Base.eval(Main, $ex)
    end
    return nothing
end

_create_workers(n::Int64)::Vector{Int64} = Distributed.addprocs(n)

function _remove_workers(pids::Vector{Int64})::Nothing
    Distributed.rmprocs(pids)
    return nothing
end

"""
    _labels_to_componentarray(labels::Vector{String}, values::Vector{Float64})

Reconstruct a `ComponentVector` from a vector of dot-separated labels and corresponding values.

Scalar fields are inferred from bare labels (e.g. `"alpha"`), arrays from indexed labels
(e.g. `"net1.layer1.weight[1,2]"`), and nested structures from dot-separated paths.

Written with help of Claude.
"""
function _labels_to_componentarray(labels, values)
    ca = _build_nested(labels, values) |> _to_componentarray
    key_order = Symbol.([first(split(_parse_label(l)[1][1], ".")) for l in labels]) |>
        unique
    return ca[key_order]
end

function _parse_label(label::String)
    m = match(r"^(.*?)\[([0-9,]+)\]$", label)
    isnothing(m) && return (split(label, "."), Int[])
    return (split(m.captures[1], "."), parse.(Int, split(m.captures[2], ",")))
end

function _build_nested(labels, values)
    groups = Dict{String, Vector}()
    for (label, val) in zip(labels, values)
        parts, idx = _parse_label(label)
        push!(get!(groups, String(parts[1]), []), (parts[2:end], idx, val))
    end
    return Dict(
        key => begin
                all_leaf = all(isempty(e[1]) for e in entries)
                all_scalar = all_leaf && isempty(entries[1][2])
                if all_scalar
                    entries[1][3]
            elseif all_leaf
                    shape = Tuple(maximum(e[2][i] for e in entries) for i in 1:length(entries[1][2]))
                    arr = zeros(Float64, shape...)
                    foreach(e -> (arr[e[2]...] = e[3]), entries)
                    length(shape) == 1 ? vec(arr) : arr
            else
                    sub_labels = [join(e[1], ".") * (isempty(e[2]) ? "" : "[$(join(e[2], ","))]") for e in entries]
                    _build_nested(sub_labels, [e[3] for e in entries])
            end
            end for (key, entries) in groups
    )
end

function _to_componentarray(d::Dict)
    nt = (Symbol(k) => (v isa Dict ? _to_componentarray(v) : v) for (k, v) in d)
    return ComponentVector(; nt...)
end
