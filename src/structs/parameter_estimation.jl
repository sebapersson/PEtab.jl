"""
    IpoptOptimizer(LBFGS::Bool)

Setup the [Ipopt](https://coin-or.github.io/Ipopt/) Interior-point Newton method optmizer
for parameter estimation.

Ipopt can be configured to use either the Hessian method from the `PEtabODEProblem`
(`LBFGS=false`) or an LBFGS scheme (`LBFGS=true`). For setting other Ipopt options,
see [`IpoptOptions`](@ref).

See also [`calibrate`](@ref) and [`calibrate_multistart`](@ref).

## Description

Ipopt is an Interior-point Newton method for constrained non-linear optimization problems.
More information on the algorithm can be found in [1].

1. Wächter and Biegler, Mathematical programming, pp 25-57 (2006)
"""
struct IpoptOptimizer
    LBFGS::Bool
end

"""
    IpoptOptions(; kwargs...)

Options for parameter estimation with `IpoptOptimizer`.

More details on the options can be found in the Ipopt
[documentation](https://coin-or.github.io/Ipopt/).

See also [`IpoptOptimizer`](@ref), [`calibrate`](@ref), and
[`calibrate_multistart`](@ref).

## Keyword Arguments

- `print_level = 0`: Output verbosity level (valid values are 0 ≤ print_level ≤ 12)
- `max_iter = 1000`: Maximum number of iterations
- `tol = 1e-8`: Relative convergence tolerance
- `acceptable_tol = 1e-6`: Acceptable relative convergence tolerance
- `max_wall_time 1e20`: Maximum wall time optimization is allowed to run
- `acceptable_obj_change_tol 1e20`: Acceptance stopping criterion based on objective
    function change.
"""
struct IpoptOptions
    print_level::Int64
    max_iter::Int64
    tol::Float64
    acceptable_tol::Float64
    max_wall_time::Float64
    acceptable_obj_change_tol::Float64
end
function IpoptOptions(; print_level::Int64 = 0, max_iter::Int64 = 1000, tol::Float64 = 1e-8,
                      acceptable_tol::Float64 = 1e-6, max_wall_time::Float64 = 1e20,
                      acceptable_obj_change_tol::Float64 = 1e20)
    return IpoptOptions(print_level, max_iter, tol, acceptable_tol, max_wall_time,
                        acceptable_obj_change_tol)
end

"""
    PEtabOptimisationResult

Parameter estimation statistics from single-start optimization with `calibrate`.

See also: [`calibrate`](@ref)

## Fields
- `xmin`: Minimizing parameter vector found by the optimization.
- `fmin`: Minimum objective value found by the optimization.
- `x0`: Starting parameter vector.
- `alg`: Parameter estimation algorithm used.
- `niterations`: Number of iterations for the optimization.
- `runtime`: Runtime in seconds for the optimization.
- `xtrace`: Parameter vector optimization trace. Empty if `save_trace = false` was
    provided to `calibrate`.
- `ftrace`: Objective function optimization trace. Empty if `save_trace = false` was
    provided to `calibrate`.
- `converged`: Convergence flag from `alg`.
- `original`: Original result struct returned by `alg`. For example, if `alg = IPNewton()`
    from Optim.jl, `original` is the Optim return struct.
"""
struct PEtabOptimisationResult
    xmin::ComponentArray{Float64}
    fmin::Float64
    x0::ComponentArray{Float64}
    alg::Symbol
    niterations::Int64
    runtime::Float64
    xtrace::Vector{Vector{Float64}}
    ftrace::Vector{Float64}
    converged::Any
    original::Any
end

"""
    PEtabMultistartResult

Parameter estimation statistics from multi-start optimization with `calibrate_multistart`.

See also [`calibrate_multistart`](@ref) and [`PEtabOptimisationResult`](@ref).

## Fields
- `xmin`: Best minimizer across all runs.
- `fmin`: Best minimum across all runs.
- `alg`: Parameter estimation algorithm.
- `nmultistarts`: Number of parameter estimation runs.
- `sampling_method`: Sampling method used for generating starting points.
- `dirsave`: Path of directory where parameter estimation run statistics are saved if
    `dirsave` was provided to `calibrate_multistart`.
- `runs`: Vector of `PEtabOptimisationResult` with the parameter estimation results
    for each run.


    PEtabMultistartResult(dirres::String; which_run::String="1")

Import multistart parameter estimation results saved at `dirres`.

Each time a new optimization run is performed, results are saved with unique numerical
endings. Results from a specific run can be retrieved by specifying the numerical ending
with `which_run`.
"""
struct PEtabMultistartResult
    xmin::ComponentArray{Float64}
    fmin::Float64
    alg::Symbol
    nmultistarts::Int
    sampling_method::String
    dirsave::Union{String, Nothing}
    runs::Vector{PEtabOptimisationResult}
end
function PEtabMultistartResult(dirres::String;
                               which_run::Integer = 1)::PEtabMultistartResult
    @assert isdir(dirres) "Directory $dirres does not exist"

    i = which_run |> string
    path_res = joinpath(dirres, "results$i.csv")
    path_parameters = joinpath(dirres, "xmins$i.csv")
    path_startguess = joinpath(dirres, "startguesses$i.csv")
    path_trace = joinpath(dirres, "trace$i.csv")
    @assert isfile(path_res) "Result file $(path_res) does not exist"
    @assert isfile(path_parameters) "Optimal parameters file $(path_parameters) does not exist"
    @assert isfile(path_startguess) "Startguess file $(path_startguess) does not exist"

    res_df = CSV.read(path_res, DataFrame)
    x_df = CSV.read(path_parameters, DataFrame)
    startguesses_df = CSV.read(path_startguess, DataFrame)
    if isfile(path_trace)
        trace_df = CSV.read(path_trace, DataFrame)
    end
    runs = Vector{PEtab.PEtabOptimisationResult}(undef, nrow(res_df))
    for i in eachindex(runs)
        if isfile(path_trace)
            trace_df_i = trace_df[findall(x -> x == i, trace_df[!, :startguess]), :]
            _ftrace = trace_df_i[!, :ftrace]
            _xtrace = [Vector{Float64}(trace_df_i[i, 1:(end - 2)])
                       for i in 1:size(trace_df_i)[1]]
        else
            _ftrace = Vector{Float64}(undef, 0)
            _xtrace = Vector{Vector{Float64}}(undef, 0)
        end
        xnames = propertynames(x_df)[1:(end - 1)] |> collect
        xmin = ComponentArray(; (xnames .=> x_df[i, 1:(end - 1)] |>
                                            Vector{Float64})...)
        xstart = ComponentArray(;
                                (xnames .=>
                                     startguesses_df[i, 1:(end - 1)] |>
                                     Vector{Float64})...)
        runs[i] = PEtabOptimisationResult(xmin, res_df[i, :fmin], xstart,
                                          Symbol(res_df[i, :alg]), res_df[i, :niterations],
                                          res_df[i, :runtime], _xtrace, _ftrace,
                                          res_df[i, :converged], nothing)
    end
    bestrun = runs[argmin([isnan(r.fmin) ? Inf : r.fmin for r in runs])]
    fmin = bestrun.fmin
    xmin = bestrun.xmin
    nmultistarts = length(runs)
    return PEtabMultistartResult(xmin, fmin, bestrun.alg, nmultistarts, "", dirres, runs)
end
