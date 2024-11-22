"""
    get_odesol(res, prob::PEtabODEProblem; kwargs...)::ODESolution

Retrieve the `ODESolution` from simulating the ODE model in `prob`. `res` can
be a parameter estimation result (e.g., `PEtabMultistartResult`) or a `Vector` with
parameters in the order expected by `prob` (see [`get_x`](@ref)).

For information on keyword arguements see [`get_ps`](@ref).

See also: [`get_u0`](@ref) and [`get_odeproblem`](@ref).
"""
function get_odesol(res::EstimationResult, prob::PEtabODEProblem;
                    cid::Union{String, Symbol, Nothing} = nothing,
                    preeq_id::Union{String, Symbol, Nothing} = nothing)::ODESolution
    @unpack model_info, probinfo = prob
    _cid = _get_cid(cid, model_info)
    _preeq_id = _get_preeq_id(preeq_id, model_info)
    oprob, cbset = get_odeproblem(res, prob; cid = _cid, preeq_id = _preeq_id)
    @unpack solver, abstol, reltol = probinfo.solver
    return solve(oprob, solver, abstol = abstol, reltol = reltol, callback = cbset)
end

"""
    get_odeproblem(res, prob::PEtabODEProblem; kwargs...) -> (sol, callbacks)

Retrieve the `ODEProblem` and callbacks (`CallbackSet`) for simulating the ODE model in
`prob`. `res` can be a parameter estimation result (e.g., `PEtabMultistartResult`) or a
`Vector` with parameters in the order expected by `prob` (see [`get_x`](@ref)).

For information on keyword arguements see [`get_ps`](@ref).

See also: [`get_u0`](@ref) and [`get_odesol`](@ref).
"""
function get_odeproblem(res::EstimationResult, prob::PEtabODEProblem;
                        cid::Union{String, Symbol, Nothing} = nothing,
                        preeq_id::Union{String, Symbol, Nothing} = nothing)
    @unpack model_info, probinfo = prob
    u0, p = _get_ps_u0(res, prob, cid, preeq_id, true)
    tmax = model_info.simulation_info.tmaxs[_get_cid(cid, model_info)]
    oprob = ODEProblem(model_info.model.sys, u0, [0.0, tmax], p, jac = true)
    oprob = remake(oprob, p = oprob.p.tunable)
    return oprob, model_info.model.callbacks
end

"""
    get_ps(res, prob::PEtabODEProblem; kwargs...)

Retrieve the `ODEProblem` parameter values for simulating the ODE model in `prob`. `res` can
be a parameter estimation result (e.g., `PEtabMultistartResult`) or a `Vector` with
parameters in the order expected by `prob` (see [`get_x`](@ref)).

See also: [`get_u0`](@ref), [`get_odeproblem`](@ref), [`get_odesol`](@ref).

## Keyword Arguments

- `retmap=true`: Whether to return the values as a map in the form `[k1 => val1, ...]`. Such
    a map can be directly used when building an `ODEProblem`. If `false`, a `Vector` is
    returned. This keyword is only applicable for `get_u0` and `get_ps`.
- `cid::Symbol`: Which simulation condition to return parameters for. If not provided,
    defaults to the first simulation condition. For other `get` functions, the
    `ODEProblem`, `u0`, or `ODESolution` for the specified `cid` is returned.
- `preeq_id`: Which potential pre-equilibration (steady-state) simulation id to use.
    If a valid `preeq_id` is provided, the ODE is first simulated to steady state for
    `preeq_id`. Then the model shifts to `cid`, and the parameters for `cid` are returned.
    For other `get` functions, the  `ODEProblem`, `u0`, or `ODESolution` for the
    specified `cid` is returned
"""
function get_ps(res::EstimationResult, prob::PEtabODEProblem;
                cid::Union{String, Symbol, Nothing} = nothing, retmap::Bool = true)
    _, p = _get_ps_u0(res, prob, cid, nothing, retmap)
    return p
end

"""
    get_u0(res, prob::PEtabODEProblem; kwargs...)

Retrieve the `ODEProblem` initial values for simulating the ODE model in `prob`. `res` can
be a parameter estimation result (e.g., `PEtabMultistartResult`) or a `Vector` with
parameters in the order expected by `prob` (see [`get_x`](@ref)).

For information on keyword arguements see [`get_ps`](@ref).

See also [`get_odeproblem`](@ref) and [`get_odesol`](@ref).
"""
function get_u0(res::EstimationResult, prob::PEtabODEProblem; retmap::Bool = true,
                cid::Union{String, Symbol, Nothing} = nothing,
                preeq_id::Union{String, Symbol, Nothing} = nothing)
    u0, _ = _get_ps_u0(res, prob, cid, preeq_id, retmap)
    return u0
end

function _get_ps_u0(res::EstimationResult, prob::PEtabODEProblem,
                    cid::Union{Nothing, Symbol, String},
                    preeq_id::Union{Nothing, Symbol, String}, retmap::Bool)
    @unpack probinfo, model_info = prob
    @unpack xindices, model, simulation_info = prob.model_info
    @unpack solver, ss_solver, cache, odeproblem, f_nns_preode = probinfo

    cid = _get_cid(cid, model_info)
    preeq_id = _get_preeq_id(preeq_id, model_info)

    # Extract model parameters
    if res isa Vector || res isa ComponentArray
        x_transformed = transform_x(res, xindices.xids[:estimate], xindices)
    else
        x_transformed = transform_x(res.xmin, xindices.xids[:estimate], xindices)
    end
    xdynamic, _, _, _, xnn = split_x(x_transformed, xindices, cache)

    # System parameters and their associated ids
    p = odeproblem.p[:]
    ps = xindices.xids[:sys]
    u0 = odeproblem.u0[:]
    u0s = first.(model_info.model.speciemap)[1:length(u0)]

    # Set constant model parameters
    _set_cond_const_parameters!(p, xdynamic, xindices)
    odeproblem.p .= p

    # In case of no pre-eq condition we are done after changing to the condition the
    # specific condition
    if isnothing(preeq_id)
        if model_info.simulation_info.has_pre_equilibration
            simid = cid
            # Needs a cid for the cache handling even when extracgint ps
            _cid = model_info.simulation_info.conditionids[:experiment][1]
        else
            simid = nothing
            _cid = cid
        end
        oprob = _switch_condition(odeproblem, _cid, xdynamic, xnn, model_info, cache,
                                  f_nns_preode; simid = simid)
    else
        # For models with pre-eq in order to correctly return the initial values the model
        # must first be simulated to steady state, and following the steady-state the
        # parameters must be correctly set
        u_ss = Vector{Float64}(undef, length(u0))
        u_t0 = Vector{Float64}(undef, length(u0))
        oprob_preeq = _switch_condition(odeproblem, preeq_id, xdynamic, model_info, cache, f_nns_preode)
        _ = solve_pre_equlibrium!(u_ss, u_t0, oprob_preeq, solver, ss_solver, true)
        # Setup the problem with correct initial values
        _cid = string(preeq_id) * string(cid) |> Symbol
        oprob = _switch_condition(odeproblem, _cid, xdynamic, xnn, model_info, cache;
                                  simid = cid)
        has_not_changed = oprob.u0 .== u_t0
        oprob.u0[has_not_changed] .= u_ss[has_not_changed]
        oprob.u0[isnan.(oprob.u0)] .= u_ss[isnan.(u0)]
    end

    u0 = oprob.u0 |> deepcopy
    p = oprob.p |> deepcopy
    _u0 = retmap ? Pair.(u0s, u0) : u0
    _p = retmap ? Pair.(ps, p) : p
    # These parameters are added to a mutated system for gradient computations, but
    # should not be exposed to the user
    ip = findall(x -> !occursin("__init__", x), string.(ps))
    return _u0, _p[ip]
end

"""
    get_x(prob::PEtabODEProblem; linear_scale = false)::ComponentArray

Get the nominal parameter vector with parameters in the correct order expected by `prob` for
parameter estimation/inference. Nominal values can optionally be specified when creating a
`PEtabParameter`, or in the parameters table if the problem is provided in the PEtab
standard format.

For ease of interaction (e.g., changing values), the parameter vector is returned as a
`ComponentArray`.  For how to interact with a `ComponentArray`, see the documentation and
the ComponentArrays.jl [documentation](https://github.com/jonniedie/ComponentArrays.jl).

See also [`PEtabParameter`](@ref).

## Keyword argument
- `linear_scale`: Whether to return parameters on the linear scale. By default, parameters
  are returned on the scale they are estimated, which by default is `log10` as this often
  improves parameter estimation performance.
"""
function get_x(prob::PEtabODEProblem; linear_scale::Bool = false)::ComponentArray
    return linear_scale ? prob.xnominal : prob.xnominal_transformed
end

"""
    solve_all_conditions(x, prob::PEtabODEProblem, solver; kwargs)

Solve the ODE model in `prob` for all simulation conditions with the provided ODE-solver.

`x` should be a `Vector` or `ComponentArray` with parameters in the order expected by
`prob` (see [`get_x`](@ref)).

# Keyword Arguments
- `abstol=1e-8`: Absolute tolerance for the ODE solver.
- `reltol=1e-8`: Relative tolerance for the ODE solver.
- `maxiters=1e4`: Maximum iterations for the ODE solver.
- `ntimepoints_save=0`: The number of time points at which to save the ODE solution for
    each condition. A value of 0 means the solution is saved at the solvers default time
    points.
- `save_observed_t=false`: When set to true, this option overrides `ntimepoints_save`
    and saves the ODE solution only at the time points where measurement data is available.

# Returns
- `odesols`: A dictionary containing the `ODESolution` for each simulation condition.
"""
function solve_all_conditions(x, prob::PEtabODEProblem, osolver; abstol = 1e-8,
                              reltol = 1e-8, maxiters = nothing, ntimepoints_save = 0,
                              save_observed_t = false)
    @unpack probinfo, model_info = prob
    xindices = model_info.xindices
    @unpack solver, cache = probinfo

    solver.abstol = abstol
    solver.reltol = reltol
    solver.solver = osolver
    if !isnothing(maxiters)
        solver.maxiters = maxiters
    end

    split_x!(x, xindices, cache)
    xdynamic_mech = get_tmp(cache.xdynamic_mech, x)
    xdynamic_ps = transform_x(xdynamic_mech, xindices, :xdynamic, cache)

    _ = solve_conditions!(model_info, xdynamic_ps, cache.xnn_dict, probinfo;
                          ntimepoints_save = ntimepoints_save,
                          save_observed_t = save_observed_t)
    return model_info.simulation_info.odesols
end

function _get_cid(cid::Union{Nothing, Symbol, String}, model_info::ModelInfo)::Symbol
    simulation_info = model_info.simulation_info
    if isnothing(cid)
        cid = simulation_info.conditionids[:simulation][1]
    end
    if cid isa String
        cid = Symbol(cid)
    end
    return cid
end

function _get_preeq_id(preeq_id::Union{Nothing, Symbol, String},
                       model_info::ModelInfo)::Union{Symbol, Nothing}
    if !isnothing(preeq_id) && preeq_id isa String
        preeq_id = Symbol(preeq_id)
    end
    return preeq_id
end
