"""
    get_odesol(res, prob::PEtabODEProblem; condition = nothing)::ODESolution

Retrieve the `ODESolution` from simulating the ODE model in `prob`. `res` can
be a parameter estimation result (e.g., `PEtabMultistartResult`) or a `Vector` with
parameters in the order expected by `prob` (see [`get_x`](@ref)).

For information on keyword arguments see [`get_ps`](@ref).

See also: [`get_u0`](@ref) and [`get_odeproblem`](@ref).
"""
function get_odesol(res::EstimationResult, prob::PEtabODEProblem;
                    condition::Union{ConditionExp, Nothing} = nothing)::ODESolution
    @unpack model_info, probinfo = prob
    oprob, cbs = get_odeproblem(res, prob; condition = condition)

    @unpack solver, abstol, reltol = probinfo.solver
    return solve(oprob, solver, abstol = abstol, reltol = reltol, callback = cbs)
end

"""
    get_odeproblem(res, prob::PEtabODEProblem; condition = nothing) -> (ode_prob, cbs)

Retrieve the `ODEProblem` and cbs (`CallbackSet`) for simulating the ODE model in
`prob`. `res` can be a parameter estimation result (e.g., `PEtabMultistartResult`) or a
`Vector` with parameters in the order expected by `prob` (see [`get_x`](@ref)).

For information on keyword arguments see [`get_ps`](@ref).

See also: [`get_u0`](@ref) and [`get_odesol`](@ref).
"""
function get_odeproblem(res::EstimationResult, prob::PEtabODEProblem;
                        condition::Union{ConditionExp, Nothing} = nothing)
    @unpack model_info, probinfo = prob
    simulation_id = _get_simulation_id(condition, model_info)

    u0, ps = _get_ps_u0(res, prob, condition, false)
    tmax = _get_tmax(condition, model_info)
    cbs = model_info.model.callbacks[simulation_id]

    odefun = ODEFunction(_get_system(prob.model_info.model.sys))
    odeprob = ODEProblem(odefun, u0, [0.0, tmax], ps)
    return odeprob, cbs
end

"""
    get_system(res, prob::PEtabODEProblem; kwargs...) -> (sys, p, u0, callbacks)

Retrieve the dynamic model system, parameter map (`p`), initial species map (`u0`), and
callbacks (`CallbackSet`) for the model in `prob`. The argument `res` can be a parameter
estimation result (e.g., `PEtabMultistartResult`) or a `Vector` of parameters in the order
expected by `prob` (see [`get_x`](@ref)).

The system type returned depends on the input to `PEtabModel`. If the model is provided as
a `ReactionSystem`, a `ReactionSystem` is returned. The same applies for an `ODESystem`. If
the model is provided via an SBML file, a `ReactionSystem` is returned.

For information on keyword arguments, see [`get_ps`](@ref).

See also: [`get_u0`](@ref) and [`get_odesol`](@ref).
"""
function get_system(res::EstimationResult, prob::PEtabODEProblem;
                    condition::Union{ConditionExp, Nothing} = nothing)
    @unpack sys, callbacks, paths = prob.model_info.model
    if haskey(paths, :SBML)
        prn, _ = load_SBML(paths[:SBML])
        sys = prn.rn
    end
    u0, p = _get_ps_u0(res, prob, condition, true)
    return sys, u0, p, callbacks
end

"""
    get_u0(res, prob::PEtabODEProblem; condition = nothing, retmap = true)

Retrieve the `ODEProblem` initial values for simulating the ODE model in `prob`. `res` can
be a parameter estimation result (e.g., `PEtabMultistartResult`) or a `Vector` with
parameters in the order expected by `prob` (see [`get_x`](@ref)).

For information on keyword arguments see [`get_ps`](@ref).

See also [`get_odeproblem`](@ref) and [`get_odesol`](@ref).
"""
function get_u0(res::EstimationResult, prob::PEtabODEProblem;
                condition::Union{ConditionExp, Nothing} = nothing, retmap::Bool = true)
    u0, _ = _get_ps_u0(res, prob, condition, retmap)
    return u0
end

"""
    get_ps(res, prob::PEtabODEProblem; condition = nothing, retmap = true)

Retrieve parameter values for simulating the ODE model in `prob`. `res` may be a parameter
estimation result (e.g. `PEtabMultistartResult`) or a parameter vector in the order
expected by `prob` (see [`get_x`](@ref)).

# Keyword Arguments
- `condition`: Simulation condition to retrieve parameters for. Ids are `Symbol`/
  `String`. If `nothing` (default), the first simulation condition is used. Format depends
  on whether the model has pre-equilibration:
  - No pre-equilibration: `simulation_id` (e.g. `:cond1`).
  - With pre-equilibration: `pre_eq_id => simulation_id` (e.g. `:pre_eq1 => :cond1`).
- `retmap = true`: If `true`, return a vector of pairs `[k1 => v1, ...]` suitable for
  passing as `p` when constructing an `ODEProblem`. If `false`, return a parameter vector.
  Only applicable for `get_ps` and `get_u0`.

See also: [`get_u0`](@ref), [`get_odeproblem`](@ref), [`get_odesol`](@ref).
"""
function get_ps(res::EstimationResult, prob::PEtabODEProblem;
                condition::Union{ConditionExp, Nothing} = nothing, retmap::Bool = true)
    _, p = _get_ps_u0(res, prob, condition, retmap)
    return p
end

function _get_ps_u0(res::EstimationResult, prob::PEtabODEProblem,
                    condition::Union{ConditionExp, Nothing}, retmap::Bool)
    @unpack probinfo, model_info = prob
    @unpack xindices, model, simulation_info = model_info
    @unpack solver, ss_solver, cache, odeproblem = probinfo

    simulation_id = _get_simulation_id(condition, model_info)
    pre_equilibration_id = _get_pre_equilibration_id(condition, model_info)
    _check_condition_ids(simulation_id, pre_equilibration_id, model_info)

    x_transformed = transform_x(_get_x(res), xindices.xids[:estimate], xindices)
    xdynamic, _, _, _ = split_x(x_transformed, xindices)

    # System parameters and their associated ids
    odeproblem = remake(odeproblem, p = convert.(eltype(xdynamic), odeproblem.p),
        u0 = convert.(eltype(xdynamic), odeproblem.u0))
    p = odeproblem.p[:]
    ps = xindices.xids[:sys]
    u0 = odeproblem.u0[:]
    u0s = first.(model_info.model.speciemap)[1:length(u0)]

    if isnothing(pre_equilibration_id)
        oprob = _switch_condition(odeproblem, simulation_id, xdynamic, model_info, cache,
                                  false)

    else
        # For models with pre-eq the model must first be simulated to steady state, and
        # following the steady-state the parameters must be correctly set
        u_ss = Vector{Float64}(undef, length(u0))
        u_t0 = Vector{Float64}(undef, length(u0))
        oprob_preeq = _switch_condition(odeproblem, pre_equilibration_id, xdynamic,
                                        model_info, cache, false)
        _ = solve_pre_equilibrium!!(u_ss, u_t0, oprob_preeq, simulation_info, solver, ss_solver, pre_equilibration_id, true)

        experiment_id = _get_experiment_id(simulation_id, pre_equilibration_id)
        oprob = _switch_condition(odeproblem, experiment_id, xdynamic, model_info, cache, true; simulation_id = simulation_id)
        has_not_changed = oprob.u0 .== u_t0
        oprob.u0[has_not_changed] .= u_ss[has_not_changed]
        oprob.u0[isnan.(oprob.u0)] .= u_ss[isnan.(u0)]
    end

    u0 = deepcopy(oprob.u0)
    p = deepcopy(oprob.p)
    _u0 = retmap ? Pair.(u0s, u0) : u0
    _p = retmap ? Pair.(ps, p) : p
    # These parameters are added to a mutated system for gradient computations, but
    # should not be exposed to the user if the model is defined in Julia
    if model.defined_in_julia
        ip = findall(x -> !occursin("__init__", x), string.(ps))
        return _u0, _p[ip]
    end
    return _u0, _p
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
    xdynamic_ps = transform_x(cache.xdynamic, xindices, :xdynamic, cache)

    _ = solve_conditions!(model_info, xdynamic_ps, probinfo;
                          ntimepoints_save = ntimepoints_save,
                          save_observed_t = save_observed_t)
    return model_info.simulation_info.odesols
end

function _get_simulation_id(condition::Union{ConditionExp, Nothing}, model_info::ModelInfo)::Symbol
    if isnothing(condition)
        return model_info.simulation_info.conditionids[:simulation][1]
    end

    return condition isa Pair ? Symbol(condition.second) : Symbol(condition)
end

function _get_pre_equilibration_id(condition::Union{ConditionExp, Nothing}, model_info::ModelInfo)::Union{Symbol, Nothing}
    simulation_info = model_info.simulation_info
    if isnothing(condition) && simulation_info.has_pre_equilibration
        return model_info.simulation_info.conditionids[:pre_equilibration][1]
    end

    if !(condition isa Pair)
        return nothing
    else
        return Symbol(condition.first)
    end
end

function _check_condition_ids(simulation_id::Symbol, ::Nothing, model_info::ModelInfo)::Nothing
    valid_ids = model_info.simulation_info.conditionids[:experiment]
    if !in(simulation_id, valid_ids)
        throw(PEtabInputError("Condition id `$(simulation_id)` not found in the PEtab \
            problem. Valid ids: $(valid_ids)."))
    end
    return nothing
end
function _check_condition_ids(simulation_id::Symbol, pre_equilibration_id::Symbol, model_info::ModelInfo)::Nothing
    @unpack conditionids = model_info.simulation_info
    experiment_id = _get_experiment_id(simulation_id, pre_equilibration_id)

    valid_ids = conditionids[:experiment]
    valid_pairs = conditionids[:pre_equilibration] .=> conditionids[:simulation]
    if !in(experiment_id, valid_ids)
        throw(PEtabInputError("Combination pre-equilibration id and simulation id \
            $(pre_equilibration_id => simulation_id) ids not found in the PEtab problem. \
            Valid pairs are: $(valid_pairs)."))
    end
    return nothing
end

function _get_system(sys::ReactionSystem)::ODESystem
    return complete(convert(ODESystem, sys))
end
function _get_system(sys::ODESystem)::ODESystem
    return sys
end

_get_x(x::Union{AbstractVector, ComponentVector}) = x
function _get_x(res::Union{PEtabOptimisationResult, PEtabMultistartResult})
    return res.xmin
end
