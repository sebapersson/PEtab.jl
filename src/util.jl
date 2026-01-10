"""
    get_odesol(res, prob::PEtabODEProblem; kwargs...)::ODESolution

Retrieve the `ODESolution` from simulating the ODE model in `prob`. `res` can
be a parameter estimation result (e.g., `PEtabMultistartResult`) or a `Vector` with
parameters in the order expected by `prob` (see [`get_x`](@ref)).

For information on keyword arguments see [`get_ps`](@ref).

See also: [`get_u0`](@ref) and [`get_odeproblem`](@ref).
"""
function get_odesol(res::EstimationResult, prob::PEtabODEProblem;
                    condition::Union{ConditionExp, Nothing} = nothing,
                    experiment::Union{ConditionExp, Nothing} = nothing)::ODESolution
    @unpack model_info, probinfo = prob
    oprob, cbs = get_odeproblem(res, prob; condition = condition, experiment = experiment)

    @unpack solver, abstol, reltol = probinfo.solver
    return solve(oprob, solver, abstol = abstol, reltol = reltol, callback = cbs)
end

"""
    get_odeproblem(res, prob::PEtabODEProblem; kwargs...) -> (ode_prob, cbs)

Retrieve the `ODEProblem` and cbs (`CallbackSet`) for simulating the ODE model in
`prob`. `res` can be a parameter estimation result (e.g., `PEtabMultistartResult`) or a
`Vector` with parameters in the order expected by `prob` (see [`get_x`](@ref)).

For information on keyword arguments see [`get_ps`](@ref).

See also: [`get_u0`](@ref) and [`get_odesol`](@ref).
"""
function get_odeproblem(res::EstimationResult, prob::PEtabODEProblem;
                        condition::Union{ConditionExp, Nothing} = nothing,
                        experiment::Union{ConditionExp, Nothing} = nothing)
    @unpack model_info, probinfo = prob
    u0, ps = _get_ps_u0(res, prob, condition, experiment, false)

    simulation_id = _get_simulation_id(condition, experiment, model_info)
    tstart = _get_start(condition, experiment, model_info)
    tmax = _get_tmax(condition, experiment, model_info)
    cbs = model_info.model.callbacks[simulation_id]

    odefun = ODEFunction(_get_system(prob.model_info.model.sys))
    odeprob = ODEProblem(odefun, u0, [tstart, tmax], ps)
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
                    condition::Union{ConditionExp, Nothing} = nothing,
                    experiment::Union{ConditionExp, Nothing} = nothing)
    @unpack sys, paths, callbacks = prob.model_info.model
    if haskey(paths, :SBML)
        prn, _ = load_SBML(paths[:SBML])
        sys = prn.rn
    end

    u0, p = _get_ps_u0(res, prob, condition, experiment, true)
    simulation_id = _get_simulation_id(condition, experiment, prob.model_info)
    return sys, u0, p, callbacks[simulation_id]
end

"""
    get_u0(res, prob::PEtabODEProblem; kwargs...)

Retrieve the `ODEProblem` initial values for simulating the ODE model in `prob`. `res` can
be a parameter estimation result (e.g., `PEtabMultistartResult`) or a `Vector` with
parameters in the order expected by `prob` (see [`get_x`](@ref)).

For information on keyword arguments see [`get_ps`](@ref).

See also [`get_odeproblem`](@ref) and [`get_odesol`](@ref).
"""
function get_u0(res::EstimationResult, prob::PEtabODEProblem; retmap::Bool = true,
                condition::Union{ConditionExp, Nothing} = nothing,
                experiment::Union{ConditionExp, Nothing} = nothing)
    u0, _ = _get_ps_u0(res, prob, condition, experiment, retmap)
    return u0
end

"""
    get_ps(res, prob::PEtabODEProblem; kwargs...)

Return parameter values for simulating the ODE model in `prob`.

`res` can be a parameter-estimation result (e.g. `PEtabMultistartResult`) or a parameter
vector in the internal order expected by `prob` (see [`get_x`](@ref)).

# Keyword arguments
- `condition`: Simulation condition to retrieve parameters for. Used for PEtab v1 problems
  and problems defined via the Julia interface. If the model has pre-equilibration, pass
  `pre_eq_id => simulation_id`; otherwise pass `simulation_id`. IDs may be `String` or
  `Symbol`.
- `experiment`: Experiment time course to retrieve parameters for (`String` or `Symbol`).
  Only applicable for problem in the PEtab v2 standard format.
- `retmap = true`: If `true`, return a vector of pairs `p = [k1 => v1, ...]` suitable for
  `ODEProblem(; p)`. If `false`, return a parameter vector.

!!! note
    `condition` and `experiment` are mutually exclusive (provide at most one).

See also: [`get_u0`](@ref), [`get_odeproblem`](@ref), [`get_odesol`](@ref).
"""
function get_ps(res::EstimationResult, prob::PEtabODEProblem; retmap::Bool = true,
                condition::Union{ConditionExp, Nothing} = nothing,
                experiment::Union{ConditionExp, Nothing} = nothing)
    _, p = _get_ps_u0(res, prob, condition, experiment, retmap)
    return p
end

function _get_ps_u0(res::EstimationResult, prob::PEtabODEProblem,
                    condition::Union{ConditionExp, Nothing},
                    experiment::Union{ConditionExp, Nothing}, retmap::Bool)
    @unpack probinfo, model_info = prob
    @unpack xindices, model, simulation_info = model_info
    @unpack solver, ss_solver, cache, odeproblem = probinfo

    _check_experiment_id(condition, experiment, model_info)
    simulation_id = _get_simulation_id(condition, experiment, model_info)
    pre_equilibration_id = _get_pre_equilibration_id(condition, experiment, model_info)
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


function _get_simulation_id(condition::Union{ConditionExp, Nothing}, experiment::Union{ConditionExp, Nothing}, model_info::ModelInfo)::Symbol
    conditionids = model_info.simulation_info.conditionids
    if isnothing(experiment) && isnothing(condition)
        return conditionids[:simulation][1]
    end

    if !isnothing(condition) && isnothing(experiment)
        return condition isa Pair ? Symbol(condition.second) : Symbol(condition)
    end

    ic = findfirst(startswith("$(experiment)_"), string.(conditionids[:simulation]))
    return conditionids[:simulation][ic]
end

function _get_pre_equilibration_id(condition::Union{ConditionExp, Nothing}, experiment::Union{ConditionExp, Nothing}, model_info::ModelInfo)::Union{Symbol, Nothing}
    @unpack has_pre_equilibration, conditionids = model_info.simulation_info
    if isnothing(condition) && isnothing(experiment) && has_pre_equilibration
        return conditionids[:pre_equilibration][1]
    end

    if !isnothing(condition) && isnothing(experiment)
        if !(condition isa Pair)
            return nothing
        else
            return Symbol(condition.first)
        end
    end

    ic = findfirst(startswith("$(experiment)_"), string.(conditionids[:pre_equilibration]))
    if isnothing(ic)
        nothing
    else
        return conditionids[:pre_equilibration][ic]
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

function _check_experiment_id(condition::Union{ConditionExp, Nothing}, experiment::Union{ConditionExp, Nothing}, model_info::ModelInfo)::Nothing
    if isnothing(experiment) && isnothing(condition)
        return nothing
    end

    petab_version = _get_version(model_info)
    if petab_version == "1.0.0" && !isnothing(experiment)
        throw(ArgumentError("`experiment` keyword is only valid for problem in the PEtab \
            v2 standard format"))
    end
    if petab_version == "1.0.0"
        return nothing
    end

    if !isnothing(condition)
        throw(ArgumentError("For PEtab v2 problems the `condition` keyword is not \
            supported; use `experiment` to select an experimental timecourse."))
    end

    if isempty(model_info.model.petab_tables[:experiments]) && !isnothing(experiment)
        throw(PEtabInputError("PEtab problem contains no experiment IDs (empty experiments \
            table); specifying experiment is invalid. Use experiment = nothing."))
    end

    valid_ids = model_info.model.petab_tables[:experiments].experimentId
    if !(string(experiment) in valid_ids)
        throw(PEtabInputError("Experiment id $(experiment) not found in the PEtab \
            problem's experiments table. Valid ids are: $(Symbol.(valid_ids))."))
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
