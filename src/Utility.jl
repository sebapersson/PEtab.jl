"""
    get_fitted_ps(res::Union{PEtabOptimisationResult, PEtabMultistartOptimisationResult}, 
                  petab_problem::PEtabODEProblem,
                  condition_id::Union{String, Symbol}; 
                  retmap=true)

From a fitted PEtab model retrieve the ODE parameters to simulate the model for the specified `condition_id`.

If `retmap=true`, a parameter vector is returned; otherwise, a vector is returned.
"""
function get_fitted_ps(res::Union{PEtabOptimisationResult, PEtabMultistartOptimisationResult}, 
                       petab_problem::PEtabODEProblem,
                       condition_id::Union{String, Symbol}; 
                       retmap=true)

    u0, p = _get_fitted_parameters(res, petab_problem, condition_id, nothing, retmap)
    return p
end


"""
    get_fitted_u0(res::Union{PEtabOptimisationResult, PEtabMultistartOptimisationResult}, 
                  petab_problem::PEtabODEProblem,
                  condition_id::Union{String, Symbol}; 
                  pre_eq_id::Union{String, Symbol, Nothing}=nothing, 
                  retmap=true)

From a fitted PEtab model retrieve the inital values (u0) to simulate the model for the specified `condition_id`.

If a `pre_eq_id` is provided, the initial values are taken from the pre-equilibration simulation corresponding to 
`pre_eq_id`. If there are potential overrides of initial values in the simulation conditions, they take priority over 
the pre-equilibrium simulation.

If `retmap=true`, a parameter vector is returned; otherwise, a vector is returned.
"""
function get_fitted_u0(res::Union{PEtabOptimisationResult, PEtabMultistartOptimisationResult}, 
                       petab_problem::PEtabODEProblem,
                       condition_id::Union{String, Symbol}; 
                       pre_eq_id::Union{String, Symbol, Nothing}=nothing, 
                       retmap::Bool=true)
    u0, p = _get_fitted_parameters(res, petab_problem, condition_id, pre_eq_id, retmap)
    return u0
end


function _get_fitted_parameters(res::Union{PEtabOptimisationResult, PEtabMultistartOptimisationResult}, 
                                petab_problem::PEtabODEProblem,
                                condition_id::Union{String, Symbol},
                                pre_eq_id::Union{String, Symbol, Nothing}, 
                                retmap::Bool=true)


    @unpack θ_indices, petab_model, simulation_info, ode_problem = petab_problem

    # Sanity check input
    _c_id = Symbol(condition_id)
    @assert _c_id ∈ simulation_info.simulation_condition_id "A simulation condition id was given that could not be found in among the petab model conditions."
    if !isnothing(pre_eq_id)
        _pre_eq_id = Symbol(pre_eq_id)
        @assert _pre_eq_id ∈ simulation_info.pre_equilibration_condition_id "A pre-equilbration simulation condition id was given that could not be found in among the petab model conditions."
    else
        _pre_eq_id = nothing
    end

    p, ps = ode_problem.p[:], first.(petab_model.parameter_map)
    u0, u0s = ode_problem.u0[:], first.(petab_model.state_map)

    θT = transformθ(res.xmin, θ_indices.θ_names, θ_indices)
    θ_dynamic, θ_observable, θ_sd, θ_non_dynamic = splitθ(θT, θ_indices)

    # Set constant model parameters 
    change_ode_parameters!(p, u0, θ_dynamic, θ_indices, petab_model)

    # In case of no pre-eq condition we are done after changing to the condition s
    # Condition specific parameters 
    if isnothing(pre_eq_id)            
        _change_simulation_condition!(p, u0, _c_id, θ_dynamic, petab_model, θ_indices)
        _u0 = retmap ? Pair.(u0s, u0) : u0
        _p = retmap ? Pair.(ps, p) : p
        return _u0, _p
    end

    # For models with pre-eq in order to correctly return the initial values the model 
    # must first be simulated to steady state, and following the parameters must 
    # be set correctly
    u_ss = Vector{Float64}(undef, length(u0)) 
    u_t0 = Vector{Float64}(undef, length(u0))
    change_simulation_condition! = (p_ode_problem, u0, conditionId) -> _change_simulation_condition!(p_ode_problem, u0, conditionId, θ_dynamic, petab_model, θ_indices)

    pre_eq_sol = solve_ode_pre_equlibrium!(u_ss,
                                           u_t0,
                                           remake(ode_problem, p=p, u0=u0),
                                           change_simulation_condition!,
                                           _pre_eq_id,
                                           petab_problem.ode_solver,
                                           petab_problem.ss_solver,
                                           petab_model.convert_tspan)

    change_simulation_condition!(p, u0, _c_id)
    # Sometimes the experimentaCondition-file changes the initial values for a state
    # whose value was changed in the preequilibration-simulation. The experimentaCondition
    # value is prioritized by only changing u0 to the steady state value for those states
    # that were not affected by change to shift_expid.
    has_not_changed = u0 .== u_t0
    u0[has_not_changed] .= u_ss[has_not_changed]

    # According to the PEtab standard we can sometimes have that initial assignment is overridden for
    # pre-eq simulation, but we do not want to override for main simulation which is done automatically
    # by change_simulation_condition!. These cases are marked as NaN
    u0[isnan.(ode_problem.u0)] .= u_ss[isnan.(u0)]

    _u0 = retmap ? Pair.(u0s, u0) : u0
    _p = retmap ? Pair.(ps, p) : p
    return _u0, _p
end