function solve_conditions!(
        model_info::ModelInfo, xdynamic::AbstractVector,
        x_ml_models::Dict{Symbol, ComponentArray}, probinfo::PEtabODEProblemInfo;
        cids::Vector{Symbol} = [:all], ntimepoints_save::Int64 = 0,
        save_observed_t::Bool = true, dense_sol::Bool = false, track_callback::Bool = false,
        sensitivities::Bool = false, derivative::Bool = false
    )::Bool
    @unpack simulation_info, model, xindices = model_info
    @unpack float_tspan = model
    @unpack cache, ml_models_pre_ode = probinfo

    if derivative == true || sensitivities == true || track_callback == true
        odesols = simulation_info.odesols_derivatives
        ss_solver = probinfo.ss_solver_gradient
        osolver = probinfo.solver_gradient
        _oprob = probinfo.odeproblem_gradient
    else
        odesols = simulation_info.odesols
        ss_solver = probinfo.ss_solver
        osolver = probinfo.solver
        _oprob = probinfo.odeproblem
    end

    oprob = remake(_oprob, p = convert.(eltype(xdynamic), _oprob.p),
                   u0 = convert.(eltype(xdynamic), _oprob.u0))

    # In case the model is first simulated to a steady state
    if simulation_info.has_pre_equilibration == true
        preeq_sols = simulation_info.odesols_preeq
        preeq_ids = _get_preeq_ids(simulation_info, cids)
        # Arrays to store steady state (pre-eq) values
        u_ss = zeros(eltype(xdynamic), length(oprob.u0), length(preeq_ids))
        u_t0 = similar(u_ss)

        for (i, preeq_id) in pairs(preeq_ids)
            oprob_preeq = _switch_condition(
                oprob, preeq_id, xdynamic, x_ml_models, model_info, cache, ml_models_pre_ode,
                false; sensitivities = sensitivities
            )
            # Sometimes due to strongly ill-conditioned Jacobian the linear-solve runs
            # into a domain error or bounds error. This is treated as integration error.
            try
                u_ss_preeq, u_t0_preeq = (@view u_ss[:, i]), (@view u_t0[:, i])
                preeq_sols[preeq_id] = solve_pre_equilibrium!!(u_ss_preeq, u_t0_preeq,
                                                               oprob_preeq, simulation_info,
                                                               osolver, ss_solver, preeq_id,
                                                               float_tspan)
            catch e
                catch_ode_error(e)
                simulation_info.could_solve[1] = false
                return false
            end
            if preeq_sols[preeq_id].retcode != ReturnCode.Terminated
                simulation_info.could_solve[1] = false
                return false
            end
        end
    end

    for (i, cid) in pairs(simulation_info.conditionids[:experiment])
        if cids[1] != :all && !(cid in cids)
            continue
        end

        posteq_simulation = simulation_info.conditionids[:pre_equilibration][i] != :None
        simid = simulation_info.conditionids[:simulation][i]
        tsave = _get_tsave(save_observed_t, simulation_info, cid, ntimepoints_save)
        dense = _is_dense(save_observed_t, dense_sol, ntimepoints_save)
        oprob_cid = _switch_condition(
            oprob, cid, xdynamic, x_ml_models, model_info, cache, ml_models_pre_ode,
            posteq_simulation; sensitivities = sensitivities, simulation_id = simid
        )

        if posteq_simulation == true
            preeq_id = simulation_info.conditionids[:pre_equilibration][i]
            ipreq = findfirst(x -> x == preeq_id, preeq_ids)
            u_ss_preeq, u_t0_preeq = (@view u_ss[:, ipreq]), (@view u_t0[:, ipreq])
            # See comment above on domain error above for why try
            try
                odesols[cid] = solve_post_equlibrium(oprob_cid, u_ss_preeq, u_t0_preeq,
                                                     simulation_info, osolver, ss_solver,
                                                     cid, simid, tsave, dense, float_tspan)
            catch e
                catch_ode_error(e)
                simulation_info.could_solve[1] = false
                return false
            end
            retcode = odesols[cid].retcode
            if !(retcode == ReturnCode.Success || retcode == ReturnCode.Terminated)
                simulation_info.could_solve[1] = false
                return false
            end

        # ODE solution without pre-equlibrium
        else
            try
                odesols[cid] = solve_no_pre_equlibrium(oprob_cid, simulation_info, osolver,
                                                       ss_solver, cid, tsave, dense,
                                                       float_tspan)
            catch e
                catch_ode_error(e)
                simulation_info.could_solve[1] = false
                return false
            end
            retcode = odesols[cid].retcode
            if !(retcode == ReturnCode.Success || retcode == ReturnCode.Terminated)
                simulation_info.could_solve[1] = false
                return false
            end
        end
    end
    return true
end
function solve_conditions!(sols::AbstractMatrix, xdynamic::AbstractVector,
                           probinfo::PEtabODEProblemInfo, model_info::ModelInfo;
                           cids::Vector{Symbol} = [:all], ntimepoints_save::Int64 = 0,
                           save_observed_t::Bool = true, dense_sol::Bool = false,
                           track_callback::Bool = false, sensitivities::Bool = false,
                           sensitivities_AD::Bool = false)::Nothing
    cache = probinfo.cache
    xdynamic_mech, x_ml_models = split_xdynamic(xdynamic, model_info.xindices, cache)
    derivative = sensitivities_AD || sensitivities
    sucess = solve_conditions!(model_info, xdynamic_mech, x_ml_models, probinfo; cids = cids,
                               ntimepoints_save = ntimepoints_save, dense_sol = dense_sol,
                               save_observed_t = save_observed_t, derivative = derivative,
                               sensitivities = sensitivities, track_callback = track_callback)
    if sucess != true
        fill!(sols, 0.0)
        return nothing
    end

    # sols stores the ODE solutions across all conditions and is needed by ForwardDiff
    # to be in this format
    simulation_info = model_info.simulation_info
    if derivative == true
        odesols = simulation_info.odesols_derivatives
    else
        odesols = simulation_info.odesols
    end
    for cid in simulation_info.conditionids[:experiment]
        odesol_indices = simulation_info.smatrixindices[cid]
        if cids[1] == :all || cid in cids
            @views sols[:, odesol_indices] .= Array(odesols[cid])
        end
    end
    return nothing
end

function solve_post_equlibrium(@nospecialize(oprob::ODEProblem), u_ss::T, u_t0::T,  simulation_info::SimulationInfo, osolver::ODESolver, ss_solver::SteadyStateSolver, cid::Symbol, simid::Symbol, tsave::Vector{Float64}, dense::Bool, float_tspan::Bool)::ODESolution where {T <: AbstractVector}
    @unpack abstol, reltol, maxiters, solver, force_dtmin, verbose = osolver
    @unpack tstarts, tmaxs = simulation_info

    # Must be done first, as any call to remake resets initial values for sensitivities
    # to 0 when working with the ForwardSensitivity ODEProblem from SciMLSensitivity
    _oprob = _get_tspan(oprob, tstarts[cid], tmaxs[cid], solver, float_tspan)

    # Sometimes the PEtab condition-file changes the initial values for a state
    # whose value was changed in the preequilibration-simulation. The condition
    # value is prioritized by only changing u0 to the steady state value for those states
    # that were not affected by change to shift_expid.
    has_not_changed = (_oprob.u0 .== u_t0)
    @views _oprob.u0[has_not_changed] .= u_ss[has_not_changed]

    # According to the PEtab standard we can sometimes have that initial assignment is
    # overridden for pre-eq simulation, but we do not want to override for main simulation.
    # This results in NaNs in _oprob.u0
    @views _oprob.u0[isnan.(_oprob.u0)] .= u_ss[isnan.(_oprob.u0)]

    # Following the PEtab standard (point above) some initial states can be NaN, however,
    # via the change condition function this ends up in some parameters in oprob.p being
    # potentially NaN. This is not allowed, as it causes a StackOverFlow in OrdinaryDiffEq
    # (only took 1h to figure out...)
    @views _oprob.p[isnan.(_oprob.p)] .= 0.0

    # This is very subtle. SBML events can have initialValue="false" in the trigger, meaning
    # the event can trigger at t0. In PEtab v2 post_equilibration is not treated as a new
    # simulation, meaning that initialValue="false" should be ignored. This must be set via
    # a flag in the initialization function of the event
    cbs = _get_cbs(_oprob, simulation_info, simid, simulation_info.sensealg)
    _set_check_trigger_init!(cbs, false)

    if !isinf(_oprob.tspan[2])
        sol = _solve(_oprob, solver, tsave, abstol, reltol, dense, maxiters, force_dtmin, verbose, cbs)
    else
        sol = _solve_ss(_oprob, simulation_info, osolver, ss_solver, cid, simid, tsave, float_tspan)
    end

    # Need to reset event initialValue for subsequent simulations.
    _set_check_trigger_init!(cbs, true)
    return sol
end

# Without @nospecialize stackoverflow in type inference due to large ODE-system
function solve_no_pre_equlibrium(@nospecialize(oprob::ODEProblem),
                                 @nospecialize(simulation_info::SimulationInfo),
                                 osolver::ODESolver, ss_solver::SteadyStateSolver,
                                 cid::Symbol, tsave::Vector{Float64}, dense::Bool,
                                 float_tspan::Bool)::ODESolution
    @unpack abstol, reltol, maxiters, solver, force_dtmin, verbose = osolver
    @unpack tstarts, tmaxs = simulation_info

    _oprob = _get_tspan(oprob, tstarts[cid], tmaxs[cid], solver, float_tspan)

    if !isinf(_oprob.tspan[2])
        cbs = _get_cbs(_oprob, simulation_info, cid, simulation_info.sensealg)
        sol = _solve(_oprob, solver, tsave, abstol, reltol, dense, maxiters, force_dtmin, verbose, cbs)
    else
        sol = _solve_ss(_oprob, simulation_info, osolver, ss_solver, cid, cid, tsave, float_tspan)
    end
    return sol
end

function _solve(oprob::ODEProblem, solver::SciMLAlgorithm, tsave::Vector{Float64},
                abstol::Float64, reltol::Float64, dense::Bool, maxiters::Int64,
                force_dtmin::Bool, verbose::Bool, cbs::SciMLBase.DECallback)::ODESolution
    return solve(oprob, solver, abstol = abstol, reltol = reltol, verbose = verbose,
                 force_dtmin = force_dtmin, maxiters = maxiters, saveat = tsave,
                 dense = dense, callback = cbs)
end

function catch_ode_error(e)::Nothing
    if e isa DomainError # BoundsError
        @warn "Bounds error ODE solve"
    elseif e isa DomainError
        @warn "Domain error on ODE solve"
    elseif e isa SingularException
        @warn "Singular exception on ODE solve"
    else
        rethrow(e)
    end
    return nothing
end
