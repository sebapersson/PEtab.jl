function solve_conditions!(model_info::ModelInfo, xdynamic::AbstractVector, xnn::Dict{Symbol, ComponentArray}, probinfo::PEtabODEProblemInfo; cids::Vector{Symbol} = [:all], ntimepoints_save::Int64 = 0, save_observed_t::Bool = true, dense_sol::Bool = false, track_callback::Bool = false, sensitivites::Bool = false, derivative::Bool = false)::Bool
    @unpack simulation_info, model, xindices = model_info
    @unpack float_tspan = model
    @unpack cache, f_nns_preode = probinfo
    if derivative == true || sensitivites == true || track_callback == true
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
    _set_cond_const_parameters!(oprob.p, xdynamic, xindices)

    # In case the model is first simulated to a steady state
    if simulation_info.has_pre_equilibration == true
        preeq_sols = simulation_info.odesols_preeq
        preeq_ids = _get_preeq_ids(simulation_info, cids)
        # Arrays to store steady state (pre-eq) values
        u_ss = zeros(eltype(xdynamic), length(oprob.u0), length(preeq_ids))
        u_t0 = similar(u_ss)

        for (i, preeq_id) in pairs(preeq_ids)
            oprob_preeq = _switch_condition(oprob, preeq_id, xdynamic, xnn, model_info,
                                            cache, f_nns_preode; sensitivites = sensitivites)
            # Sometimes due to strongly ill-conditioned Jacobian the linear-solve runs
            # into a domain error or bounds error. This is treated as integration error.
            try
                u_ss_preeq, u_t0_preeq = (@view u_ss[:, i]), (@view u_t0[:, i])
                preeq_sols[preeq_id] = solve_pre_equlibrium!(u_ss_preeq, u_t0_preeq,
                                                             oprob_preeq, osolver,
                                                             ss_solver,
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

        simid = simulation_info.conditionids[:simulation][i]
        tsave = _get_tsave(save_observed_t, simulation_info, cid, ntimepoints_save)
        dense = _is_dense(save_observed_t, dense_sol, ntimepoints_save)
        oprob_cid = _switch_condition(oprob, cid, xdynamic, xnn, model_info, cache,
                                      f_nns_preode; sensitivites = sensitivites,
                                      simid = simid)

        if simulation_info.conditionids[:pre_equilibration][i] != :None
            preeq_id = simulation_info.conditionids[:pre_equilibration][i]
            ipreq = findfirst(x -> x == preeq_id, preeq_ids)
            u_ss_preeq, u_t0_preeq = (@view u_ss[:, ipreq]), (@view u_t0[:, ipreq])
            # See comment above on domain error above for why try
            try
                odesols[cid] = solve_post_equlibrium(oprob_cid, u_ss_preeq, u_t0_preeq,
                                                     osolver, simulation_info, cid,
                                                     tsave, dense, float_tspan)
            catch e
                catch_ode_error(e)
                simulation_info.could_solve[1] = false
                return false
            end
            if odesols[cid].retcode != ReturnCode.Success
                simulation_info.could_solve[1] = false
                return false
            end

        # ODE solution without pre-equlibrium
        else
            try
                odesols[cid] = solve_no_pre_equlibrium(oprob_cid, osolver, simulation_info,
                                                       cid, tsave, dense, float_tspan)
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
function solve_conditions!(sols::AbstractMatrix, xdynamic_tot::AbstractVector,
                           probinfo::PEtabODEProblemInfo, model_info::ModelInfo;
                           cids::Vector{Symbol} = [:all], ntimepoints_save::Int64 = 0,
                           save_observed_t::Bool = true, dense_sol::Bool = false,
                           track_callback::Bool = false, sensitivites::Bool = false,
                           sensitivites_AD::Bool = false)::Nothing
    cache = probinfo.cache
    xdynamic_mech, xnn = split_xdynamic(xdynamic_tot, model_info.xindices, cache)
    if sensitivites_AD == true && cache.nxdynamic[1] != length(xdynamic_tot)
        _xdynamic_mech = xdynamic_mech[cache.xdynamic_output_order]
    else
        _xdynamic_mech = xdynamic_mech
    end
    derivative = sensitivites_AD || sensitivites
    sucess = solve_conditions!(model_info, _xdynamic_mech, xnn, probinfo; cids = cids,
                               ntimepoints_save = ntimepoints_save, dense_sol = dense_sol,
                               save_observed_t = save_observed_t, derivative = derivative,
                               sensitivites = sensitivites, track_callback = track_callback)
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
    istart, iend = 1, 0
    for cid in simulation_info.conditionids[:experiment]
        iend += length(simulation_info.tsaves[cid])
        if cids[1] == :all || cid in cids
            @views sols[:, istart:iend] .= Array(odesols[cid])
        end
        istart = iend + 1
    end
    return nothing
end

# TODO: Should not need simulation_info (as callback should have everything), fix after
# mtkv9 update
function solve_post_equlibrium(@nospecialize(oprob::ODEProblem), u_ss::T, u_t0::T,
                               osolver::ODESolver, simulation_info::SimulationInfo,
                               cid::Symbol, tsave::Vector{Float64}, dense::Bool,
                               float_tspan::Bool)::ODESolution where {T <: AbstractVector}
    @unpack abstol, reltol, maxiters, solver, force_dtmin, verbose = osolver

    # Must be done first, as any call to remake resets initial values for sensitivites
    # to 0 when working with the ForwardSensitivity ODEProblem from SciMLSensitivity
    _oprob = _get_tspan(oprob, simulation_info.tmaxs[cid], solver, float_tspan)

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

    # If case of adjoint sensitivity analysis we need to track the callback to get correct
    # gradients, hence sensealg
    cbs = _get_cbs(_oprob, simulation_info, cid, simulation_info.sensealg)
    sol = _solve(_oprob, solver, tsave, abstol, reltol, abstol, reltol, dense, maxiters,
                 force_dtmin, verbose, cbs)
    return sol
end

# Without @nospecialize stackoverflow in type inference due to large ODE-system
function solve_no_pre_equlibrium(@nospecialize(oprob::ODEProblem), osolver::ODESolver,
                                 simulation_info::SimulationInfo, cid::Symbol,
                                 tsave::Vector{Float64}, dense::Bool,
                                 float_tspan::Bool)::ODESolution
    @unpack abstol, reltol, maxiters, solver, force_dtmin, verbose = osolver

    _oprob = _get_tspan(oprob, simulation_info.tmaxs[cid], solver, float_tspan)
    cbs = _get_cbs(_oprob, simulation_info, cid, simulation_info.sensealg)
    return _solve(_oprob, solver, tsave, abstol, reltol, abstol, reltol, dense, maxiters,
                  force_dtmin, verbose, cbs)
end

function _solve(oprob::ODEProblem, solver::SciMLAlgorithm, tsave::Vector{Float64},
                abstol::Float64, reltol::Float64, abstol_ss::Float64, reltol_ss::Float64,
                dense::Bool, maxiters::Int64, force_dtmin::Bool,
                verbose::Bool, cbs::SciMLBase.DECallback)::ODESolution
    # If t_max = inf the model is simulated to steady state using the TerminateSteadyState callback.
    if isinf(oprob.tspan[2]) || oprob.tspan[2] == 1e8
        return solve(oprob, solver, abstol = abstol, reltol = reltol, verbose = verbose,
                     force_dtmin = force_dtmin, maxiters = maxiters, save_on = false,
                     save_start = false, save_end = true, dense = dense,
                     callback = TerminateSteadyState(abstol_ss, reltol_ss))
    else
        return solve(oprob, solver, abstol = abstol, reltol = reltol, verbose = verbose,
                     force_dtmin = force_dtmin, maxiters = maxiters, saveat = tsave,
                     dense = dense, callback = cbs)
    end
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
