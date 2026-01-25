function solve_pre_equilibrium!!(
        u_ss::T, u_t0::T, @nospecialize(oprob::ODEProblem),
        @nospecialize(simulation_info::SimulationInfo), osolver::ODESolver,
        ss_solver::SteadyStateSolver, preeq_id::Symbol, float_tspan::Bool
    )::Union{ODESolution, SciMLBase.NonlinearSolution} where {T <: AbstractVector}
    if ss_solver.method === :Simulate
        sol = _solve_ss(
            oprob, simulation_info, osolver, ss_solver, preeq_id, preeq_id, Float64[],
            float_tspan
        )
        if sol.retcode == ReturnCode.Terminated || sol.retcode == ReturnCode.Success
            u_ss .= sol.u[end]
            u_t0 .= sol.prob.u0
        end
    end

    if ss_solver.method === :Rootfinding
        sol = rootfind_ss(oprob, ss_solver)
        if root_sol.retcode == ReturnCode.Success
            u_ss .= sol.u
            u_t0 .= sol.prob.u0
        end
    end
    return sol
end

function _solve_ss(
        @nospecialize(oprob::ODEProblem), @nospecialize(simulation_info::SimulationInfo),
        osolver::ODESolver, ss_solver::SteadyStateSolver, cid::Symbol, simid::Symbol,
        tsave::Vector{Float64}, float_tspan::Bool
    )::ODESolution
    @unpack abstol, reltol, maxiters, verbose, solver = osolver

    if haskey(simulation_info.tstarts, cid)
        t_start = simulation_info.tstarts[cid]
    else
        t_start = 0.0
    end
    _oprob = _get_tspan(oprob, t_start, Inf, solver, float_tspan)

    # Steady-state callback is not allowed per PEtab standard to terminate the simulation
    # until the model has been simulated for all measurement points
    if !isempty(tsave) && length(tsave) > 1
        ss_solver.tmin_simulate[1] = tsave[end - 1]
    else
        ss_solver.tmin_simulate[1] = 0.0
    end

    # PEtab events are allowed to trigger during steady-state simulations
    cbs = _get_cbs(_oprob, simulation_info, simid, simulation_info.sensealg)
    _cbs = CallbackSet(
        cbs.discrete_callbacks..., cbs.continuous_callbacks..., ss_solver.callback_ss
    )
    return solve(
        _oprob, solver, abstol = abstol, reltol = reltol, maxiters = maxiters,
        saveat = tsave, dense = false, callback = _cbs, verbose = verbose
    )
end

function rootfind_ss(
        oprob::ODEProblem, ss_solver::SteadyStateSolver
    )::SciMLBase.NonlinearSolution
    @unpack abstol, reltol, maxiters, rootfinding_alg = ss_solver
    prob = remake(ss_solver.nprob, u0 = oprob.u0[:], p = oprob.p[:])
    return solve(
        prob, rootfinding_alg, abstol = abstol, reltol = reltol, maxiters = maxiters
    )
end
