function solve_pre_equlibrium!(u_ss::T, u_t0::T, @nospecialize(oprob::ODEProblem), osolver::ODESolver, ss_solver::SteadyStateSolver, float_tspan::Bool)::Union{ODESolution, SciMLBase.NonlinearSolution} where T <: AbstractVector
    if ss_solver.method === :Simulate
        sol = simulate_to_ss(oprob, osolver, ss_solver, float_tspan)
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

function simulate_to_ss(@nospecialize(oprob::ODEProblem), osolver::ODESolver,
                        ss_solver::SteadyStateSolver, float_tspan::Bool)::ODESolution
    @unpack abstol, reltol, maxiters, force_dtmin, verbose, solver = osolver
    _oprob = _get_tspan(oprob, Inf, solver, float_tspan)
    return solve(_oprob, solver, abstol = abstol, reltol = reltol, maxiters = maxiters,
                 dense = false, callback = ss_solver.callback_ss, verbose = verbose)
end

function rootfind_ss(oprob::ODEProblem, ss_solver::SteadyStateSolver)::SciMLBase.NonlinearSolution
    @unpack abstol, reltol, maxiters, rootfinding_alg = ss_solver
    prob = remake(ss_solver.nprob, u0 = oprob.u0[:],
                  p = oprob.p[:])
    return solve(prob, rootfinding_alg, abstol = abstol, reltol = reltol,
                 maxiters = maxiters)
end
