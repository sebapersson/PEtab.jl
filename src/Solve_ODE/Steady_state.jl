function _get_steady_state_solver(check_simulation_steady_state::Symbol,
                                  abstol,
                                  reltol,
                                  maxiters)::SteadyStateSolver
    @assert check_simulation_steady_state ∈ [:Newton, :wrms] "When steady states are computed via simulations check_simulation_steady_state must be :wrms or :Newton not $check_simulation_steady_state"

    return SteadyStateSolver(:Simulate, nothing, check_simulation_steady_state, abstol,
                             reltol, maxiters, nothing, nothing)
end
function _get_steady_state_solver(rootfinding_alg::Union{Nothing,
                                                         NonlinearSolve.AbstractNonlinearSolveAlgorithm},
                                  abstol,
                                  reltol,
                                  maxiters)::SteadyStateSolver

    # Sanity check user input
    @assert typeof(rootfinding_alg)<:Union{Nothing,
                                           NonlinearSolve.AbstractNonlinearSolveAlgorithm} "When steady states are computed via rootfinding rootfinding_alg must be nothing or a NonlinearSolveAlgorithm (https://docs.sciml.ai/NonlinearSolve/stable/solvers/NonlinearSystemSolvers/)"

    _rootfinding_alg = isnothing(rootfinding_alg) ? NonlinearSolve.TrustRegion() :
                       rootfinding_alg
    return SteadyStateSolver(:Rootfinding, _rootfinding_alg, :nothing, abstol, reltol,
                             maxiters, nothing, nothing)
end
function _get_steady_state_solver(ss_solver::SteadyStateSolver,
                                  oprob::ODEProblem)::SteadyStateSolver
    @unpack abstol, reltol, maxiters = ss_solver
    if ss_solver.method === :Simulate
        if ss_solver.check_simulation_steady_state === :Newton
            jacobian = zeros(Float64, length(oprob.u0), length(oprob.u0))
            condition_ss_callback = (u, t, integrator) -> condition_terminate_ss(u, t,
                                                                                 integrator,
                                                                                 abstol,
                                                                                 reltol,
                                                                                 true,
                                                                                 oprob.f.jac,
                                                                                 jacobian)
        elseif ss_solver.check_simulation_steady_state === :wrms
            jacobian = zeros(Float64, 0, 0)
            condition_ss_callback = (u, t, integrator) -> condition_terminate_ss(u, t,
                                                                                 integrator,
                                                                                 abstol,
                                                                                 reltol,
                                                                                 false,
                                                                                 oprob.f.jac,
                                                                                 jacobian)
        end
        callback_ss = DiscreteCallback(condition_ss_callback, affect_terminate_ss!,
                                       save_positions = (false, true))
    end

    if ss_solver.method === :Rootfinding
        callback_ss = nothing
    end

    # Sanity check user input
    return SteadyStateSolver(ss_solver.method,
                             ss_solver.rootfinding_alg,
                             ss_solver.check_simulation_steady_state,
                             abstol,
                             reltol,
                             maxiters,
                             callback_ss,
                             NonlinearProblem(oprob))
end

function solve_pre_equlibrium!(u_ss::T, u_t0::T, @nospecialize(oprob::ODEProblem), osolver::ODESolver, ss_solver::SteadyStateSolver, float_tspan::Bool)::Union{ODESolution, SciMLBase.NonlinearSolution} where T <: AbstractVector
    if ss_solver.method === :Simulate
        sol = simulate_to_ss(oprob, osolver, ss_solver, float_tspan)
        if sol.retcode == ReturnCode.Terminated || sol.retcode == ReturnCode.Success
            u_ss .= sol.u[end]
            u_t0 .= sol.prob.u0
        end
        return sol
    end

    # Have to fix
    if ss_solver.method === :Rootfinding
        root_sol = rootfind_ss(oprob, change_simulation_condition!,
                               pre_equilibration_id, ss_solver)
        if root_sol.retcode == ReturnCode.Success
            u_ss .= root_sol.u
            u_t0 .= root_sol.prob.u0
        end
        return root_sol
    end
end

function simulate_to_ss(@nospecialize(oprob::ODEProblem), osolver::ODESolver,
                        ss_solver::SteadyStateSolver, float_tspan::Bool)::ODESolution
    @unpack abstol, reltol, maxiters, force_dtmin, verbose, solver = osolver
    _oprob = _get_tspan(oprob, Inf, solver, float_tspan)
    return solve(_oprob, solver, abstol = abstol, reltol = reltol, maxiters = maxiters,
                 dense = false, callback = ss_solver.callback_ss, verbose = verbose)
end

# Callback in case steady-state is found via  model simulation
function condition_terminate_ss(u, t, integrator,
                                abstol::Float64,
                                reltol::Float64,
                                check_newton::Bool,
                                compute_jacobian::Function,
                                jacobian::AbstractMatrix)::Bool
    testval = first(get_tmp_cache(integrator))
    DiffEqBase.get_du!(testval, integrator)

    check_wrms = true
    local Δu

    if check_newton == true
        compute_jacobian(jacobian, SBMLImporter._to_float.(u), SBMLImporter._to_float.(integrator.p),
                         SBMLImporter._to_float(t))
        # In case Jacobian is non-invertible default wrms
        try
            Δu = jacobian \ testval
            check_wrms = false
        catch
            @warn "Jacobian non-invertible resorts to wrms for steady state simulations (displays max 10 times)" maxlog=10
            check_wrms = true
        end
    end

    if check_wrms == false
        value_check = sqrt(sum((Δu / (reltol * integrator.u .+ abstol)) .^ 2) / length(u))
    else
        value_check = sqrt(sum((testval ./ (reltol * integrator.u .+ abstol)) .^ 2) /
                           length(u))
    end

    return value_check < 1.0
end
function affect_terminate_ss!(integrator)
    terminate!(integrator)
end

function rootfind_ss(oprob::ODEProblem,
                     change_simulation_condition!::Function,
                     pre_equilibration_id::Symbol,
                     ss_solver::SteadyStateSolver)::SciMLBase.NonlinearSolution
    nonlinear_problem = remake(ss_solver.nonlinearsolve_problem, u0 = oprob.u0[:],
                               p = oprob.p[:])
    change_simulation_condition!(nonlinear_problem.p, nonlinear_problem.u0,
                                 pre_equilibration_id)
    root_sol = solve(nonlinear_problem,
                     ss_solver.rootfinding_alg,
                     abstol = ss_solver.abstol,
                     reltol = ss_solver.reltol,
                     maxiters = ss_solver.maxiters)

    return root_sol
end
