function _get_steady_state_solver(check_simulation_steady_state::Symbol,
                                      abstol,
                                      reltol,
                                      maxiters)::SteadyStateSolver

    @assert check_simulation_steady_state ∈ [:Newton, :wrms] "When steady states are computed via simulations check_simulation_steady_state must be :wrms or :Newton not $check_simulation_steady_state"

    return SteadyStateSolver(:Simulate, nothing, check_simulation_steady_state, abstol, reltol, maxiters, nothing, nothing)
end
function _get_steady_state_solver(rootfinding_alg::Union{Nothing, NonlinearSolve.AbstractNonlinearSolveAlgorithm},
                                  abstol,
                                  reltol,
                                  maxiters)::SteadyStateSolver

    # Sanity check user input
    @assert typeof(rootfinding_alg) <: Union{Nothing, NonlinearSolve.AbstractNonlinearSolveAlgorithm} "When steady states are computed via rootfinding rootfinding_alg must be nothing or a NonlinearSolveAlgorithm (https://docs.sciml.ai/NonlinearSolve/stable/solvers/NonlinearSystemSolvers/)"

    _rootfinding_alg = isnothing(rootfinding_alg) ? NonlinearSolve.TrustRegion() : rootfinding_alg
    return SteadyStateSolver(:Rootfinding, _rootfinding_alg, :nothing, abstol, reltol, maxiters, nothing, nothing)
end
function _get_steady_state_solver(ss_solver::SteadyStateSolver,
                                      odeProblem::ODEProblem,
                                      abstol,
                                      reltol,
                                      maxiters)::SteadyStateSolver

    _abstol = isnothing(ss_solver.abstol) ? abstol : ss_solver.abstol
    _reltol = isnothing(ss_solver.reltol) ? reltol : ss_solver.reltol
    _maxiters = isnothing(ss_solver.reltol) ? maxiters : ss_solver.maxiters

    if ss_solver.method === :Simulate
        if ss_solver.check_simulation_steady_state === :Newton
            jacobian = zeros(Float64, length(odeProblem.u0), length(odeProblem.u0))
            condCallback = (u, t, integrator) -> conditionTerminateSS(u, t, integrator, _abstol, _reltol, true,
                                                                      odeProblem.f.jac, jacobian)
        elseif ss_solver.check_simulation_steady_state === :wrms
            jacobian = zeros(Float64, 0, 0)
            condCallback = (u, t, integrator) -> conditionTerminateSS(u, t, integrator, _abstol, _reltol, false,
                                                                      odeProblem.f.jac, jacobian)
        end
        callback_ss = DiscreteCallback(condCallback, affectTerminateSS!, save_positions=(false, true))
    end

    if ss_solver.method === :Rootfinding
        callback_ss = nothing
    end

    # Sanity check user input
    return SteadyStateSolver(ss_solver.method,
                                    ss_solver.rootfinding_alg,
                                    ss_solver.check_simulation_steady_state,
                                    _abstol,
                                    _reltol,
                                    _maxiters,
                                    callback_ss,
                                    NonlinearProblem(odeProblem))
end


function solveODEPreEqulibrium!(uAtSS::AbstractVector,
                                uAtT0::AbstractVector,
                                odeProblem::ODEProblem,
                                changeExperimentalCondition!::Function,
                                preEquilibrationId::Symbol,
                                ode_solver::ODESolver,
                                ss_solver::SteadyStateSolver,
                                convert_tspan::Bool)::Union{ODESolution, SciMLBase.NonlinearSolution}

    if ss_solver.method === :Simulate
        odeSolution = simulateToSS(odeProblem, ode_solver.solver, changeExperimentalCondition!, preEquilibrationId,
                                   ode_solver, ss_solver, convert_tspan)
        if odeSolution.retcode == ReturnCode.Terminated || odeSolution.retcode == ReturnCode.Success
            uAtSS .= odeSolution.u[end]
            uAtT0 .= odeSolution.prob.u0
        end
        return odeSolution
    end

    if ss_solver.method === :Rootfinding

        rootSolution = rootfindSS(odeProblem, changeExperimentalCondition!, preEquilibrationId, ss_solver)
        if rootSolution.retcode == ReturnCode.Success
            uAtSS .= rootSolution.u
            uAtT0 .= rootSolution.prob.u0
        end
        return rootSolution
    end
end


function simulateToSS(odeProblem::ODEProblem,
                      solver::S,
                      changeExperimentalCondition!::Function,
                      preEquilibrationId::Symbol,
                      ode_solver::ODESolver,
                      ss_solver::SteadyStateSolver,
                      convert_tspan::Bool)::ODESolution where S<:SciMLAlgorithm

    changeExperimentalCondition!(odeProblem.p, odeProblem.u0, preEquilibrationId)
    _odeProblem = setTspanODEProblem(odeProblem, Inf, ode_solver.solver, convert_tspan)

    abstol, reltol, maxiters = ode_solver.abstol, ode_solver.reltol, ode_solver.maxiters
    callback_ss = ss_solver.callback_ss

    sol = solve(_odeProblem, solver, abstol=abstol, reltol=reltol, maxiters=maxiters, dense=false, callback=callback_ss)
    return sol
end


# Callback in case steady-state is found via  model simulation
function conditionTerminateSS(u, t, integrator,
                              abstol::Float64,
                              reltol::Float64,
                              checkNewton::Bool,
                              computeJacobian!::Function,
                              jacobian::AbstractMatrix)

    testval = first(get_tmp_cache(integrator))
    DiffEqBase.get_du!(testval, integrator)

    checkWrms = true
    local Δu

    if checkNewton == true
        computeJacobian!(jacobian, dualToFloat.(u), dualToFloat.(integrator.p), dualToFloat(t))
        # In case Jacobian is non-invertible default wrms
        try
            Δu = jacobian \ testval
            checkWrms = false
        catch
            @warn "Jacobian non-invertible resorts to wrms for steady state simulations (displays max 10 times)" maxlog=10
            checkWrms = true
        end
    end

    if checkWrms == false
        valCheck = sqrt(sum((Δu / (reltol * integrator.u .+ abstol)).^2) / length(u))
    else
        valCheck = sqrt(sum((testval ./ (reltol * integrator.u .+ abstol)).^2) / length(u))
    end

    return valCheck < 1.0
end
function affectTerminateSS!(integrator)
    terminate!(integrator)
end


function rootfindSS(odeProblem::ODEProblem,
                    changeExperimentalCondition!::Function,
                    preEquilibrationId::Symbol,
                    ss_solver::SteadyStateSolver)::SciMLBase.NonlinearSolution

    nonlinear_problem = remake(ss_solver.nonlinearsolve_problem, u0=odeProblem.u0[:], p=odeProblem.p[:])
    changeExperimentalCondition!(nonlinear_problem.p, nonlinear_problem.u0, preEquilibrationId)
    nonlinearSolution = solve(nonlinear_problem,
                              ss_solver.rootfinding_alg,
                              abstol=ss_solver.abstol,
                              reltol=ss_solver.reltol,
                              maxiters=ss_solver.maxiters)

    return nonlinearSolution
end