function _getSteadyStateSolverOptions(howCheckSimulationReachedSteadyState::Symbol, 
                                      abstol,
                                      reltol, 
                                      maxiters)::SteadyStateSolverOptions

    @assert howCheckSimulationReachedSteadyState ∈ [:Newton, :wrms] "When steady states are computed via simulations howCheckSimulationReachedSteadyState must be :wrms or :Newton not $howCheckSimulationReachedSteadyState"

    return SteadyStateSolverOptions(:Simulate, nothing, howCheckSimulationReachedSteadyState, abstol, reltol, maxiters, nothing, nothing)
end
function _getSteadyStateSolverOptions(rootfindingAlgorithm::Union{Nothing, NonlinearSolve.AbstractNonlinearSolveAlgorithm},
                                      abstol,
                                      reltol, 
                                      maxiters)::SteadyStateSolverOptions

    # Sanity check user input 
    @assert typeof(rootfindingAlgorithm) <: Union{Nothing, NonlinearSolve.AbstractNonlinearSolveAlgorithm} "When steady states are computed via rootfinding rootfindingAlgorithm must be nothing or a NonlinearSolveAlgorithm (https://docs.sciml.ai/NonlinearSolve/stable/solvers/NonlinearSystemSolvers/)"
    
    _rootfindingAlgorithm = isnothing(rootfindingAlgorithm) ? NonlinearSolve.TrustRegion() : rootfindingAlgorithm
    return SteadyStateSolverOptions(:Rootfinding, _rootfindingAlgorithm, :nothing, abstol, reltol, maxiters, nothing, nothing)                                      
end
function _getSteadyStateSolverOptions(ssSolverOptions::SteadyStateSolverOptions,
                                      odeProblem::ODEProblem, 
                                      abstol, 
                                      reltol, 
                                      maxiters)::SteadyStateSolverOptions

    _abstol = isnothing(ssSolverOptions.abstol) ? abstol : ssSolverOptions.abstol
    _reltol = isnothing(ssSolverOptions.reltol) ? reltol : ssSolverOptions.reltol
    _maxiters = isnothing(ssSolverOptions.reltol) ? maxiters : ssSolverOptions.maxiters

    if ssSolverOptions.method === :Simulate 
        if ssSolverOptions.howCheckSimulationReachedSteadyState === :Newton
            jacobian = zeros(Float64, length(odeProblem.u0), length(odeProblem.u0))
            condCallback = (u, t, integrator) -> conditionTerminateSS(u, t, integrator, _abstol, _reltol, true, 
                                                                      odeProblem.f.jac, jacobian)
        elseif ssSolverOptions.howCheckSimulationReachedSteadyState === :wrms
            jacobian = zeros(Float64, 0, 0)
            condCallback = (u, t, integrator) -> conditionTerminateSS(u, t, integrator, _abstol, _reltol, false, 
                                                                      odeProblem.f.jac, jacobian)
        end
        callbackSS = DiscreteCallback(condCallback, affectTerminateSS!, save_positions=(false, true))
    end

    if ssSolverOptions.method === :Rootfinding 
        callbackSS = nothing
    end

    # Sanity check user input 
    return SteadyStateSolverOptions(ssSolverOptions.method, 
                                    ssSolverOptions.rootfindingAlgorithm, 
                                    ssSolverOptions.howCheckSimulationReachedSteadyState, 
                                    _abstol, 
                                    _reltol, 
                                    _maxiters,
                                    callbackSS,
                                    NonlinearProblem(odeProblem))                                      
end


function solveODEPreEqulibrium!(uAtSS::AbstractVector,
                                uAtT0::AbstractVector,
                                odeProblem::ODEProblem,
                                changeExperimentalCondition!::Function,
                                preEquilibrationId::Symbol,
                                odeSolverOptions::ODESolverOptions,
                                ssSolverOptions::SteadyStateSolverOptions,
                                convertTspan::Bool)::Union{ODESolution, SciMLBase.NonlinearSolution}

    if ssSolverOptions.method === :Simulate                           
        odeSolution = simulateToSS(odeProblem, odeSolverOptions.solver, changeExperimentalCondition!, preEquilibrationId,
                                   odeSolverOptions, ssSolverOptions, convertTspan)
        if odeSolution.retcode == ReturnCode.Terminated || odeSolution.retcode == ReturnCode.Success
            uAtSS .= odeSolution.u[end]
            uAtT0 .= odeSolution.prob.u0
        end
        return odeSolution
    end
    
    if ssSolverOptions.method === :Rootfinding
        
        rootSolution = rootfindSS(odeProblem, changeExperimentalCondition!, preEquilibrationId, ssSolverOptions)
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
                      odeSolverOptions::ODESolverOptions,
                      ssSolverOptions::SteadyStateSolverOptions,
                      convertTspan::Bool)::ODESolution where S<:SciMLAlgorithm

    changeExperimentalCondition!(odeProblem.p, odeProblem.u0, preEquilibrationId)
    _odeProblem = setTspanODEProblem(odeProblem, Inf, odeSolverOptions.solver, convertTspan)          

    abstol, reltol, maxiters = odeSolverOptions.abstol, odeSolverOptions.reltol, odeSolverOptions.maxiters
    callbackSS = ssSolverOptions.callbackSS

    sol = solve(_odeProblem, solver, abstol=abstol, reltol=reltol, maxiters=maxiters, dense=false, callback=callbackSS)
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
                    ssSolverOptions::SteadyStateSolverOptions)::SciMLBase.NonlinearSolution

    nonlinearProblem = remake(ssSolverOptions.nonlinearSolveProblem, u0=odeProblem.u0[:], p=odeProblem.p[:])
    changeExperimentalCondition!(nonlinearProblem.p, nonlinearProblem.u0, preEquilibrationId)
    nonlinearSolution = solve(nonlinearProblem, 
                              ssSolverOptions.rootfindingAlgorithm, 
                              abstol=ssSolverOptions.abstol, 
                              reltol=ssSolverOptions.reltol, 
                              maxiters=ssSolverOptions.maxiters)           

    return nonlinearSolution
end