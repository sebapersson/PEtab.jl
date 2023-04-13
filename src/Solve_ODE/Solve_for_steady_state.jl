"""
    getSteadyStateSolverOptions(method::Symbol;
                                howCheckSimulationReachedSteadyState::Symbol=:wrms,
                                rootfindingAlgorithm=nothing,
                                abstol=nothing, 
                                reltol=nothing, 
                                maxiters=nothing)::SteadyStateSolverOptions

Setup steady-state solver options for finding steady-state via **either** method=:Rootfinding or method=:Simulate.

For `:Rootfinding` the steady state u* is found by solving the problem du = f(u, p, t) ≈ 0 with tolerances abstol and reltol via an automatically choosen optimisation algorithm (rootfindingAlgorithm=nothing) or via any algorithm in NonlinearSolve.jl (https://docs.sciml.ai/NonlinearSolve/stable/solvers/NonlinearSystemSolvers/), e.g. rootfindingAlgorithm=NonlinearSolve.TrustRegion(). (abstol, reltol, maxiters) defaults to (1e-8, 1e-8, 1e4).

For `:Simulate` the steady state u* is found by simulating the ODE-system until du = f(u, p, t) ≈ 0. Two options are availble for `howCheckSimulationReachedSteadyState`;
- `:wrms` : Weighted root-mean square √(∑((du ./ (reltol * u .+ abstol)).^2) / length(u)) < 1
- `:Newton` : If Newton-step Δu is sufficiently small √(∑((Δu ./ (reltol * u .+ abstol)).^2) / length(u)) < 1. 
Newton often perform better but requires an invertible Jacobian. In case not fulfilled code switches automatically to wrms. (abstol, reltol) defaults to ODE solver tolerances divided by 100 and maxiters to ODE solver value.
        
`maxiters` refers to either maximum number of rootfinding steps, or maximum number of integration steps.
"""
function getSteadyStateSolverOptions(method::Symbol;
                                     howCheckSimulationReachedSteadyState::Symbol=:wrms,
                                     rootfindingAlgorithm::Union{Nothing, NonlinearSolve.AbstractNonlinearSolveAlgorithm}=nothing,
                                     abstol=nothing, 
                                     reltol=nothing, 
                                     maxiters::Union{Nothing, Int64}=nothing)::SteadyStateSolverOptions

    @assert method ∈ [:Rootfinding, :Simulate] "Method used to find steady state can either be :Rootfinding or :Simulate not $method"
    
    if method === :Simulate
        return _getSteadyStateSolverOptions(howCheckSimulationReachedSteadyState, abstol, reltol, maxiters)
    else
        return _getSteadyStateSolverOptions(rootfindingAlgorithm, abstol, reltol, maxiters)
    end
end


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

    _abstol = isnothing(abstol) ? 1e-8 : abstol
    _reltol = isnothing(reltol) ? 1e-8 : reltol
    _maxiters = isnothing(maxiters) ? Int64(1e4) : maxiters

    _rootfindingAlgorithm = isnothing(rootfindingAlgorithm) ? NonlinearSolve.TrustRegion() : rootfindingAlgorithm

    return SteadyStateSolverOptions(:Rootfinding, _rootfindingAlgorithm, :nothing, _abstol, _reltol, _maxiters, nothing, nothing)                                      
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


# Solve ODE steady state 
function solveODEPreEqulibrium!(uAtSS::AbstractVector,
                                uAtT0::AbstractVector,
                                odeProblem::ODEProblem,
                                changeExperimentalCondition!::Function,
                                preEquilibrationId::Symbol,
                                odeSolverOptions::ODESolverOptions,
                                ssSolverOptions::SteadyStateSolverOptions,
                                convertTspan::Bool)::Union{ODESolution, SciMLBase.NonlinearSolution}

    if ssSolverOptions.method === :Simulate                           
        odeSolution = simulateToSS(odeProblem, changeExperimentalCondition!, preEquilibrationId,
                                   odeSolverOptions, ssSolverOptions, convertTspan)
        if odeSolution.retcode == ReturnCode.Terminated
            uAtSS .= odeSolution.u[end]
            uAtT0 .= odeSolution.prob.u0
        end
        return odeSolution
    
    elseif ssSolverOptions.method === :Rootfinding
        
        rootSolution = rootfindToSS(odeProblem, changeExperimentalCondition!, preEquilibrationId, ssSolverOptions)
        if rootSolution.retcode == ReturnCode.Success
            uAtSS .= rootSolution.u
            uAtT0 .= rootSolution.prob.u0
        end
        return rootSolution
    end
end


function simulateToSS(odeProblem::ODEProblem,
                      changeExperimentalCondition!::Function,
                      preEquilibrationId::Symbol,
                      odeSolverOptions::ODESolverOptions,
                      ssSolverOptions::SteadyStateSolverOptions,
                      convertTspan::Bool)::ODESolution

    changeExperimentalCondition!(odeProblem.p, odeProblem.u0, preEquilibrationId)
    _odeProblem = remakeODEProblemPreSolve(odeProblem, Inf, odeSolverOptions.solver, convertTspan)          

    solver, abstol, reltol, maxiters = odeSolverOptions.solver, odeSolverOptions.abstol, odeSolverOptions.reltol, odeSolverOptions.maxiters
    callbackSS = ssSolverOptions.callbackSS

    sol = solve(_odeProblem, solver, abstol=abstol, reltol=reltol, maxiters=maxiters, dense=false, callback=callbackSS)
    return sol
end


function rootfindToSS(odeProblem::ODEProblem,
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


import Base.show
function show(io::IO, a::SteadyStateSolverOptions)

    printstyled("SteadyStateSolverOptions", color=116)
    if a.method === :Simulate
        print(" with method ")
        printstyled(":Simulate", color=116)
        print(" ODE-model until du = f(u, p, t) ≈ 0.")
        if a.howCheckSimulationReachedSteadyState === :wrms
            @printf("\nSimulation terminated if wrms fulfill;\n√(∑((du ./ (reltol * u .+ abstol)).^2) / length(u)) < 1\n")
        else
            @printf("\nSimulation terminated if Newton-step Δu fulfill;\n√(∑((Δu ./ (reltol * u .+ abstol)).^2) / length(u)) < 1\n")
        end
        if isnothing(a.abstol)        
            @printf("with (abstol, reltol) = default values.")
        else
            @printf("with (abstol, reltol) = (%.1e, %.1e)", a.abstol, a.reltol)
        end
    end

    if a.method === :Rootfinding
        print(" with method ")
        printstyled(":Rootfinding", color=116)
        print(" to solve du = f(u, p, t) ≈ 0.")
        if isnothing(a.rootfindingAlgorithm)
            @printf("\nAlgorithm : NonlinearSolve's heruistic. Options ")
        else
            algStr = string(a.rootfindingAlgorithm)
            iEnd = findfirst(x -> x == '{', algStr)
            algStr = algStr[1:iEnd-1] * "()"
            @printf("\nAlgorithm : %s. Options ", algStr)
        end
        @printf("(abstol, reltol, maxiters) = (%.1e, %.1e, %d)", a.abstol, a.reltol, a.maxiters)
    end
end
