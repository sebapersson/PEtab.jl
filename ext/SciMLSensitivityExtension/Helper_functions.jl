function PEtab.setUpGradient(whichMethod::Symbol,
                             odeProblem::ODEProblem,
                             odeSolverOptions::ODESolverOptions,
                             ssSolverOptions::SteadyStateSolverOptions,
                             petabODECache::PEtab.PEtabODEProblemCache,
                             petabODESolverCache::PEtab.PEtabODESolverCache,
                             petabModel::PEtabModel,
                             simulationInfo::PEtab.SimulationInfo,
                             θ_indices::PEtab.ParameterIndices,
                             measurementInfo::PEtab.MeasurementsInfo,
                             parameterInfo::PEtab.ParametersInfo,
                             sensealg::Union{InterpolatingAdjoint, QuadratureAdjoint},
                             priorInfo::PEtab.PriorInfo;
                             chunkSize::Union{Nothing, Int64}=nothing,
                             sensealgSS=nothing,
                             numberOfprocesses::Int64=1,
                             jobs=nothing,
                             results=nothing,
                             splitOverConditions::Bool=false)

    _sensealgSS = isnothing(sensealgSS) ? InterpolatingAdjoint(autojacvec=ReverseDiffVJP()) : sensealgSS
    # Fast but numerically unstable method
    if simulationInfo.haspreEquilibrationConditionId == true && typeof(_sensealgSS) <: SteadyStateAdjoint
        @warn "If using adjoint sensitivity analysis for a model with PreEq-criteria the most the most efficient sensealgSS is as provided SteadyStateAdjoint. However, SteadyStateAdjoint fails if the Jacobian is singular hence we recomend you check that the Jacobian is non-singular."
    end

    iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = PEtab.getIndicesParametersNotInODESystem(θ_indices)
    computeCostNotODESystemθ = (x) -> PEtab.computeCostNotSolveODE(x[iθ_sd], x[iθ_observable], x[iθ_nonDynamic],
        petabModel, simulationInfo, θ_indices, measurementInfo, parameterInfo, petabODECache, expIDSolve=[:all],
        computeGradientNotSolveAdjoint=true)

    _computeGradient! = (gradient, θ_est) -> computeGradientAdjointEquations!(gradient,
                                                                              θ_est,
                                                                              odeSolverOptions,
                                                                              ssSolverOptions,
                                                                              computeCostNotODESystemθ,
                                                                              sensealg,
                                                                              _sensealgSS,
                                                                              odeProblem,
                                                                              petabModel,
                                                                              simulationInfo,
                                                                              θ_indices,
                                                                              measurementInfo,
                                                                              parameterInfo,
                                                                              priorInfo,
                                                                              petabODECache,
                                                                              petabODESolverCache,
                                                                              expIDSolve=[:all])
    
    return _computeGradient!
end


function PEtab.setSensealg(sensealg, ::Val{:Adjoint})

    println("Gets here as it should")
    if !isnothing(sensealg)
        @assert any(typeof(sensealg) .<: [InterpolatingAdjoint, QuadratureAdjoint]) "For gradient method :Adjoint allowed sensealg args are InterpolatingAdjoint, QuadratureAdjoint not $sensealg"
        return sensealg
    end

    return InterpolatingAdjoint(autojacvec=ReverseDiffVJP())
end


function PEtab.getCallbackSet(odeProblem::ODEProblem,
                              simulationInfo::PEtab.SimulationInfo,
                              simulationConditionId::Symbol,
                              sensealg::Union{InterpolatingAdjoint, QuadratureAdjoint})::SciMLBase.DECallback

    cbSet = SciMLSensitivity.track_callbacks(simulationInfo.callbacks[simulationConditionId], odeProblem.tspan[1],
                                                 odeProblem.u0, odeProblem.p, sensealg)
    simulationInfo.trackedCallbacks[simulationConditionId] = cbSet
    return cbSet
end