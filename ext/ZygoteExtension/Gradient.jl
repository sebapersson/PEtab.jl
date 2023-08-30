# Compute gradient using Zygote
function computeGradientZygote(gradient::Vector{Float64},
                               θ_est::Vector{Float64},
                               odeProblem::ODEProblem,
                               petabModel::PEtabModel,
                               simulationInfo::PEtab.SimulationInfo,
                               θ_indices::PEtab.ParameterIndices,
                               measurementInfo::PEtab.MeasurementsInfo,
                               parameterInfo::PEtab.ParametersInfo,
                               solveOdeModelAllConditions::Function,
                               priorInfo::PEtab.PriorInfo,
                               petabODECache::PEtab.PEtabODEProblemCache)

    # Split input into observeble and dynamic parameters
    θ_dynamic, θ_observable, θ_sd, θ_nonDynamic = PEtab.splitParameterVector(θ_est, θ_indices)

    # For Zygote the code must be out-of place. Hence a special likelihood funciton is needed.
    computeGradientZygoteDynamicθ = (x) -> _computeCostZygote(x, θ_sd, θ_observable, θ_nonDynamic, odeProblem,
                                                              petabModel, simulationInfo, θ_indices, measurementInfo,
                                                              parameterInfo,solveOdeModelAllConditions)
    gradient[θ_indices.iθ_dynamic] .= Zygote.gradient(computeGradientZygoteDynamicθ, θ_dynamic)[1]

    # Compute gradient for parameters which are not in ODE-system. Important to keep in mind that Sd- and observable
    # parameters can overlap in θ_est.
    iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = PEtab.getIndicesParametersNotInODESystem(θ_indices)
    computeCostNotODESystemθ = (x) -> PEtab.computeCostNotSolveODE(x[iθ_sd], x[iθ_observable], x[iθ_nonDynamic],
                                                                   petabModel, simulationInfo, θ_indices, measurementInfo,
                                                                   parameterInfo, petabODECache)
    @views ReverseDiff.gradient!(gradient[iθ_notOdeSystem], computeCostNotODESystemθ, θ_est[iθ_notOdeSystem])
end
