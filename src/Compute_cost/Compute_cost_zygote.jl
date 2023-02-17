
# Compute cost (likelihood) for Zygote, which means only using out-of-place functions

function computeCostZygote(θ_est,
                           odeProblem::ODEProblem,
                           petabModel::PEtabModel,
                           simulationInfo::SimulationInfo,
                           θ_indices::ParameterIndices,
                           measurementInfo::MeasurementsInfo,
                           parameterInfo::ParametersInfo,
                           changeODEProblemParameters::Function,
                           solveOdeModelAllConditions::Function,
                           priorInfo::PriorInfo)

    θ_dynamic, θ_observable, θ_sd, θ_nonDynamic = splitParameterVector(θ_est, θ_indices)

    cost = _computeCostZygote(θ_dynamic, θ_sd, θ_observable, θ_nonDynamic, odeProblem,
                              petabModel, simulationInfo, θ_indices, measurementInfo,
                              parameterInfo, changeODEProblemParameters, solveOdeModelAllConditions)

    if priorInfo.hasPriors == true
        θ_estT = transformθZygote(θ_est, θ_indices.θ_estNames, parameterInfo)
        cost += computePriors(θ_est, θ_estT, θ_indices.θ_estNames, priorInfo)
    end

    return cost
end


# Computes the likelihood in such a in a Zygote compatible way, which mainly means that no arrays are mutated.
function _computeCostZygote(θ_dynamic,
                            θ_sd,
                            θ_observable,
                            θ_nonDynamic,
                            odeProblem::ODEProblem,
                            petabModel::PEtabModel,
                            simulationInfo::SimulationInfo,
                            θ_indices::ParameterIndices,
                            measurementInfo::MeasurementsInfo,
                            parameterInfo::ParametersInfo,
                            changeODEProblemParameters::Function,
                            solveOdeModelAllConditions::Function)::Real

    θ_dynamicT = transformθZygote(θ_dynamic, θ_indices.θ_dynamicNames, parameterInfo)
    θ_sdT = transformθZygote(θ_sd, θ_indices.θ_sdNames, parameterInfo)
    θ_observableT = transformθZygote(θ_observable, θ_indices.θ_observableNames, parameterInfo)
    θ_nonDynamicT = transformθZygote(θ_nonDynamic, θ_indices.θ_nonDynamicNames, parameterInfo)

    _p, _u0 = changeODEProblemParameters(odeProblem.p, θ_dynamicT)
    _odeProblem = remake(odeProblem, p = convert.(eltype(θ_dynamic), _p), u0 = convert.(eltype(θ_dynamic), _u0))

    # Compute yMod and sd-val by looping through all experimental conditons. At the end
    # update the likelihood
    cost = convert(eltype(θ_dynamic), 0.0)
    for experimentalConditionId in simulationInfo.experimentalConditionId

        tMax = simulationInfo.timeMax[experimentalConditionId]
        odeSolution, success = solveOdeModelAllConditions(_odeProblem, experimentalConditionId, θ_dynamicT, tMax)
        if success != true
            return Inf
        end

        cost += computeCostExpCond(odeSolution, _p, θ_sdT, θ_observableT, θ_nonDynamicT, petabModel,
                                   experimentalConditionId, θ_indices, measurementInfo, parameterInfo, simulationInfo,
                                   computeGradientθDynamicZygote=true)

        if isinf(cost)
            return cost
        end
    end

    return cost
end
