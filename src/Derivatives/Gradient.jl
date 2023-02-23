#=
    The top-level functions for computing the gradient via i) exactly via forward-mode autodiff, ii) forward sensitivty
    eqations, iii) adjoint sensitivity analysis and iv) Zygote interface.

    Due to it slow speed Zygote does not have full support for all models, e.g, models with priors and pre-eq criteria.
=#


# Compute the gradient via forward mode automatic differentitation
function computeGradientAutoDiff!(gradient::Vector{Float64},
                                  θ_est::Vector{Float64},
                                  computeCostNotODESystemθ::Function,
                                  computeCostDynamicθ::Function,
                                  petabODECache::PEtabODEProblemCache,
                                  cfg::ForwardDiff.GradientConfig,
                                  simulationInfo::SimulationInfo,
                                  θ_indices::ParameterIndices,
                                  priorInfo::PriorInfo,
                                  expIDSolve::Vector{Symbol} = [:all])

    splitParameterVector!(θ_est, θ_indices, petabODECache)

    try
        ForwardDiff.gradient!(petabODECache.gradientDyanmicθ, computeCostDynamicθ, petabODECache.θ_dynamic, cfg)
        @views gradient[θ_indices.iθ_dynamic] .= petabODECache.gradientDyanmicθ
    catch
        gradient .= 1e8
        return
    end

    # Check if we could solve the ODE (first), and if Inf was returned (second)
    if couldSolveODEModel(simulationInfo, expIDSolve) == false
        gradient .= 0.0
        return
    end
    if all(petabODECache.gradientDyanmicθ .== 0.0)
        gradient .= 1e8
        return
    end
    
    θ_notOdeSystem = @view θ_est[θ_indices.iθ_notOdeSystem]
    ReverseDiff.gradient!(petabODECache.gradientNotODESystemθ, computeCostNotODESystemθ, θ_notOdeSystem)
    @views gradient[θ_indices.iθ_notOdeSystem] .= petabODECache.gradientNotODESystemθ

    # If we have prior contribution its gradient is computed via autodiff for all parameters
    if priorInfo.hasPriors == true
        computeGradientPrior!(gradient, θ_est, θ_indices, priorInfo)
    end
end


# Compute the gradient via forward mode automatic differentitation where the final gradient is computed via 
# n ForwardDiff-calls accross all experimental condtions. The most efficient approach for models with many 
# parameters which are unique to each experimental condition.
function computeGradientAutoDiffSplitOverConditions!(gradient::Vector{Float64},
                                                     θ_est::Vector{Float64},
                                                     computeCostNotODESystemθ::Function,
                                                     _computeCostDynamicθ::Function,
                                                     petabODECache::PEtabODEProblemCache,
                                                     simulationInfo::SimulationInfo,
                                                     θ_indices::ParameterIndices,
                                                     priorInfo::PriorInfo, 
                                                     expIDSolve=[:all])

    splitParameterVector!(θ_est, θ_indices, petabODECache)
    θ_dynamic = petabODECache.θ_dynamic
    petabODECache.gradientDyanmicθ .= 0.0

    for conditionId in simulationInfo.experimentalConditionId
        mapConditionId = θ_indices.mapsConiditionId[conditionId]
        iθ_experimentalCondition = unique(vcat(θ_indices.mapODEProblem.iθDynamic, mapConditionId.iθDynamic))
        θ_input = θ_dynamic[iθ_experimentalCondition]
        computeCostDynamicθ = (θ_arg) ->    begin
                                                    _θ_dynamic = convert.(eltype(θ_arg), θ_dynamic)
                                                    @views _θ_dynamic[iθ_experimentalCondition] .= θ_arg
                                                    return _computeCostDynamicθ(_θ_dynamic, [conditionId])
                                            end
        try
            @views petabODECache.gradientDyanmicθ[iθ_experimentalCondition] .+= ForwardDiff.gradient(computeCostDynamicθ, θ_input)::Vector{Float64}
        catch
            gradient .= 1e8
            return
        end
    end

    # Check if we could solve the ODE (first), and if Inf was returned (second)
    if couldSolveODEModel(simulationInfo, expIDSolve) == false
        gradient .= 1e8
        return
    end
    if all(petabODECache.gradientDyanmicθ .== 0.0)
        gradient .= 1e8
        return
    end
    @views gradient[θ_indices.iθ_dynamic] .= petabODECache.gradientDyanmicθ
    
    θ_notOdeSystem = @view θ_est[θ_indices.iθ_notOdeSystem]
    ReverseDiff.gradient!(petabODECache.gradientNotODESystemθ, computeCostNotODESystemθ, θ_notOdeSystem)
    @views gradient[θ_indices.iθ_notOdeSystem] .= petabODECache.gradientNotODESystemθ

    # If we have prior contribution its gradient is computed via autodiff for all parameters
    if priorInfo.hasPriors == true
        computeGradientPrior!(gradient, θ_est, θ_indices, priorInfo)
    end
end


# Compute the gradient via forward sensitivity equations
function computeGradientForwardEquations!(gradient::Vector{Float64},
                                          θ_est::Vector{Float64},
                                          computeCostNotODESystemθ::Function,
                                          petabModel::PEtabModel,
                                          odeProblem::ODEProblem,
                                          sensealg::Union{Symbol, SciMLSensitivity.AbstractForwardSensitivityAlgorithm},
                                          simulationInfo::SimulationInfo,
                                          θ_indices::ParameterIndices,
                                          measurementInfo::MeasurementsInfo,
                                          parameterInfo::ParametersInfo,
                                          changeODEProblemParameters!::Function,
                                          solveOdeModelAllConditions!::Function,
                                          priorInfo::PriorInfo, 
                                          cfg::Union{ForwardDiff.JacobianConfig, Nothing}, 
                                          petabODECache::PEtabODEProblemCache;
                                          splitOverConditions::Bool=false,
                                          expIDSolve::Vector{Symbol} = [:all])

    splitParameterVector!(θ_est, θ_indices, petabODECache)
    θ_dynamic = petabODECache.θ_dynamic 
    θ_observable = petabODECache.θ_observable
    θ_sd = petabODECache.θ_sd
    θ_nonDynamic = petabODECache.θ_nonDynamic
    
    # Calculate gradient seperately for dynamic and non dynamic parameter.
    computeGradientForwardEqDynamicθ!(petabODECache.gradientDyanmicθ, θ_dynamic, θ_sd, θ_observable, θ_nonDynamic, petabModel,
                                      sensealg, odeProblem, simulationInfo, θ_indices, measurementInfo, parameterInfo,
                                      changeODEProblemParameters!, solveOdeModelAllConditions!, cfg, petabODECache, expIDSolve=expIDSolve, 
                                      splitOverConditions=splitOverConditions)
    @views gradient[θ_indices.iθ_dynamic] .= petabODECache.gradientDyanmicθ

    # Happens when at least one forward pass fails and I set the gradient to 1e8
    if all(petabODECache.gradientDyanmicθ .== 1e8)
        gradient .= 1e8
        return
    end

    θ_notOdeSystem = @view θ_est[θ_indices.iθ_notOdeSystem]
    ReverseDiff.gradient!(petabODECache.gradientNotODESystemθ, computeCostNotODESystemθ, θ_notOdeSystem)
    @views gradient[θ_indices.iθ_notOdeSystem] .= petabODECache.gradientNotODESystemθ

    if priorInfo.hasPriors == true
        computeGradientPrior!(gradient, θ_est, θ_indices, priorInfo)
    end
end


# Compute gradient via adjoint sensitivity analysis
function computeGradientAdjointEquations!(gradient::Vector{Float64},
                                          θ_est::Vector{Float64},
                                          adjointODESolver::SciMLAlgorithm,
                                          computeCostNotODESystemθ::Function,
                                          sensealg::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm,
                                          sensealgSS::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm,
                                          solverAbsTol::Float64,
                                          solverRelTol::Float64,
                                          odeProblem::ODEProblem,
                                          petabModel::PEtabModel,
                                          simulationInfo::SimulationInfo,
                                          θ_indices::ParameterIndices,
                                          measurementInfo::MeasurementsInfo,
                                          parameterInfo::ParametersInfo,
                                          changeODEProblemParameters!::Function,
                                          solveOdeModelAllConditions!::Function,
                                          priorInfo::PriorInfo, 
                                          petabODECache::PEtabODEProblemCache;
                                          expIDSolve::Vector{Symbol} = [:all])

    splitParameterVector!(θ_est, θ_indices, petabODECache)
    θ_dynamic = petabODECache.θ_dynamic 
    θ_observable = petabODECache.θ_observable
    θ_sd = petabODECache.θ_sd
    θ_nonDynamic = petabODECache.θ_nonDynamic

    # Calculate gradient seperately for dynamic and non dynamic parameter.
    computeGradientAdjointDynamicθ(petabODECache.gradientDyanmicθ, θ_dynamic, θ_sd, θ_observable, θ_nonDynamic, odeProblem, adjointODESolver,
                                   solverAbsTol, solverRelTol, sensealg, petabModel, simulationInfo, θ_indices, measurementInfo, parameterInfo,
                                   changeODEProblemParameters!, solveOdeModelAllConditions!, petabODECache; expIDSolve=expIDSolve,
                                   sensealgSS=sensealgSS)
    @views gradient[θ_indices.iθ_dynamic] .= petabODECache.gradientDyanmicθ

    # Happens when at least one forward pass fails and I set the gradient to 1e8
    if all(petabODECache.gradientDyanmicθ .== 1e8)
        gradient .= 1e8
        return
    end

    θ_notOdeSystem = @view θ_est[θ_indices.iθ_notOdeSystem]
    ReverseDiff.gradient!(petabODECache.gradientNotODESystemθ, computeCostNotODESystemθ, θ_notOdeSystem)
    @views gradient[θ_indices.iθ_notOdeSystem] .= petabODECache.gradientNotODESystemθ

    if priorInfo.hasPriors == true
        computeGradientPrior!(gradient, θ_est, θ_indices, priorInfo)
    end
end


# Compute gradient using Zygote
function computeGradientZygote(gradient::Vector{Float64},
                               θ_est::Vector{Float64},
                               odeProblem::ODEProblem,
                               petabModel::PEtabModel,
                               simulationInfo::SimulationInfo,
                               θ_indices::ParameterIndices,
                               measurementInfo::MeasurementsInfo,
                               parameterInfo::ParametersInfo,
                               changeODEProblemParameters::Function,
                               solveOdeModelAllConditions::Function,
                               priorInfo::PriorInfo, 
                               petabODECache::PEtabODEProblemCache)

    # Split input into observeble and dynamic parameters
    θ_dynamic, θ_observable, θ_sd, θ_nonDynamic = splitParameterVector(θ_est, θ_indices)

    # For Zygote the code must be out-of place. Hence a special likelihood funciton is needed.
    computeGradientZygoteDynamicθ = (x) -> _computeCostZygote(x, θ_sd, θ_observable, θ_nonDynamic, odeProblem,
                                                              petabModel, simulationInfo, θ_indices, measurementInfo,
                                                              parameterInfo, changeODEProblemParameters,
                                                              solveOdeModelAllConditions)
    gradient[θ_indices.iθ_dynamic] .= Zygote.gradient(computeGradientZygoteDynamicθ, θ_dynamic)[1]

    # Compute gradient for parameters which are not in ODE-system. Important to keep in mind that Sd- and observable
    # parameters can overlap in θ_est.
    iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = getIndicesParametersNotInODESystem(θ_indices)
    computeCostNotODESystemθ = (x) -> computeCostNotSolveODE(x[iθ_sd], x[iθ_observable], x[iθ_nonDynamic],
                                                             petabModel, simulationInfo, θ_indices, measurementInfo,
                                                             parameterInfo, petabODECache)
    @views ReverseDiff.gradient!(gradient[iθ_notOdeSystem], computeCostNotODESystemθ, θ_est[iθ_notOdeSystem])
end


# Compute prior contribution to log-likelihood
function computeGradientPrior!(gradient::AbstractVector,
                               θ::AbstractVector,
                               θ_indices::ParameterIndices,
                               priorInfo::PriorInfo)

    _evalPriors = (θ_est) -> begin
                                θ_estT = transformθ(θ_est, θ_indices.θ_estNames, θ_indices)
                                return computePriors(θ_est, θ_estT, θ_indices.θ_estNames, priorInfo)
                            end
    gradient .+= ForwardDiff.gradient(_evalPriors, θ)
end
