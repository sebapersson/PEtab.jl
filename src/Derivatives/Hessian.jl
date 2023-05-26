#=
    The top-level functions for computing the hessian via i) exactly via autodiff, ii) block-approximation via
    auto-diff and iii) guass-newton approximation.
=#


function computeHessian!(hessian::Matrix{Float64},
                         θ_est::Vector{Float64},
                         _evalHessian::Function,
                         cfg::ForwardDiff.HessianConfig,
                         simulationInfo::SimulationInfo,
                         θ_indices::ParameterIndices, 
                         priorInfo::PriorInfo)


    # Only try to compute hessian if we could compute the cost
    if all([simulationInfo.odeSolutions[id].retcode == ReturnCode.Success || simulationInfo.odeSolutions[id].retcode == ReturnCode.Terminated for id in simulationInfo.experimentalConditionId])
        try
            ForwardDiff.hessian!(hessian, _evalHessian, θ_est, cfg)
            @views hessian .= Symmetric(hessian)
        catch
            hessian .= 0.0
        end
    else
        hessian .= 0.0
    end

    if priorInfo.hasPriors == true
        computeHessianPrior!(hessian, θ_est, θ_indices, priorInfo)
    end
end


# Compute the hessian via forward mode automatic differentitation where the final hessian is computed via 
# n ForwardDiff-calls accross all experimental condtions. The most efficient approach for models with many 
# parameters which are unique to each experimental condition.
function computeHessianSplitOverConditions!(hessian::Matrix{Float64},
                                            θ_est::Vector{Float64},
                                            _evalHessian::Function,
                                            simulationInfo::SimulationInfo,
                                            θ_indices::ParameterIndices, 
                                            priorInfo::PriorInfo, 
                                            expIDSolve::Vector{Symbol} = [:all])

    hessian .= 0.0
    for conditionId in simulationInfo.experimentalConditionId
        mapConditionId = θ_indices.mapsConiditionId[conditionId]
        iθ_experimentalCondition = unique(vcat(θ_indices.mapODEProblem.iθDynamic, mapConditionId.iθDynamic, θ_indices.iθ_notOdeSystem))
        θ_input = θ_est[iθ_experimentalCondition]
        hTmp = zeros(length(θ_input), length(θ_input))
        evalHessian = (θ_arg) -> begin
                                    _θ_est = convert.(eltype(θ_arg), θ_est)
                                    _θ_est[iθ_experimentalCondition] .= θ_arg
                                    return _evalHessian(_θ_est, [conditionId])
                                 end
        ForwardDiff.hessian!(hTmp, evalHessian, θ_input)
        try
            ForwardDiff.hessian!(hTmp, evalHessian, θ_input)
        catch
            hessian .= 0.0
            return
        end
        @inbounds for i in eachindex(iθ_experimentalCondition)
            @inbounds for j in eachindex(iθ_experimentalCondition)
                hessian[iθ_experimentalCondition[i], iθ_experimentalCondition[j]] += hTmp[i, j]
            end
        end
    end

    if priorInfo.hasPriors == true
        computeHessianPrior!(hessian, θ_est, θ_indices, priorInfo)
    end
end


function computeHessianBlockApproximation!(hessian::Matrix{Float64},
                                           θ_est::Vector{Float64},
                                           computeCostNotODESystemθ::Function,
                                           computeCostDynamicθ::Function,
                                           petabODECache::PEtabODEProblemCache,
                                           cfg::ForwardDiff.HessianConfig,
                                           simulationInfo::SimulationInfo,
                                           θ_indices::ParameterIndices,
                                           priorInfo::PriorInfo;
                                           expIDSolve::Vector{Symbol} = [:all])
                                  
    # Avoid incorrect non-zero values
    hessian .= 0.0

    splitParameterVector!(θ_est, θ_indices, petabODECache)
    θ_dynamic = petabODECache.θ_dynamic 

    try
        if !isempty(θ_indices.iθ_dynamic)
            @views ForwardDiff.hessian!(hessian[θ_indices.iθ_dynamic, θ_indices.iθ_dynamic], computeCostDynamicθ, θ_dynamic, cfg)
        else
            computeCostDynamicθ(θ_dynamic)
        end
    catch
        hessian .= 0.0
        return
    end

    # Check if we could solve the ODE (first), and if Inf was returned (second)
    if couldSolveODEModel(simulationInfo, expIDSolve) == false
        hessian .= 0.0
        return
    end
    if !isempty(θ_dynamic) && all((@view hessian[θ_indices.iθ_dynamic, θ_indices.iθ_dynamic]) .== 0.0)
        return
    end

    iθ_notOdeSystem = θ_indices.iθ_notOdeSystem
    @views ForwardDiff.hessian!(hessian[iθ_notOdeSystem, iθ_notOdeSystem], computeCostNotODESystemθ, θ_est[iθ_notOdeSystem])

    # Even though this is a hessian approximation, due to ease of implementation and low run-time we compute the
    # full hessian for the priors
    if priorInfo.hasPriors == true
        computeHessianPrior!(hessian, θ_est, θ_indices, priorInfo)
    end
end

function computeHessianBlockApproximationSplitOverConditions!(hessian::Matrix{Float64},
                                                              θ_est::Vector{Float64},
                                                              computeCostNotODESystemθ::Function,
                                                              _computeCostDynamicθ::Function,
                                                              petabODECache::PEtabODEProblemCache,
                                                              simulationInfo::SimulationInfo,
                                                              θ_indices::ParameterIndices,
                                                              priorInfo::PriorInfo;
                                                              expIDSolve::Vector{Symbol} = [:all])

    # Avoid incorrect non-zero values
    hessian .= 0.0

    splitParameterVector!(θ_est, θ_indices, petabODECache)
    θ_dynamic = petabODECache.θ_dynamic

    for conditionId in simulationInfo.experimentalConditionId
        mapConditionId = θ_indices.mapsConiditionId[conditionId]
        iθ_experimentalCondition = unique(vcat(θ_indices.mapODEProblem.iθDynamic, mapConditionId.iθDynamic))
        θ_input = θ_dynamic[iθ_experimentalCondition]
        hTmp = zeros(length(θ_input), length(θ_input))
        computeCostDynamicθ = (θ_arg) ->    begin
                                                    _θ_dynamic = convert.(eltype(θ_arg), θ_dynamic)
                                                    @views _θ_dynamic[iθ_experimentalCondition] .= θ_arg
                                                    return _computeCostDynamicθ(_θ_dynamic, [conditionId])
                                            end
        try
            ForwardDiff.hessian!(hTmp, computeCostDynamicθ, θ_input)
        catch
            hessian .= 0.0
            return
        end
        @inbounds for i in eachindex(iθ_experimentalCondition)
            @inbounds for j in eachindex(iθ_experimentalCondition)
                hessian[iθ_experimentalCondition[i], iθ_experimentalCondition[j]] += hTmp[i, j]
            end
        end
    end

    # Check if we could solve the ODE (first), and if Inf was returned (second)
    if couldSolveODEModel(simulationInfo, expIDSolve) == false
        hessian .= 0.0
        return
    end
    if all((@view hessian[θ_indices.iθ_dynamic, θ_indices.iθ_dynamic]) .== 0.0)
        return
    end

    iθ_notOdeSystem = θ_indices.iθ_notOdeSystem
    @views ForwardDiff.hessian!(hessian[iθ_notOdeSystem, iθ_notOdeSystem], computeCostNotODESystemθ, θ_est[iθ_notOdeSystem])

    # Even though this is a hessian approximation, due to ease of implementation and low run-time we compute the
    # full hessian for the priors
    if priorInfo.hasPriors == true
        computeHessianPrior!(hessian, θ_est, θ_indices, priorInfo)
    end
end


function computeGaussNewtonHessianApproximation!(out::Matrix{Float64},
                                                 θ_est::Vector{Float64},
                                                 odeProblem::ODEProblem,
                                                 computeResidualsNotSolveODE!::Function,
                                                 petabModel::PEtabModel,
                                                 simulationInfo::SimulationInfo,
                                                 θ_indices::ParameterIndices,
                                                 measurementInfo::MeasurementsInfo,
                                                 parameterInfo::ParametersInfo,
                                                 solveOdeModelAllConditions!::Function,
                                                 priorInfo::PriorInfo, 
                                                 cfg::ForwardDiff.JacobianConfig, 
                                                 cfgNotSolveODE::ForwardDiff.JacobianConfig,
                                                 petabODECache::PEtabODEProblemCache;
                                                 reuseS::Bool=false,
                                                 splitOverConditions::Bool=false,
                                                 returnJacobian::Bool=false,
                                                 expIDSolve::Vector{Symbol} = [:all], 
                                                 isRemade::Bool=false)

    # Avoid incorrect non-zero values
    out .= 0.0

    splitParameterVector!(θ_est, θ_indices, petabODECache)
    θ_dynamic = petabODECache.θ_dynamic 
    θ_observable = petabODECache.θ_observable
    θ_sd = petabODECache.θ_sd
    θ_nonDynamic = petabODECache.θ_nonDynamic
    jacobianGN = petabODECache.jacobianGN
    jacobianGN .= 0.0

    # Calculate gradient seperately for dynamic and non dynamic parameter.
    computeJacobianResidualsDynamicθ!((@view jacobianGN[θ_indices.iθ_dynamic, :]), θ_dynamic, θ_sd,
                                      θ_observable, θ_nonDynamic, petabModel, odeProblem,
                                      simulationInfo, θ_indices, measurementInfo, parameterInfo,
                                      solveOdeModelAllConditions!, cfg, petabODECache;
                                      expIDSolve=expIDSolve, reuseS=reuseS, splitOverConditions=splitOverConditions, 
                                      isRemade=isRemade)

    # Happens when at least one forward pass fails
    if !isempty(θ_dynamic) && all(jacobianGN[θ_indices.iθ_dynamic, :] .== 1e8)
        out .= 0.0
        return
    end
    @views ForwardDiff.jacobian!(jacobianGN[θ_indices.iθ_notOdeSystem, :]', computeResidualsNotSolveODE!, petabODECache.residualsGN, θ_est[θ_indices.iθ_notOdeSystem], cfgNotSolveODE)

    if priorInfo.hasPriors == true
        println("Warning : With Gauss Newton we do not support priors")
    end

    # In case of testing we might want to return the jacobian, else we are interested in the Guass-Newton approximaiton.
    if returnJacobian == false
        out .= jacobianGN * transpose(jacobianGN)
    else
        out .= jacobianGN
    end
end



# Compute prior contribution to log-likelihood, note θ in on the parameter scale (e.g might be on log-scale)
function computeHessianPrior!(hessian::AbstractVector,
                              θ::AbstractVector,
                              θ_indices::ParameterIndices,
                              priorInfo::PriorInfo)

    _evalPriors = (θ_est) -> begin
                                θ_estT =  transformθ(θ_est, θ_indices.θ_estNames, θ_indices)
                                return -1.0 * computePriors(θ_est, θ_estT, θ_indices.θ_estNames, priorInfo) # We work with -loglik
                            end
    hessian .+= ForwardDiff.hessian(_evalPriors, θ)
end
