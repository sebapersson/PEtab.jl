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

    ForwardDiff.hessian!(hessian, _evalHessian, θ_est, cfg)
    # Only try to compute hessian if we could compute the cost
    if all([simulationInfo.odeSolutions[id].retcode == ReturnCode.Success for id in simulationInfo.experimentalConditionId])
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


function computeHessianBlockApproximation!(hessian::Matrix{Float64},
                                           θ_est::Vector{Float64},
                                           odeProblem::ODEProblem,
                                           petabModel::PEtabModel,
                                           simulationInfo::SimulationInfo,
                                           θ_indices::ParameterIndices,
                                           measurementInfo::MeasurementsInfo,
                                           parameterInfo::ParametersInfo,
                                           changeODEProblemParameters!::Function,
                                           solveOdeModelAllConditions!::Function,
                                           priorInfo::PriorInfo,
                                           chunkSize::Union{Nothing, Int64};
                                           splitOverConditions::Bool=false,
                                           expIDSolve::Vector{Symbol} = [:all])

    # Avoid incorrect non-zero values
    hessian .= 0.0

    θ_dynamic, θ_observable, θ_sd, θ_nonDynamic = splitParameterVector(θ_est, θ_indices)

    if splitOverConditions == false
        # Compute hessian for parameters which are a part of the ODE-system (dynamic parameters)
        computeCostDynamicθ = (x) -> computeCostSolveODE(x, θ_sd, θ_observable, θ_nonDynamic, odeProblem, petabModel,
                                                        simulationInfo, θ_indices, measurementInfo, parameterInfo,
                                                        changeODEProblemParameters!, solveOdeModelAllConditions!,
                                                        computeHessian=true,
                                                        expIDSolve=expIDSolve)

        if !isnothing(chunkSize)
            cfg = ForwardDiff.HessianConfig(computeCostDynamicθ, θ_dynamic, ForwardDiff.Chunk(chunkSize))
        else
            cfg = ForwardDiff.HessianConfig(computeCostDynamicθ, θ_dynamic, ForwardDiff.Chunk(θ_dynamic))
        end

        try
            @views ForwardDiff.hessian!(hessian[θ_indices.iθ_dynamic, θ_indices.iθ_dynamic], computeCostDynamicθ, θ_dynamic, cfg)
        catch
            hessian .= 0.0
            return
        end

    elseif splitOverConditions == true && simulationInfo.haspreEquilibrationConditionId == false
        hessian .= 0.0
        for conditionId in simulationInfo.experimentalConditionId
            mapConditionId = θ_indices.mapsConiditionId[conditionId]
            iθ_experimentalCondition = unique(vcat(θ_indices.mapODEProblem.iθDynamic, mapConditionId.iθDynamic))
            θ_input = θ_dynamic[iθ_experimentalCondition]
            hTmp = zeros(length(θ_input), length(θ_input))
            computeCostDynamicθExpCond = (θ_arg) -> begin
                                                        _θ_dynamic = convert.(eltype(θ_arg), θ_dynamic)
                                                        _θ_dynamic[iθ_experimentalCondition] .= θ_arg
                                                        return computeCostSolveODE(_θ_dynamic, θ_sd, θ_observable, θ_nonDynamic, odeProblem, petabModel,
                                                                                   simulationInfo, θ_indices, measurementInfo, parameterInfo,
                                                                                   changeODEProblemParameters!, solveOdeModelAllConditions!,
                                                                                   computeHessian=true, expIDSolve=[conditionId])
                                                    end
            try
                ForwardDiff.hessian!(hTmp, computeCostDynamicθExpCond, θ_input)
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

    else splitOverConditions == true && simulationInfo.haspreEquilibrationConditionId == true
        println("Compatabillity error : Currently we only support to split gradient compuations accross experimentalConditionId:s for models without preequilibration")
    end

    # Check if we could solve the ODE (first), and if Inf was returned (second)
    if couldSolveODEModel(simulationInfo, expIDSolve) == false
        hessian .= 0.0
    end
    if all(hessian[θ_indices.iθ_dynamic, θ_indices.iθ_dynamic] .== 0.0)
        return
    end

    # Compute hessian for parameters which are not in ODE-system. Important to keep in mind that Sd- and observable
    # parameters can overlap in θ_est.
    iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = getIndicesParametersNotInODESystem(θ_indices)
    computeCostNotODESystemθ = (x) -> computeCostNotSolveODE(x[iθ_sd], x[iθ_observable], x[iθ_nonDynamic],
                                                             petabModel, simulationInfo, θ_indices, measurementInfo,
                                                             parameterInfo, expIDSolve=expIDSolve,
                                                             computeGradientNotSolveAutoDiff=true)
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
                                                 changeODEProblemParameters!::Function,
                                                 solveOdeModelAllConditions!::Function,
                                                 priorInfo::PriorInfo, 
                                                 cfg::ForwardDiff.JacobianConfig, 
                                                 cfgNotSolveODE::ForwardDiff.JacobianConfig,
                                                 petabODECache::PEtabODEProblemCache;
                                                 reuseS::Bool=false,
                                                 returnJacobian::Bool=false,
                                                 expIDSolve::Vector{Symbol} = [:all])

    # Avoid incorrect non-zero values
    out .= 0.0

    splitParameterVector!(θ_est, θ_indices, petabODECache)
    θ_dynamic = petabODECache.θ_dynamic 
    θ_observable = petabODECache.θ_observable
    θ_sd = petabODECache.θ_sd
    θ_nonDynamic = petabODECache.θ_nonDynamic
    jacobianGN = petabODECache.jacobianGN

    # Calculate gradient seperately for dynamic and non dynamic parameter.
    computeJacobianResidualsDynamicθ!((@view jacobianGN[θ_indices.iθ_dynamic, :]), θ_dynamic, θ_sd,
                                      θ_observable, θ_nonDynamic, petabModel, odeProblem,
                                      simulationInfo, θ_indices, measurementInfo, parameterInfo,
                                      changeODEProblemParameters!, solveOdeModelAllConditions!, 
                                      cfg, petabODECache;
                                      expIDSolve=expIDSolve, reuseS=reuseS)

    # Happens when at least one forward pass fails
    if all(jacobianGN[θ_indices.iθ_dynamic, :] .== 1e8)
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
                                return computePriors(θ_est, θ_estT, θ_indices.θ_estNames, priorInfo)
                            end
    hessian .+= ForwardDiff.hessian(_evalPriors, θ)
end
