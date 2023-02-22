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
        @views ForwardDiff.hessian!(hessian[θ_indices.iθ_dynamic, θ_indices.iθ_dynamic], computeCostDynamicθ, θ_dynamic, cfg)
    catch
        hessian .= 0.0
        return
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
