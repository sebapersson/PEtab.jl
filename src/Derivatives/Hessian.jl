#=
    The top-level functions for computing the hessian via i) exactly via autodiff, ii) block-approximation via
    auto-diff and iii) guass-newton approximation.
=#


function compute_hessian!(hessian::Matrix{Float64},
                         θ_est::Vector{Float64},
                         _evalHessian::Function,
                         cfg::ForwardDiff.HessianConfig,
                         simulation_info::SimulationInfo,
                         θ_indices::ParameterIndices,
                         priorInfo::PriorInfo)

    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough. 
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final 
    # retcode we cannot catch these cases 
    simulation_info.couldSolve[1] = true
    if all([simulation_info.odeSolutions[id].retcode == ReturnCode.Success || simulation_info.odeSolutions[id].retcode == ReturnCode.Terminated for id in simulation_info.experimentalConditionId])
        try
            ForwardDiff.hessian!(hessian, _evalHessian, θ_est, cfg)
            @views hessian .= Symmetric(hessian)
        catch
            hessian .= 0.0
        end
    else
        hessian .= 0.0
    end

    # Check if we could solve the ODE (first), and if Inf was returned (second)
    if simulation_info.couldSolve[1] != true
        hessian .= 0.0
        return
    end

    if priorInfo.hasPriors == true
        compute_hessianPrior!(hessian, θ_est, θ_indices, priorInfo)
    end
end


# Compute the hessian via forward mode automatic differentitation where the final hessian is computed via
# n ForwardDiff-calls accross all experimental condtions. The most efficient approach for models with many
# parameters which are unique to each experimental condition.
function compute_hessianSplitOverConditions!(hessian::Matrix{Float64},
                                            θ_est::Vector{Float64},
                                            _evalHessian::Function,
                                            simulation_info::SimulationInfo,
                                            θ_indices::ParameterIndices,
                                            priorInfo::PriorInfo,
                                            expIDSolve::Vector{Symbol} = [:all])

    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough. 
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final 
    # retcode we cannot catch these cases                                             
    simulation_info.couldSolve[1] = true                                            

    hessian .= 0.0
    for conditionId in simulation_info.experimentalConditionId
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

    # Check if we could solve the ODE (first), and if Inf was returned (second)
    if simulation_info.couldSolve[1] != true
        hessian .= 0.0
        return
    end

    if priorInfo.hasPriors == true
        compute_hessianPrior!(hessian, θ_est, θ_indices, priorInfo)
    end
end


function compute_hessianBlockApproximation!(hessian::Matrix{Float64},
                                           θ_est::Vector{Float64},
                                           compute_costNotODESystemθ::Function,
                                           compute_costDynamicθ::Function,
                                           petabODECache::PEtabODEProblemCache,
                                           cfg::ForwardDiff.HessianConfig,
                                           simulation_info::SimulationInfo,
                                           θ_indices::ParameterIndices,
                                           priorInfo::PriorInfo;
                                           expIDSolve::Vector{Symbol} = [:all])

    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough. 
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final 
    # retcode we cannot catch these cases                                            
    simulation_info.couldSolve[1] = true

    # Avoid incorrect non-zero values
    hessian .= 0.0

    splitParameterVector!(θ_est, θ_indices, petabODECache)
    θ_dynamic = petabODECache.θ_dynamic

    try
        if !isempty(θ_indices.iθ_dynamic)
            @views ForwardDiff.hessian!(hessian[θ_indices.iθ_dynamic, θ_indices.iθ_dynamic], compute_costDynamicθ, θ_dynamic, cfg)
        else
            compute_costDynamicθ(θ_dynamic)
        end
    catch
        hessian .= 0.0
        return
    end

    # Check if we could solve the ODE (first), and if Inf was returned (second)
    if simulation_info.couldSolve[1] != true
        hessian .= 0.0
        return
    end

    iθ_notOdeSystem = θ_indices.iθ_notOdeSystem
    @views ForwardDiff.hessian!(hessian[iθ_notOdeSystem, iθ_notOdeSystem], compute_costNotODESystemθ, θ_est[iθ_notOdeSystem])

    # Even though this is a hessian approximation, due to ease of implementation and low run-time we compute the
    # full hessian for the priors
    if priorInfo.hasPriors == true
        compute_hessianPrior!(hessian, θ_est, θ_indices, priorInfo)
    end
end

function compute_hessianBlockApproximationSplitOverConditions!(hessian::Matrix{Float64},
                                                              θ_est::Vector{Float64},
                                                              compute_costNotODESystemθ::Function,
                                                              _compute_costDynamicθ::Function,
                                                              petabODECache::PEtabODEProblemCache,
                                                              simulation_info::SimulationInfo,
                                                              θ_indices::ParameterIndices,
                                                              priorInfo::PriorInfo;
                                                              expIDSolve::Vector{Symbol} = [:all])

    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough. 
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final 
    # retcode we cannot catch these cases                                                               
    simulation_info.couldSolve[1] = true

    # Avoid incorrect non-zero values
    hessian .= 0.0

    splitParameterVector!(θ_est, θ_indices, petabODECache)
    θ_dynamic = petabODECache.θ_dynamic

    for conditionId in simulation_info.experimentalConditionId
        mapConditionId = θ_indices.mapsConiditionId[conditionId]
        iθ_experimentalCondition = unique(vcat(θ_indices.mapODEProblem.iθDynamic, mapConditionId.iθDynamic))
        θ_input = θ_dynamic[iθ_experimentalCondition]
        hTmp = zeros(length(θ_input), length(θ_input))
        compute_costDynamicθ = (θ_arg) ->    begin
                                                    _θ_dynamic = convert.(eltype(θ_arg), θ_dynamic)
                                                    @views _θ_dynamic[iθ_experimentalCondition] .= θ_arg
                                                    return _compute_costDynamicθ(_θ_dynamic, [conditionId])
                                            end
        try
            ForwardDiff.hessian!(hTmp, compute_costDynamicθ, θ_input)
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
    if simulation_info.couldSolve[1] != true
        hessian .= 0.0
        return
    end

    iθ_notOdeSystem = θ_indices.iθ_notOdeSystem
    @views ForwardDiff.hessian!(hessian[iθ_notOdeSystem, iθ_notOdeSystem], compute_costNotODESystemθ, θ_est[iθ_notOdeSystem])

    # Even though this is a hessian approximation, due to ease of implementation and low run-time we compute the
    # full hessian for the priors
    if priorInfo.hasPriors == true
        compute_hessianPrior!(hessian, θ_est, θ_indices, priorInfo)
    end
end


function computeGaussNewtonHessianApproximation!(out::Matrix{Float64},
                                                 θ_est::Vector{Float64},
                                                 odeProblem::ODEProblem,
                                                 compute_residualsNotSolveODE!::Function,
                                                 petab_model::PEtabModel,
                                                 simulation_info::SimulationInfo,
                                                 θ_indices::ParameterIndices,
                                                 measurementInfo::MeasurementsInfo,
                                                 parameterInfo::ParametersInfo,
                                                 solveOdeModelAllConditions!::Function,
                                                 priorInfo::PriorInfo,
                                                 cfg::ForwardDiff.JacobianConfig,
                                                 cfgNotSolveODE::ForwardDiff.JacobianConfig,
                                                 petabODECache::PEtabODEProblemCache;
                                                 reuse_sensitivities::Bool=false,
                                                 split_over_conditions::Bool=false,
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
                                      θ_observable, θ_nonDynamic, petab_model, odeProblem,
                                      simulation_info, θ_indices, measurementInfo, parameterInfo,
                                      solveOdeModelAllConditions!, cfg, petabODECache;
                                      expIDSolve=expIDSolve, reuse_sensitivities=reuse_sensitivities, split_over_conditions=split_over_conditions,
                                      isRemade=isRemade)

    # Happens when at least one forward pass fails
    if !isempty(θ_dynamic) && all(jacobianGN[θ_indices.iθ_dynamic, :] .== 1e8)
        out .= 0.0
        return
    end
    @views ForwardDiff.jacobian!(jacobianGN[θ_indices.iθ_notOdeSystem, :]', compute_residualsNotSolveODE!, petabODECache.residualsGN, θ_est[θ_indices.iθ_notOdeSystem], cfgNotSolveODE)

    # In case of testing we might want to return the jacobian, else we are interested in the Guass-Newton approximaiton.
    if returnJacobian == false
        out .= jacobianGN * transpose(jacobianGN)
    else
        out .= jacobianGN
        # Even though this is a hessian approximation, due to ease of implementation and low run-time we compute the
        # full hessian for the priors
        if priorInfo.hasPriors == true
            compute_hessianPrior!(out, θ_est, θ_indices, priorInfo)
        end
    end
end



# Compute prior contribution to log-likelihood, note θ in on the parameter scale (e.g might be on log-scale)
function compute_hessianPrior!(hessian::AbstractMatrix,
                              θ::AbstractVector,
                              θ_indices::ParameterIndices,
                              priorInfo::PriorInfo)

    _evalPriors = (θ_est) -> begin
                                θ_estT =  transformθ(θ_est, θ_indices.θ_names, θ_indices)
                                return -1.0 * computePriors(θ_est, θ_estT, θ_indices.θ_names, priorInfo) # We work with -loglik
                            end
    hessian .+= ForwardDiff.hessian(_evalPriors, θ)
end
