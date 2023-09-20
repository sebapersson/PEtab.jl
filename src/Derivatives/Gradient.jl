#=
    The top-level functions for computing the gradient via i) exactly via forward-mode autodiff, ii) forward sensitivty
    eqations, iii) adjoint sensitivity analysis and iv) Zygote interface.

    Due to it slow speed Zygote does not have full support for all models, e.g, models with priors and pre-eq criteria.
=#


# Compute the gradient via forward mode automatic differentitation
function compute_gradientAutoDiff!(gradient::Vector{Float64},
                                  θ_est::Vector{Float64},
                                  compute_costNotODESystemθ::Function,
                                  compute_costDynamicθ::Function,
                                  petabODECache::PEtabODEProblemCache,
                                  cfg::ForwardDiff.GradientConfig,
                                  simulation_info::SimulationInfo,
                                  θ_indices::ParameterIndices,
                                  priorInfo::PriorInfo,
                                  expIDSolve::Vector{Symbol} = [:all];
                                  isRemade::Bool=false)

    fill!(gradient, 0.0)
    splitParameterVector!(θ_est, θ_indices, petabODECache)
    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough. 
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final 
    # retcode we cannot catch these cases 
    simulation_info.couldSolve[1] = true

    # Case where based on the original PEtab file read into Julia we do not have any parameter vectors fixated.
    if isRemade == false || length(petabODECache.gradientDyanmicθ) == petabODECache.nθ_dynamicEst[1]
        tmp = petabODECache.nθ_dynamicEst[1]
        petabODECache.nθ_dynamicEst[1] = length(petabODECache.θ_dynamic)
        try
            # In case of no dynamic parameters we still need to solve the ODE in order to obtain the gradient for
            # non-dynamic parameters
            if length(petabODECache.gradientDyanmicθ) > 0
                ForwardDiff.gradient!(petabODECache.gradientDyanmicθ, compute_costDynamicθ, petabODECache.θ_dynamic, cfg)
                @views gradient[θ_indices.iθ_dynamic] .= petabODECache.gradientDyanmicθ
            else
                compute_costDynamicθ(petabODECache.θ_dynamic)
            end
        catch
            gradient .= 0.0
            return
        end
        petabODECache.nθ_dynamicEst[1] = tmp
    end

    # Case when we have dynamic parameters fixed. Here it is not always worth to move accross all chunks
    if !(isRemade == false || length(petabODECache.gradientDyanmicθ) == petabODECache.nθ_dynamicEst[1])
        try
            if petabODECache.nθ_dynamicEst[1] != 0
                C = length(cfg.seeds)
                nForwardPasses = Int64(ceil(petabODECache.nθ_dynamicEst[1] / C))
                __θ_dynamic = petabODECache.θ_dynamic[petabODECache.θ_dynamicInputOrder]
                forwardDiffGradientChunks(compute_costDynamicθ, petabODECache.gradientDyanmicθ, __θ_dynamic, ForwardDiff.Chunk(C); nForwardPasses=nForwardPasses)
                @views gradient[θ_indices.iθ_dynamic] .= petabODECache.gradientDyanmicθ[petabODECache.θ_dynamicOutputOrder]
            else
                compute_costDynamicθ(petabODECache.θ_dynamic)
            end
        catch
            gradient .= 0.0
            return
        end
    end

    # Check if we could solve the ODE (first), and if Inf was returned (second)
    if simulation_info.couldSolve[1] != true
        gradient .= 0.0
        return 
    end

    θ_notOdeSystem = @view θ_est[θ_indices.iθ_notOdeSystem]
    ForwardDiff.gradient!(petabODECache.gradientNotODESystemθ, compute_costNotODESystemθ, θ_notOdeSystem)
    @views gradient[θ_indices.iθ_notOdeSystem] .= petabODECache.gradientNotODESystemθ

    # If we have prior contribution its gradient is computed via autodiff for all parameters
    if priorInfo.hasPriors == true
        compute_gradientPrior!(gradient, θ_est, θ_indices, priorInfo)
    end
end


# Compute the gradient via forward mode automatic differentitation where the final gradient is computed via
# n ForwardDiff-calls accross all experimental condtions. The most efficient approach for models with many
# parameters which are unique to each experimental condition.
function compute_gradientAutoDiffSplitOverConditions!(gradient::Vector{Float64},
                                                     θ_est::Vector{Float64},
                                                     compute_costNotODESystemθ::Function,
                                                     _compute_costDynamicθ::Function,
                                                     petabODECache::PEtabODEProblemCache,
                                                     simulation_info::SimulationInfo,
                                                     θ_indices::ParameterIndices,
                                                     priorInfo::PriorInfo,
                                                     expIDSolve=[:all])

    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough. 
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final 
    # retcode we cannot catch these cases                                                         
    simulation_info.couldSolve[1] = true                                                     

    splitParameterVector!(θ_est, θ_indices, petabODECache)
    θ_dynamic = petabODECache.θ_dynamic
    petabODECache.gradientDyanmicθ .= 0.0

    for conditionId in simulation_info.experimentalConditionId
        mapConditionId = θ_indices.mapsConiditionId[conditionId]
        iθ_experimentalCondition = unique(vcat(θ_indices.mapODEProblem.iθDynamic, mapConditionId.iθDynamic))
        θ_input = θ_dynamic[iθ_experimentalCondition]
        compute_costDynamicθ = (θ_arg) ->    begin
                                                    _θ_dynamic = convert.(eltype(θ_arg), θ_dynamic)
                                                    @views _θ_dynamic[iθ_experimentalCondition] .= θ_arg
                                                    return _compute_costDynamicθ(_θ_dynamic, [conditionId])
                                            end
        try
            if length(θ_input) ≥ 1
                @views petabODECache.gradientDyanmicθ[iθ_experimentalCondition] .+= ForwardDiff.gradient(compute_costDynamicθ, θ_input)::Vector{Float64}
            else
                compute_costDynamicθ(θ_input)
            end
        catch
            gradient .= 1e8
            return
        end
    end

    # Check if we could solve the ODE (first), and if Inf was returned (second)
    if simulation_info.couldSolve[1] != true
        gradient .= 0.0
        return 
    end
    @views gradient[θ_indices.iθ_dynamic] .= petabODECache.gradientDyanmicθ

    θ_notOdeSystem = @view θ_est[θ_indices.iθ_notOdeSystem]
    ForwardDiff.gradient!(petabODECache.gradientNotODESystemθ, compute_costNotODESystemθ, θ_notOdeSystem)
    @views gradient[θ_indices.iθ_notOdeSystem] .= petabODECache.gradientNotODESystemθ

    # If we have prior contribution its gradient is computed via autodiff for all parameters
    if priorInfo.hasPriors == true
        compute_gradientPrior!(gradient, θ_est, θ_indices, priorInfo)
    end
end


# Compute the gradient via forward sensitivity equations
function compute_gradientForwardEquations!(gradient::Vector{Float64},
                                          θ_est::Vector{Float64},
                                          compute_costNotODESystemθ::Function,
                                          petab_model::PEtabModel,
                                          odeProblem::ODEProblem,
                                          sensealg,
                                          simulation_info::SimulationInfo,
                                          θ_indices::ParameterIndices,
                                          measurementInfo::MeasurementsInfo,
                                          parameterInfo::ParametersInfo,
                                          solveOdeModelAllConditions!::Function,
                                          priorInfo::PriorInfo,
                                          cfg::Union{ForwardDiff.JacobianConfig, Nothing},
                                          petabODECache::PEtabODEProblemCache;
                                          split_over_conditions::Bool=false,
                                          expIDSolve::Vector{Symbol} = [:all],
                                          isRemade::Bool=false)

    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough. 
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final 
    # retcode we cannot catch these cases                                             
    simulation_info.couldSolve[1] = true

    splitParameterVector!(θ_est, θ_indices, petabODECache)
    θ_dynamic = petabODECache.θ_dynamic
    θ_observable = petabODECache.θ_observable
    θ_sd = petabODECache.θ_sd
    θ_nonDynamic = petabODECache.θ_nonDynamic

    # Calculate gradient seperately for dynamic and non dynamic parameter.
    compute_gradientForwardEqDynamicθ!(petabODECache.gradientDyanmicθ, θ_dynamic, θ_sd, θ_observable, θ_nonDynamic, petab_model,
                                      sensealg, odeProblem, simulation_info, θ_indices, measurementInfo, parameterInfo,
                                      solveOdeModelAllConditions!, cfg, petabODECache, expIDSolve=expIDSolve,
                                      split_over_conditions=split_over_conditions, isRemade=isRemade)
    @views gradient[θ_indices.iθ_dynamic] .= petabODECache.gradientDyanmicθ

    # Happens when at least one forward pass fails and I set the gradient to 1e8
    if !isempty(petabODECache.gradientDyanmicθ) && all(petabODECache.gradientDyanmicθ .== 0.0)
        gradient .= 0.0
        return
    end

    θ_notOdeSystem = @view θ_est[θ_indices.iθ_notOdeSystem]
    ReverseDiff.gradient!(petabODECache.gradientNotODESystemθ, compute_costNotODESystemθ, θ_notOdeSystem)
    @views gradient[θ_indices.iθ_notOdeSystem] .= petabODECache.gradientNotODESystemθ

    if priorInfo.hasPriors == true
        compute_gradientPrior!(gradient, θ_est, θ_indices, priorInfo)
    end
end


# Compute prior contribution to log-likelihood
function compute_gradientPrior!(gradient::AbstractVector,
                               θ::AbstractVector,
                               θ_indices::ParameterIndices,
                               priorInfo::PriorInfo)

    _evalPriors = (θ_est) -> begin
                                θ_estT = transformθ(θ_est, θ_indices.θ_names, θ_indices)
                                return -1.0 * computePriors(θ_est, θ_estT, θ_indices.θ_names, priorInfo) # We work with -loglik
                            end
    gradient .+= ForwardDiff.gradient(_evalPriors, θ)
end
