#=
    The top-level functions for computing the hessian via i) exactly via autodiff, ii) block-approximation via
    auto-diff and iii) guass-newton approximation.
=#


function computeHessian!(hessian::Matrix{Float64},
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

    if splitOverConditions == false
        _evalHessian = (θ_est) -> computeCost(θ_est, odeProblem, petabModel, simulationInfo, θ_indices,
                                              measurementInfo, parameterInfo, changeODEProblemParameters!,
                                              solveOdeModelAllConditions!, priorInfo, computeHessian=true,
                                              expIDSolve=expIDSolve)

        if !isnothing(chunkSize)
            cfg = ForwardDiff.HessianConfig(_evalHessian, θ_est, ForwardDiff.Chunk(chunkSize))
        else
            cfg = ForwardDiff.HessianConfig(_evalHessian, θ_est, ForwardDiff.Chunk(θ_est))
        end

        # Only try to compute hessian if we could compute the cost
        if all([simulationInfo.odeSolutions[id].retcode == ReturnCode.Success for id in simulationInfo.experimentalConditionId])
            try
                ForwardDiff.hessian!(hessian, _evalHessian, θ_est, cfg)
                hessian .= Symmetric(hessian)
            catch
                hessian .= 0.0
            end
        else
            hessian .= 0.0
        end

    elseif splitOverConditions == true && simulationInfo.haspreEquilibrationConditionId == false
        hessian .= 0.0
        for conditionId in simulationInfo.experimentalConditionId
            mapConditionId = θ_indices.mapsConiditionId[conditionId]
            iθ_experimentalCondition = unique(vcat(θ_indices.mapODEProblem.iθDynamic, mapConditionId.iθDynamic, θ_indices.iθ_notOdeSystem))
            θ_input = θ_est[iθ_experimentalCondition]
            hTmp = zeros(length(θ_input), length(θ_input))
            computeCostDynamicθExpCond = (θ_arg) -> begin
                                                        _θ_est = convert.(eltype(θ_arg), θ_est)
                                                        _θ_est[iθ_experimentalCondition] .= θ_arg
                                                        return computeCost(_θ_est, odeProblem, petabModel, simulationInfo, θ_indices,
                                                                           measurementInfo, parameterInfo, changeODEProblemParameters!,
                                                                           solveOdeModelAllConditions!, priorInfo, computeHessian=true,
                                                                           expIDSolve=[conditionId])
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
                                                 petabModel::PEtabModel,
                                                 simulationInfo::SimulationInfo,
                                                 θ_indices::ParameterIndices,
                                                 measurementInfo::MeasurementsInfo,
                                                 parameterInfo::ParametersInfo,
                                                 changeODEProblemParameters!::Function,
                                                 solveOdeModelAllConditions!::Function,
                                                 priorInfo::PriorInfo;
                                                 reuseS::Bool=false,
                                                 returnJacobian::Bool=false,
                                                 expIDSolve::Vector{Symbol} = [:all])

    # Avoid incorrect non-zero values
    out .= 0.0

    θ_dynamic, θ_observable, θ_sd, θ_nonDynamic = splitParameterVector(θ_est, θ_indices)

    # In case the sensitivites are computed (as here) via automatic differentitation we need to pre-allocate an
    # sensitivity matrix all experimental conditions (to efficiently levarage autodiff and handle scenarios are
    # pre-equlibrita model). Here we pre-allocate said matrix.
    nModelStates = length(odeProblem.u0)
    nTimePointsSaveAt = sum(length(simulationInfo.timeObserved[experimentalConditionId]) for experimentalConditionId in simulationInfo.experimentalConditionId)
    # If the ForwardSensitivity equation approach used is autodiff the sensitivity needed by GN is already pre-allocated
    if isempty(simulationInfo.S)
        @assert reuseS == false
        S::Matrix{Float64} = zeros(Float64, (nTimePointsSaveAt*nModelStates, length(θ_dynamic)))
    else
        S = simulationInfo.S
    end

    # For Guass-Newton we compute the gradient via J*J' where J is the Jacobian of the residuals, here we pre-allocate
    # the entire matrix.
    jacobian::Matrix{Float64} = zeros(length(θ_est), length(measurementInfo.time))

    # Calculate gradient seperately for dynamic and non dynamic parameter.
    computeJacobianResidualsDynamicθ!((@view jacobian[θ_indices.iθ_dynamic, :]), θ_dynamic, θ_sd,
                                      θ_observable, θ_nonDynamic, S, petabModel, odeProblem,
                                      simulationInfo, θ_indices, measurementInfo, parameterInfo,
                                      changeODEProblemParameters!, solveOdeModelAllConditions!;
                                      expIDSolve=expIDSolve, reuseS=reuseS)

    # Happens when at least one forward pass fails
    if all(jacobian[θ_indices.iθ_dynamic, :] .== 1e8)
        out .= 0.0
        return
    end

    # Compute hessian for parameters which are not in ODE-system. Important to keep in mind that Sd- and observable
    # parameters can overlap in θ_est.
    iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = getIndicesParametersNotInODESystem(θ_indices)
    computeResidualsNotODESystemθ = (x) -> computeResidualsNotSolveODE(x[iθ_sd], x[iθ_observable],
                                                                       x[iθ_nonDynamic], petabModel, simulationInfo, θ_indices,
                                                                       measurementInfo, parameterInfo, expIDSolve=expIDSolve)
    @views ForwardDiff.jacobian!(jacobian[iθ_notOdeSystem, :]', computeResidualsNotODESystemθ, θ_est[iθ_notOdeSystem])

    if priorInfo.hasPriors == true
        println("Warning : With Gauss Newton we do not support priors")
    end

    # In case of testing we might want to return the jacobian, else we are interested in the Guass-Newton approximaiton.
    if returnJacobian == false
        out .= jacobian * jacobian'
    else
        out .= jacobian
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
