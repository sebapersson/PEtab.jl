function computeJacobianResidualsDynamicθ!(jacobian::Union{Matrix{Float64}, SubArray},
                                           θ_dynamic::Vector{Float64},
                                           θ_sd::Vector{Float64},
                                           θ_observable::Vector{Float64},
                                           θ_nonDynamic::Vector{Float64},
                                           petabModel::PEtabModel,
                                           odeProblem::ODEProblem,
                                           simulationInfo::SimulationInfo,
                                           θ_indices::ParameterIndices,
                                           measurementInfo::MeasurementsInfo,
                                           parameterInfo::ParametersInfo,
                                           solveOdeModelAllConditions!::Function, 
                                           cfg::ForwardDiff.JacobianConfig, 
                                           petabODECache::PEtabODEProblemCache;
                                           expIDSolve::Vector{Symbol} = [:all], 
                                           reuseS::Bool=false, 
                                           splitOverConditions::Bool=false, 
                                           isRemade::Bool=false)

    θ_dynamicT = transformθ(θ_dynamic, θ_indices.θ_dynamicNames, θ_indices, :θ_dynamic, petabODECache)
    θ_sdT = transformθ(θ_sd, θ_indices.θ_sdNames, θ_indices, :θ_sd, petabODECache)
    θ_observableT = transformθ(θ_observable, θ_indices.θ_observableNames, θ_indices, :θ_observable, petabODECache)
    θ_nonDynamicT = transformθ(θ_nonDynamic, θ_indices.θ_nonDynamicNames, θ_indices, :θ_nonDynamic, petabODECache)

    if reuseS == false
        # Solve the expanded ODE system for the sensitivites
        success = solveForSensitivites(odeProblem, simulationInfo, θ_indices, petabModel, :ForwardDiff, θ_dynamicT,
                                       solveOdeModelAllConditions!, cfg, petabODECache, expIDSolve, splitOverConditions, 
                                       isRemade)

        if success != true
            @warn "Failed to solve sensitivity equations"
            jacobian .= 1e8
            return
        end                                    
    end
    if isempty(θ_dynamic)
        jacobian .= 0.0
        return 
    end

    # Compute the gradient by looping through all experimental conditions.
    for i in eachindex(simulationInfo.experimentalConditionId)
        experimentalConditionId = simulationInfo.experimentalConditionId[i]
        simulationConditionId = simulationInfo.simulationConditionId[i]

        if expIDSolve[1] != :all && experimentalConditionId ∉ expIDSolve
            continue
        end

        sol = simulationInfo.odeSolutionsDerivatives[experimentalConditionId]

        # If we have a callback it needs to be properly handled
        computeJacobianResidualsExpCond!(jacobian, sol, petabODECache, θ_dynamicT, θ_sdT, θ_observableT, θ_nonDynamicT,
                                         experimentalConditionId, simulationConditionId, simulationInfo, petabModel, θ_indices,
                                         measurementInfo, parameterInfo)
    end
end


function computeJacobianResidualsExpCond!(jacobian::AbstractMatrix,
                                          sol::ODESolution,
                                          petabODECache::PEtabODEProblemCache,
                                          θ_dynamic::Vector{Float64},
                                          θ_sd::Vector{Float64},
                                          θ_observable::Vector{Float64},
                                          θ_nonDynamic::Vector{Float64},
                                          experimentalConditionId::Symbol,
                                          simulationConditionId::Symbol,
                                          simulationInfo::SimulationInfo,
                                          petabModel::PEtabModel,
                                          θ_indices::ParameterIndices,
                                          measurementInfo::MeasurementsInfo,
                                          parameterInfo::ParametersInfo)

    iPerTimePoint = simulationInfo.iPerTimePoint[experimentalConditionId]
    timeObserved = simulationInfo.timeObserved[experimentalConditionId]
    timePositionInODESolutions = simulationInfo.timePositionInODESolutions[experimentalConditionId]

    # To compute
    compute∂G∂u = (out, u, p, t, i, it) -> begin compute∂G∂_(out, u, p, t, i, it,
                                                             measurementInfo, parameterInfo,
                                                             θ_indices, petabModel,
                                                             θ_dynamic, θ_sd, θ_observable, θ_nonDynamic,
                                                             petabODECache.∂h∂u, petabODECache.∂σ∂u, compute∂G∂U=true,
                                                             computeResiduals=true)
                                            end
    compute∂G∂p = (out, u, p, t, i, it) -> begin compute∂G∂_(out, u, p, t, i, it,
                                                             measurementInfo, parameterInfo,
                                                             θ_indices, petabModel,
                                                             θ_dynamic, θ_sd, θ_observable, θ_nonDynamic,
                                                             petabODECache.∂h∂p, petabODECache.∂σ∂p, compute∂G∂U=false,
                                                             computeResiduals=true)
                                            end

    # Extract relevant parameters for the experimental conditions
    mapConditionId = θ_indices.mapsConiditionId[simulationConditionId]
    iθ_experimentalCondition = vcat(θ_indices.mapODEProblem.iθDynamic, mapConditionId.iθDynamic)

    # Loop through solution and extract sensitivites
    nModelStates = length(petabModel.stateNames)
    p = petabODECache.p
    p .= dualToFloat.(sol.prob.p)
    u = petabODECache.u
    ∂G∂p = petabODECache.∂G∂p
    ∂G∂u = petabODECache.∂G∂u
    _gradient = petabODECache._gradient
    for i in eachindex(timeObserved)
        u .= dualToFloat.((@view sol[:, i]))
        t = timeObserved[i]
        iStart, iEnd = (timePositionInODESolutions[i]-1)*nModelStates+1, (timePositionInODESolutions[i]-1)*nModelStates + nModelStates
        _S = @view petabODECache.S[iStart:iEnd, iθ_experimentalCondition]
        for iMeasurement in iPerTimePoint[i]
            compute∂G∂u(∂G∂u, u, p, t, 1, [[iMeasurement]])
            compute∂G∂p(∂G∂p, u, p, t, 1, [[iMeasurement]])
            @views _gradient[iθ_experimentalCondition] .= transpose(_S)*∂G∂u

            # Thus far have have computed dY/dθ for the residuals, but for parameters on the log-scale we want dY/dθ_log.
            # We can adjust via; dY/dθ_log = log(10) * θ * dY/dθ
            adjustGradientTransformedParameters!((@view jacobian[:, iMeasurement]), _gradient,
                                                 ∂G∂p, θ_dynamic, θ_indices, simulationConditionId, autoDiffSensitivites=true)

        end
    end
end


# To compute the gradient for non-dynamic parameters
function computeResidualsNotSolveODE!(residuals::AbstractVector,
                                      θ_sd::AbstractVector,
                                      θ_observable::AbstractVector,
                                      θ_nonDynamic::AbstractVector,
                                      petabModel::PEtabModel,
                                      simulationInfo::SimulationInfo,
                                      θ_indices::ParameterIndices,
                                      measurementInfo::MeasurementsInfo,
                                      parameterInfo::ParametersInfo, 
                                      petabODECache::PEtabODEProblemCache;
                                      expIDSolve::Vector{Symbol} = [:all])::AbstractVector

    # To be able to use ReverseDiff sdParamEstUse and obsParamEstUse cannot be overwritten.
    # Hence new vectors have to be created.
    θ_sdT = transformθ(θ_sd, θ_indices.θ_sdNames, θ_indices, :θ_sd, petabODECache)
    θ_observableT = transformθ(θ_observable, θ_indices.θ_observableNames, θ_indices, :θ_observable, petabODECache)
    θ_nonDynamicT = transformθ(θ_nonDynamic, θ_indices.θ_nonDynamicNames, θ_indices, :θ_nonDynamic, petabODECache)

    # Compute residuals per experimental conditions
    for experimentalConditionId in simulationInfo.experimentalConditionId

        if expIDSolve[1] != :all && experimentalConditionId ∉ expIDSolve
            continue
        end

        odeSolution = simulationInfo.odeSolutionsDerivatives[experimentalConditionId]
        sucess = computeResidualsExpCond!(residuals, odeSolution, θ_sdT, θ_observableT, θ_nonDynamicT,
                                          petabModel, experimentalConditionId, simulationInfo, θ_indices, measurementInfo,
                                          parameterInfo, petabODECache)
        if sucess == false
            residuals .= Inf
            break
        end
    end

    return residuals
end


# For an experimental condition compute residuals
function computeResidualsExpCond!(residuals::AbstractVector,
                                  odeSolution::ODESolution,
                                  θ_sd::AbstractVector,
                                  θ_observable::AbstractVector,
                                  θ_nonDynamic::AbstractVector,
                                  petabModel::PEtabModel,
                                  experimentalConditionId::Symbol,
                                  simulationInfo::SimulationInfo,
                                  θ_indices::ParameterIndices,
                                  measurementInfo::MeasurementsInfo,
                                  parameterInfo::ParametersInfo, 
                                  petabODECache::PEtabODEProblemCache)::Bool

    if !(odeSolution.retcode == ReturnCode.Success || odeSolution.retcode == ReturnCode.Terminated)
        return false
    end

    u = petabODECache.u
    p = petabODECache.p

    # Compute yMod and sd for all observations having id conditionID
    for iMeasurement in simulationInfo.iMeasurements[experimentalConditionId]

        t = measurementInfo.time[iMeasurement]
        u .= dualToFloat.((@view odeSolution[:, simulationInfo.iTimeODESolution[iMeasurement]]))
        p .= dualToFloat.(odeSolution.prob.p)
        hTransformed = computehTransformed(u, t, p, θ_observable, θ_nonDynamic, petabModel, iMeasurement, measurementInfo, θ_indices, parameterInfo)
        σ = computeσ(u, t, p, θ_sd, θ_nonDynamic, petabModel, iMeasurement, measurementInfo, θ_indices, parameterInfo)

        # By default a positive ODE solution is not enforced (even though the user can provide it as option).
        # In case with transformations on the data the code can crash, hence Inf is returned in case the
        # model data transformation can not be perfomred.
        if isinf(hTransformed)
            return false
        end

        if measurementInfo.measurementTransformation[iMeasurement] == :lin
            residuals[iMeasurement] = (hTransformed - measurementInfo.measurement[iMeasurement]) / σ
        elseif measurementInfo.measurementTransformation[iMeasurement] == :log10
            residuals[iMeasurement] = (hTransformed - measurementInfo.measurementT[iMeasurement]) / σ
        elseif measurementInfo.measurementTransformation[iMeasurement] == :log
            residuals[iMeasurement] = (hTransformed - measurementInfo.measurementT[iMeasurement]) / σ
        else
            println("Transformation ", measurementInfo.measurementTransformation[iMeasurement], " not yet supported.")
            return false
        end
    end
    return true
end
