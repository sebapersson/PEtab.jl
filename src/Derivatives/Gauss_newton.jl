function computeJacobianResidualsDynamicθ!(jacobian::Union{Matrix{Float64}, SubArray},
                                           θ_dynamic::Vector{Float64},
                                           θ_sd::Vector{Float64},
                                           θ_observable::Vector{Float64},
                                           θ_nonDynamic::Vector{Float64},
                                           S::Matrix{Float64},
                                           petabModel::PEtabModel,
                                           odeProblem::ODEProblem,
                                           simulationInfo::SimulationInfo,
                                           θ_indices::ParameterIndices,
                                           measurementInfo::MeasurementsInfo,
                                           parameterInfo::ParametersInfo,
                                           changeODEProblemParameters!::Function,
                                           solveOdeModelAllConditions!::Function;
                                           expIDSolve::Vector{Symbol} = [:all], 
                                           reuseS::Bool=false)

    θ_dynamicT = transformθ(θ_dynamic, θ_indices.θ_dynamicNames, θ_indices)
    θ_sdT = transformθ(θ_sd, θ_indices.θ_sdNames, θ_indices)
    θ_observableT = transformθ(θ_observable, θ_indices.θ_observableNames, θ_indices)
    θ_nonDynamicT = transformθ(θ_nonDynamic, θ_indices.θ_nonDynamicNames, θ_indices)

    # Solve the expanded ODE system for the sensitivites. When running optmization algorithms like Fides the same
    # sensitivity matrix can be used when computing both the hessian approxmiation and gradient
    if reuseS == false
        success = solveForSensitivites(S, odeProblem, simulationInfo, petabModel, :AutoDiff, θ_dynamicT,
                                       solveOdeModelAllConditions!, changeODEProblemParameters!, expIDSolve)
        if success != true
            println("Failed to solve sensitivity equations")
            jacobian .= 1e8
            return
        end
    end

    jacobian .= 0.0
    # Compute the gradient by looping through all experimental conditions.
    for i in eachindex(simulationInfo.experimentalConditionId)
        experimentalConditionId = simulationInfo.experimentalConditionId[i]
        simulationConditionId = simulationInfo.simulationConditionId[i]

        if expIDSolve[1] != :all && experimentalConditionId ∉ expIDSolve
            continue
        end

        sol = simulationInfo.odeSolutionsDerivatives[experimentalConditionId]

        # If we have a callback it needs to be properly handled
        computeJacobianResidualsExpCond!(jacobian, sol, S, θ_dynamicT, θ_sdT, θ_observableT, θ_nonDynamicT,
                                         experimentalConditionId, simulationConditionId, simulationInfo, petabModel, θ_indices,
                                         measurementInfo, parameterInfo)
    end
end


function computeJacobianResidualsExpCond!(jacobian::AbstractMatrix,
                                          sol::ODESolution,
                                          S::Matrix{Float64},
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

    # Pre allcoate vectors needed for computations
    ∂h∂u, ∂σ∂u, ∂h∂p, ∂σ∂p = allocateObservableFunctionDerivatives(sol, petabModel)

    # To compute
    compute∂G∂u = (out, u, p, t, i, it) -> begin compute∂G∂_(out, u, p, t, i, it,
                                                             measurementInfo, parameterInfo,
                                                             θ_indices, petabModel,
                                                             θ_dynamic, θ_sd, θ_observable, θ_nonDynamic,
                                                             ∂h∂u, ∂σ∂u, compute∂G∂U=true,
                                                             computeResiduals=true)
                                            end
    compute∂G∂p = (out, u, p, t, i, it) -> begin compute∂G∂_(out, u, p, t, i, it,
                                                             measurementInfo, parameterInfo,
                                                             θ_indices, petabModel,
                                                             θ_dynamic, θ_sd, θ_observable, θ_nonDynamic,
                                                             ∂h∂p, ∂σ∂p, compute∂G∂U=false,
                                                             computeResiduals=true)
                                            end

    # Extract relevant parameters for the experimental conditions
    mapConditionId = θ_indices.mapsConiditionId[simulationConditionId]
    iθ_experimentalCondition = vcat(θ_indices.mapODEProblem.iθDynamic, mapConditionId.iθDynamic)

    # Loop through solution and extract sensitivites
    nModelStates = length(petabModel.stateNames)
    p = dualToFloat.(sol.prob.p)
    ∂G∂p = zeros(Float64, length(sol.prob.p))
    ∂G∂u = zeros(Float64, nModelStates)
    _gradient = zeros(Float64, length(θ_indices.iθ_dynamic))
    for i in eachindex(timeObserved)
        u = dualToFloat.(sol[:, i])
        t = timeObserved[i]
        iStart, iEnd = (timePositionInODESolutions[i]-1)*nModelStates+1, (timePositionInODESolutions[i]-1)*nModelStates + nModelStates
        _S = @view S[iStart:iEnd, iθ_experimentalCondition]
        for iMeasurement in iPerTimePoint[i]
            compute∂G∂u(∂G∂u, u, p, t, 1, [[iMeasurement]])
            compute∂G∂p(∂G∂p, u, p, t, 1, [[iMeasurement]])
            _gradient .= 0.0 # Not-ideal for performance (indexing tricky though)
            @views _gradient[iθ_experimentalCondition] .+= _S'*∂G∂u

            # Thus far have have computed dY/dθ for the residuals, but for parameters on the log-scale we want dY/dθ_log.
            # We can adjust via; dY/dθ_log = log(10) * θ * dY/dθ
            adjustGradientTransformedParameters!((@views jacobian[:, iMeasurement]), _gradient,
                                                 ∂G∂p, θ_dynamic, θ_indices, simulationConditionId, autoDiffSensitivites=true)

        end
    end
end


# To compute the gradient for non-dynamic parameters
function computeResidualsNotSolveODE(θ_sd::AbstractVector,
                                     θ_observable::AbstractVector,
                                     θ_nonDynamic::AbstractVector,
                                     petabModel::PEtabModel,
                                     simulationInfo::SimulationInfo,
                                     θ_indices::ParameterIndices,
                                     measurementInfo::MeasurementsInfo,
                                     parameterInfo::ParametersInfo;
                                     expIDSolve::Vector{Symbol} = [:all])::AbstractVector

    residuals = zeros(eltype(θ_sd), length(measurementInfo.time))

    # To be able to use ReverseDiff sdParamEstUse and obsParamEstUse cannot be overwritten.
    # Hence new vectors have to be created.
    θ_sdT = transformθ(θ_sd, θ_indices.θ_sdNames, θ_indices)
    θ_observableT = transformθ(θ_observable, θ_indices.θ_observableNames, θ_indices)
    θ_nonDynamicT = transformθ(θ_nonDynamic, θ_indices.θ_nonDynamicNames, θ_indices)

    # Compute residuals per experimental conditions
    odeSolutions = simulationInfo.odeSolutionsDerivatives
    for experimentalConditionId in simulationInfo.experimentalConditionId

        if expIDSolve[1] != :all && experimentalConditionId ∉ expIDSolve
            continue
        end

        odeSolution = odeSolutions[experimentalConditionId]
        sucess = computeResidualsExpCond!(residuals, odeSolution, θ_sdT, θ_observableT, θ_nonDynamicT,
                                          petabModel, experimentalConditionId, simulationInfo, θ_indices, measurementInfo,
                                          parameterInfo)
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
                                  parameterInfo::ParametersInfo)::Bool

    if !(odeSolution.retcode == ReturnCode.Success || odeSolution.retcode == ReturnCode.Terminated)
        return false
    end

    # Compute yMod and sd for all observations having id conditionID
    nModelStates = length(petabModel.stateNames)
    for iMeasurement in simulationInfo.iMeasurements[experimentalConditionId]

        t = measurementInfo.time[iMeasurement]
        u = dualToFloat.(odeSolution[1:nModelStates, simulationInfo.iTimeODESolution[iMeasurement]])
        p = dualToFloat.(odeSolution.prob.p)
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
        else
            println("Transformation ", measurementInfo.measurementTransformation[iMeasurement], "not yet supported.")
            return false
        end
    end
    return true
end
