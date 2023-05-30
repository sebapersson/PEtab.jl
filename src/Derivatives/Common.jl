function getIndicesParametersNotInODESystem(θ_indices::ParameterIndices)::Tuple

    θ_observableNames = θ_indices.θ_observableNames
    θ_sdNames = θ_indices.θ_sdNames
    θ_nonDynamicNames = θ_indices.θ_nonDynamicNames
    iθ_notOdeSystemNames = θ_indices.θ_notOdeSystemNames

    iθ_sd = [findfirst(x -> x == θ_sdNames[i], iθ_notOdeSystemNames) for i in eachindex(θ_sdNames)]
    iθ_observable = [findfirst(x -> x == θ_observableNames[i],  iθ_notOdeSystemNames) for i in eachindex(θ_observableNames)]
    iθ_nonDynamic = [findfirst(x -> x == θ_nonDynamicNames[i],  iθ_notOdeSystemNames) for i in eachindex(θ_nonDynamicNames)]
    iθ_notOdeSystem = θ_indices.iθ_notOdeSystem

    return iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem
end


function couldSolveODEModel(simulationInfo::SimulationInfo, expIDSolve::Vector{Symbol})::Bool
    for experimentalId in simulationInfo.experimentalConditionId
        if !(expIDSolve[1] == :all || experimentalId ∈ expIDSolve)
            continue
        end
        if !(simulationInfo.odeSolutionsDerivatives[experimentalId].retcode == ReturnCode.Success || 
             simulationInfo.odeSolutionsDerivatives[experimentalId].retcode == ReturnCode.Terminated)
            return false
        end
    end
    return true
end


# Function to compute ∂G∂u and ∂G∂p for an observation assuming a fixed ODE-solution
function compute∂G∂_(∂G∂_,
                     u::AbstractVector,
                     p::Vector{Float64}, # odeProblem.p
                     t::Float64,
                     i::Integer,
                     iPerTimePoint::Vector{Vector{Int64}},
                     measurementInfo::MeasurementsInfo,
                     parameterInfo::ParametersInfo,
                     θ_indices::ParameterIndices,
                     petabModel::PEtabModel,
                     θ_sd::Vector{Float64},
                     θ_observable::Vector{Float64},
                     θ_nonDynamic::Vector{Float64},
                     ∂h∂_::Vector{Float64},
                     ∂σ∂_::Vector{Float64};
                     compute∂G∂U::Bool=true,
                     computeResiduals::Bool=false)

    fill!(∂G∂_, 0.0)
    for iMeasurementData in iPerTimePoint[i]
        fill!(∂h∂_, 0.0)
        fill!(∂σ∂_, 0.0)

        hTransformed = computehTransformed(u, t, p, θ_observable, θ_nonDynamic, petabModel, iMeasurementData, measurementInfo, θ_indices, parameterInfo)
        σ = computeσ(u, t, p, θ_sd, θ_nonDynamic, petabModel, iMeasurementData, measurementInfo, θ_indices, parameterInfo)

        # Maps needed to correctly extract the right SD and observable parameters
        mapθ_sd = θ_indices.mapθ_sd[iMeasurementData]
        mapθ_observable = θ_indices.mapθ_observable[iMeasurementData]
        if compute∂G∂U == true
            petabModel.compute_∂h∂u!(u, t, p, θ_observable, θ_nonDynamic, measurementInfo.observableId[iMeasurementData], mapθ_observable, ∂h∂_)
            petabModel.compute_∂σ∂u!(u, t, θ_sd, p, θ_nonDynamic, parameterInfo, measurementInfo.observableId[iMeasurementData], mapθ_sd, ∂σ∂_)
        else
            petabModel.compute_∂h∂p!(u, t, p, θ_observable, θ_nonDynamic, measurementInfo.observableId[iMeasurementData], mapθ_observable, ∂h∂_)
            petabModel.compute_∂σ∂p!(u, t, θ_sd, p, θ_nonDynamic, parameterInfo, measurementInfo.observableId[iMeasurementData], mapθ_sd, ∂σ∂_)
        end

        if measurementInfo.measurementTransformation[iMeasurementData] === :log10
            yObs = measurementInfo.measurementT[iMeasurementData]
            ∂h∂_ .*= 1 / (log(10) * exp10(hTransformed))
        elseif measurementInfo.measurementTransformation[iMeasurementData] === :log
            yObs = measurementInfo.measurementT[iMeasurementData]
            ∂h∂_ .*= 1 / exp(hTransformed)
        elseif measurementInfo.measurementTransformation[iMeasurementData] === :lin
            yObs = measurementInfo.measurement[iMeasurementData]
        end

        # In case of Guass Newton approximation we target the residuals (y_mod - y_obs)/σ
        if computeResiduals == false
            ∂G∂h = ( hTransformed - yObs ) / σ^2
            ∂G∂σ = 1/σ - (( hTransformed - yObs )^2 / σ^3)
        else
            ∂G∂h = 1.0 / σ
            ∂G∂σ = -(hTransformed - yObs) / σ^2
        end

        @views ∂G∂_ .+= (∂G∂h*∂h∂_ .+ ∂G∂σ*∂σ∂_)[:]
    end
    return
end


function adjustGradientTransformedParameters!(gradient::Union{AbstractVector, SubArray},
                                              _gradient::AbstractVector,
                                              ∂G∂p::AbstractVector,
                                              θ_dynamic::Vector{Float64},
                                              θ_indices::ParameterIndices,
                                              simulationConditionId::Symbol;
                                              autoDiffSensitivites::Bool=false,
                                              adjoint::Bool=false)

    mapConditionId = θ_indices.mapsConiditionId[simulationConditionId]
    mapODEProblem = θ_indices.mapODEProblem

    # Transform gradient parameter that for each experimental condition appear in the ODE system
    iChange = θ_indices.mapODEProblem.iθDynamic
    if autoDiffSensitivites == true
        gradient1 = _gradient[mapODEProblem.iθDynamic] .+ ∂G∂p[mapODEProblem.iODEProblemθDynamic]
    else
        gradient1 = _gradient[mapODEProblem.iODEProblemθDynamic] .+ ∂G∂p[mapODEProblem.iODEProblemθDynamic]
    end
    @views gradient[iChange] .+= _adjustGradientTransformedParameters(gradient1,
                                                                      θ_dynamic[mapODEProblem.iθDynamic],
                                                                      θ_indices.θ_dynamicNames[mapODEProblem.iθDynamic],
                                                                      θ_indices)                                                                    

    # For forward sensitives via autodiff ∂G∂p is on the same scale as odeProblem.p, while
    # S-matrix is on the same scale as θ_dynamic. To be able to handle condition specific
    # parameters mapping to several odeProblem.p parameters the sensitivity matrix part and
    # ∂G∂p must be treated seperately.
    if autoDiffSensitivites == true   
        _iθDynamic = unique(mapConditionId.iθDynamic)      
        gradient[_iθDynamic] .+= _adjustGradientTransformedParameters(_gradient[_iθDynamic],
                                                                      θ_dynamic[_iθDynamic],
                                                                      θ_indices.θ_dynamicNames[_iθDynamic],
                                                                      θ_indices)
        out = _adjustGradientTransformedParameters(∂G∂p[mapConditionId.iODEProblemθDynamic],
                                                   θ_dynamic[mapConditionId.iθDynamic],
                                                   θ_indices.θ_dynamicNames[mapConditionId.iθDynamic],
                                                   θ_indices)
        @inbounds for i in eachindex(mapConditionId.iODEProblemθDynamic)
            gradient[mapConditionId.iθDynamic[i]] += out[i]
        end
    end

    # Here both ∂G∂p and _gradient are on the same scale a odeProblem.p. One condition specific parameter 
    # can map to several parameters in odeProblem.p
    if adjoint == true || autoDiffSensitivites == false
        out = _adjustGradientTransformedParameters(_gradient[mapConditionId.iODEProblemθDynamic] .+ ∂G∂p[mapConditionId.iODEProblemθDynamic],
                                                   θ_dynamic[mapConditionId.iθDynamic],
                                                   θ_indices.θ_dynamicNames[mapConditionId.iθDynamic],
                                                   θ_indices)
        @inbounds for i in eachindex(mapConditionId.iODEProblemθDynamic)
            gradient[mapConditionId.iθDynamic[i]] += out[i]
        end
    end
end


function _adjustGradientTransformedParameters(_gradient::AbstractVector{T},
                                              θ::AbstractVector{T},
                                              θ_names::AbstractVector{Symbol},
                                              θ_indices::ParameterIndices)::Vector{T} where T

    out = similar(_gradient)
    @inbounds for (i, θ_name) in pairs(θ_names)
        θ_scale = θ_indices.θ_scale[θ_name]
        if θ_scale === :log10
            out[i] = log(10) * _gradient[i] * θ[i]
        elseif θ_scale === :log
            out[i] = _gradient[i] * θ[i]
        elseif θ_scale === :lin
            out[i] = _gradient[i]
        end
    end

    return out
end
