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
        if expIDSolve[1] == :all || experimentalId ∈ expIDSolve
            if simulationInfo.odeSolutionsDerivatives[experimentalId].retcode != ReturnCode.Success
                return false
            end
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
                     θ_dynamic::Vector{Float64},
                     θ_sd::Vector{Float64}, 
                     θ_observable::Vector{Float64}, 
                     θ_nonDynamic::Vector{Float64}, 
                     ∂h∂_::Vector{Float64}, 
                     ∂σ∂_::Vector{Float64}; 
                     compute∂G∂U::Bool=true, 
                     computeResiduals::Bool=false)

    ∂G∂_ .= 0.0
    for iMeasurementData in iPerTimePoint[i]
        ∂h∂_ .= 0.0
        ∂σ∂_ .= 0.0

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

        if measurementInfo.measurementTransformation[iMeasurementData] == :log10
            yObs = measurementInfo.measurementT[iMeasurementData]
            ∂h∂_ .*= 1 / (log(10) * exp10(hTransformed))
        elseif measurementInfo.measurementTransformation[iMeasurementData] == :lin
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

        ∂G∂_ .+= (∂G∂h*∂h∂_ .+ ∂G∂σ*∂σ∂_)[:]
    end
    return
end


# Allocate derivates needed when computing ∂G∂u and ∂G∂p
function allocateObservableFunctionDerivatives(sol::ODESolution, petabModel::PEtabModel)

    nModelStates = length(petabModel.stateNames)
    ∂h∂u = zeros(Float64, nModelStates)
    ∂σ∂u = zeros(Float64, nModelStates)
    ∂h∂p = zeros(Float64, length(sol.prob.p))
    ∂σ∂p = zeros(Float64, length(sol.prob.p))
    return ∂h∂u, ∂σ∂u, ∂h∂p, ∂σ∂p
end


function adjustGradientTransformedParameters!(gradient::Union{AbstractVector, SubArray}, 
                                              _gradient::AbstractVector, 
                                              ∂G∂p::Union{AbstractVector, Nothing}, 
                                              θ_dynamic::Vector{Float64},
                                              θ_indices::ParameterIndices,
                                              simulationConditionId::Symbol; 
                                              autoDiffSensitivites::Bool=false, 
                                              adjoint::Bool=false)

    mapConditionId = θ_indices.mapsConiditionId[simulationConditionId]                                 
    
    #iθ_experimentalCondition = vcat(θ_indices.mapODEProblem.iθDynamic, mapConditionId.iθDynamic)    
    
    # In case we compute the sensitivtes via automatic differentation the parameters in _gradient=S'*∂G∂u will appear in the 
    # same order as they appear in θ_est. In case we do not compute sensitivtes via autodiff, or do adjoint sensitity analysis, 
    # the parameters in _gradient=S'∂G∂u appear in the same order as in odeProblem.p.
    if autoDiffSensitivites == true && adjoint == false
        _gradient1 = _gradient[θ_indices.mapODEProblem.iθDynamic] .+ ∂G∂p[θ_indices.mapODEProblem.iODEProblemθDynamic]
        _gradient2 = _gradient[unique(mapConditionId.iθDynamic)] 

    elseif adjoint == false
        _gradient1 = _gradient[θ_indices.mapODEProblem.iODEProblemθDynamic] .+ ∂G∂p[θ_indices.mapODEProblem.iODEProblemθDynamic]
        _gradient2 = _gradient[mapConditionId.iODEProblemθDynamic] .+ ∂G∂p[mapConditionId.iODEProblemθDynamic]
    end

    # For adjoint sensitivity analysis ∂G∂p is already incorperated into the gradient, and the parameters appear in the 
    # same order as in ODEProblem 
    if adjoint == true
        _gradient1 = _gradient[θ_indices.mapODEProblem.iODEProblemθDynamic] 
        _gradient2 = _gradient[mapConditionId.iODEProblemθDynamic]
    end
    
    # Transform gradient parameter that for each experimental condition appear in the ODE system  
    iChange = θ_indices.mapODEProblem.iθDynamic                                                          
    gradient[iChange] .+= _adjustGradientTransformedParameters(_gradient1,
                                                               θ_dynamic[θ_indices.mapODEProblem.iθDynamic], 
                                                               θ_indices.θ_dynamicNames[θ_indices.mapODEProblem.iθDynamic], 
                                                               θ_indices)
    
    # Transform gradient for parameters which are specific to certain experimental conditions. Here we must account that 
    # for an experimental condition on parameter can map to several of the parameters in the ODE-system, which is solved 
    # by the for-loop.
    if autoDiffSensitivites == false
        out  = _adjustGradientTransformedParameters(_gradient2,
                                                    θ_dynamic[mapConditionId.iθDynamic], 
                                                    θ_indices.θ_dynamicNames[mapConditionId.iθDynamic], 
                                                    θ_indices)     
        @inbounds for i in eachindex(mapConditionId.iθDynamic)                                                
            gradient[mapConditionId.iθDynamic[i]] += out[i]
        end
    else
        # For forward sensitives via autodiff ∂G∂p is on the same scale as odeProblem.p, while 
        # S-matrix is on the same scale as θ_dynamic. To be able to handle condition specific 
        # parameters mapping to several odeProblem.p parameters the sensitivity matrix part and 
        # ∂G∂p must be treated seperately. 
        _iθDynamic = unique(mapConditionId.iθDynamic)
        gradient[_iθDynamic] .+= _adjustGradientTransformedParameters(_gradient2,
                                                                      θ_dynamic[_iθDynamic], 
                                                                      θ_indices.θ_dynamicNames[_iθDynamic], 
                                                                      θ_indices)     
        out = _adjustGradientTransformedParameters(∂G∂p[mapConditionId.iODEProblemθDynamic],
                                                   θ_dynamic[mapConditionId.iθDynamic], 
                                                   θ_indices.θ_dynamicNames[mapConditionId.iθDynamic], 
                                                   θ_indices)    

        for i in eachindex(mapConditionId.iODEProblemθDynamic)
            gradient[mapConditionId.iθDynamic[i]] += out[i]
        end
    end
end


function _adjustGradientTransformedParameters(_gradient::AbstractVector, 
                                              θ::AbstractVector,
                                              θ_names::Vector{Symbol},
                                              θ_indices::ParameterIndices)::AbstractVector
    
    out = similar(_gradient)
    @inbounds for (i, θ_name) in pairs(θ_names)
        θ_scale = θ_indices.θ_scale[θ_name]
        if θ_scale == :log10
            out[i] = log(10) * _gradient[i] * θ[i]
        elseif θ_scale == :log
            out[i] = _gradient[i] * θ[i]
        elseif θ_scale == :lin
            out[i] = _gradient[i] 
        end
    end

    return out
end