# Functions used by both the ODE-solvers and PeTab importer.


"""
    setParamToFileValues!(paramMap, state_map, paramData::ParamData)

    Function that sets the parameter and state values in paramMap and state_map
    to those in the PeTab parameters file.

    Used when setting up the PeTab cost function, and when solving the ODE-system
    for the values in the parameters-file.
"""
function setParamToFileValues!(paramMap, state_map, paramData::ParametersInfo)

    parameter_names = string.(paramData.parameterId)
    parameter_namesStr = string.([paramMap[i].first for i in eachindex(paramMap)])
    state_namesStr = replace.(string.([state_map[i].first for i in eachindex(state_map)]), "(t)" => "")
    for i in eachindex(parameter_names)

        parameterName = parameter_names[i]
        valChangeTo = paramData.nominalValue[i]

        # Check for value to change to in parameter file
        i_param = findfirst(x -> x == parameterName, parameter_namesStr)
        i_state = findfirst(x -> x == parameterName, state_namesStr)

        if !isnothing(i_param)
            paramMap[i_param] = Pair(paramMap[i_param].first, valChangeTo)
        elseif !isnothing(i_state)
            state_map[i_state] = Pair(state_map[i_state].first, valChangeTo)
        end
    end
end


function splitParameterVector(θ_est::AbstractVector{T},
                              θ_indices::ParameterIndices)::Tuple{AbstractVector{T}, AbstractVector{T}, AbstractVector{T}, AbstractVector{T}} where T

    θ_dynamic = @view θ_est[θ_indices.iθ_dynamic]
    θ_observable = @view θ_est[θ_indices.iθ_observable]
    θ_sd = @view θ_est[θ_indices.iθ_sd]
    θ_nonDynamic = @view θ_est[θ_indices.iθ_nonDynamic]

    return θ_dynamic, θ_observable, θ_sd, θ_nonDynamic
end


function splitParameterVector!(θ_est::AbstractVector,
                               θ_indices::ParameterIndices,
                               petabODECache::PEtabODEProblemCache)

    @views petabODECache.θ_dynamic .= θ_est[θ_indices.iθ_dynamic]
    @views petabODECache.θ_observable .= θ_est[θ_indices.iθ_observable]
    @views petabODECache.θ_sd .= θ_est[θ_indices.iθ_sd]
    @views petabODECache.θ_nonDynamic .= θ_est[θ_indices.iθ_nonDynamic]
end


function computeσ(u::AbstractVector,
                  t::Float64,
                  θ_dynamic::AbstractVector,
                  θ_sd::AbstractVector,
                  θ_nonDynamic::AbstractVector,
                  petab_model::PEtabModel,
                  iMeasurement::Int64,
                  measurementInfo::MeasurementsInfo,
                  θ_indices::ParameterIndices,
                  parameterInfo::ParametersInfo)::Real

    # Compute associated SD-value or extract said number if it is known
    mapθ_sd = θ_indices.mapθ_sd[iMeasurement]
    if mapθ_sd.isSingleConstant == true
        σ = mapθ_sd.constantValues[1]
    else
        σ = petab_model.compute_σ(u, t, θ_sd, θ_dynamic, θ_nonDynamic, parameterInfo, measurementInfo.observableId[iMeasurement], mapθ_sd)
    end

    return σ
end


# Compute observation function h
function computehTransformed(u::AbstractVector,
                             t::Float64,
                             θ_dynamic::AbstractVector,
                             θ_observable::AbstractVector,
                             θ_nonDynamic::AbstractVector,
                             petab_model::PEtabModel,
                             iMeasurement::Int64,
                             measurementInfo::MeasurementsInfo,
                             θ_indices::ParameterIndices,
                             parameterInfo::ParametersInfo)::Real

    mapθ_observable = θ_indices.mapθ_observable[iMeasurement]
    h = petab_model.compute_h(u, t, θ_dynamic, θ_observable, θ_nonDynamic, parameterInfo, measurementInfo.observableId[iMeasurement], mapθ_observable)
    # Transform yMod is necessary
    hTransformed = transformMeasurementOrH(h, measurementInfo.measurementTransformation[iMeasurement])

    return hTransformed
end


function computeh(u::AbstractVector{T},
                  t::Float64,
                  θ_dynamic::AbstractVector,
                  θ_observable::AbstractVector,
                  θ_nonDynamic::AbstractVector,
                  petab_model::PEtabModel,
                  iMeasurement::Int64,
                  measurementInfo::MeasurementsInfo,
                  θ_indices::ParameterIndices,
                  parameterInfo::ParametersInfo)::Real where T

    mapθ_observable = θ_indices.mapθ_observable[iMeasurement]
    h = petab_model.compute_h(u, t, θ_dynamic, θ_observable, θ_nonDynamic, parameterInfo, measurementInfo.observableId[iMeasurement], mapθ_observable)
    return h
end



"""
    transformMeasurementOrH(val::Real, transformationArr::Array{Symbol, 1})

    Transform val using either :lin (identify), :log10 and :log transforamtions.
"""
function transformMeasurementOrH(val::T, transform::Symbol)::T where T
    if transform == :lin
        return val
    elseif transform == :log10
        return val > 0 ? log10(val) : Inf
    elseif transform == :log
        return val > 0 ? log(val) : Inf
    else
        println("Error : $transform is not an allowed transformation")
        println("Only :lin, :log10 and :log are supported.")
    end
end


# Function to extract observable or noise parameters when computing h or σ
function getObsOrSdParam(θ::AbstractVector, parameter_map::θObsOrSdParameterMap)

    # Helper function to map SD or obs-parameters in non-mutating way
    function map1Tmp(iVal)
        whichI = sum(parameter_map.shouldEstimate[1:iVal])
        return parameter_map.indexInθ[whichI]
    end
    function map2Tmp(iVal)
        whichI = sum(.!parameter_map.shouldEstimate[1:iVal])
        return whichI
    end

    # In case of no SD/observable parameter exit function
    if parameter_map.nParameters == 0
        return
    end

    # In case of single-value return do not have to return an array and think about type
    if parameter_map.nParameters == 1
        if parameter_map.shouldEstimate[1] == true
            return θ[parameter_map.indexInθ][1]
        else
            return parameter_map.constantValues[1]
        end
    end

    nParametersToEstimte = sum(parameter_map.shouldEstimate)
    if nParametersToEstimte == parameter_map.nParameters
        return θ[parameter_map.indexInθ]

    elseif nParametersToEstimte == 0
        return parameter_map.constantValues

    # Computaionally most demanding case. Here a subset of the parameters
    # are to be estimated. This code must be non-mutating to support Zygote which
    # negatively affects performance
    elseif nParametersToEstimte > 0
        _values = [parameter_map.shouldEstimate[i] == true ? θ[map1Tmp(i)] : 0.0 for i in 1:parameter_map.nParameters]
        values = [parameter_map.shouldEstimate[i] == false ? parameter_map.constantValues[map2Tmp(i)] : _values[i] for i in 1:parameter_map.nParameters]
        return values
    end
end


# Transform parameter from log10 scale to normal scale, or reverse transform
function transformθ!(θ::AbstractVector,
                     n_parameters_estimate::Vector{Symbol},
                     θ_indices::ParameterIndices;
                     reverseTransform::Bool=false)

    @inbounds for (i, θ_name) in pairs(n_parameters_estimate)
        θ[i] = transformθElement(θ[i], θ_indices.θ_scale[θ_name], reverseTransform=reverseTransform)
    end
end

# Transform parameter from log10 scale to normal scale, or reverse transform
function transformθ(θ::AbstractVector,
                    n_parameters_estimate::Vector{Symbol},
                    θ_indices::ParameterIndices;
                    reverseTransform::Bool=false)::AbstractVector

    if isempty(θ)
        return similar(θ)
    else
        out = [transformθElement(θ[i], θ_indices.θ_scale[θ_name], reverseTransform=reverseTransform) for (i, θ_name) in pairs(n_parameters_estimate)]
        return out
    end
end
function transformθ(θ::AbstractVector{T},
                    n_parameters_estimate::Vector{Symbol},
                    θ_indices::ParameterIndices,
                    whichθ::Symbol,
                    petabODECache::PEtabODEProblemCache;
                    reverseTransform::Bool=false)::AbstractVector{T} where T

    if whichθ === :θ_dynamic
        θ_out = get_tmp(petabODECache.θ_dynamicT, θ)
    elseif whichθ === :θ_sd
        θ_out = get_tmp(petabODECache.θ_sdT, θ)
    elseif whichθ === :θ_nonDynamic
        θ_out = get_tmp(petabODECache.θ_nonDynamicT, θ)
    elseif whichθ === :θ_observable
        θ_out = get_tmp(petabODECache.θ_observableT, θ)
    end

    @inbounds for (i, θ_name) in pairs(n_parameters_estimate)
        θ_out[i] = transformθElement(θ[i], θ_indices.θ_scale[θ_name], reverseTransform=reverseTransform)
    end

    return θ_out
end


function transformθElement(θ_element,
                           scale::Symbol;
                           reverseTransform::Bool=false)::Real

    if scale === :lin
        return θ_element
    elseif scale === :log10
        return reverseTransform == true ? log10(θ_element) : exp10(θ_element)
    elseif scale === :log
        return reverseTransform == true ? log(θ_element) : exp(θ_element)
    end
end


function changeODEProblemParameters!(pODEProblem::AbstractVector,
                                     u0::AbstractVector,
                                     θ::AbstractVector,
                                     θ_indices::ParameterIndices,
                                     petab_model::PEtabModel)

    mapODEProblem = θ_indices.mapODEProblem
    pODEProblem[mapODEProblem.iODEProblemθDynamic] .= θ[mapODEProblem.iθDynamic]
    petab_model.compute_u0!(u0, pODEProblem)

    return nothing
end


function changeODEProblemParameters(pODEProblem::AbstractVector,
                                    θ::AbstractVector,
                                    θ_indices::ParameterIndices,
                                    petab_model::PEtabModel)

    # Helper function to not-inplace map parameters
    function mapParamToEst(j::Integer, mapDynParam::MapODEProblem)
        whichIndex = findfirst(x -> x == j, mapDynParam.iODEProblemθDynamic)
        return mapODEProblem.iθDynamic[whichIndex]
    end

    mapODEProblem = θ_indices.mapODEProblem
    outpODEProblem = [i ∈ mapODEProblem.iODEProblemθDynamic ? θ[mapParamToEst(i, mapODEProblem)] : pODEProblem[i] for i in eachindex(pODEProblem)]
    outu0 = petab_model.compute_u0(outpODEProblem)

    return outpODEProblem, outu0
end


"""
    dualToFloat(x::ForwardDiff.Dual)::Real

    Via recursion convert a Dual to a Float.
"""
function dualToFloat(x::ForwardDiff.Dual)::Real
    return dualToFloat(x.value)
end
"""
    dualToFloat(x::AbstractFloat)::AbstractFloat
"""
function dualToFloat(x::AbstractFloat)::AbstractFloat
    return x
end
