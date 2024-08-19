# Functions used by both the ODE-solvers and PeTab importer.

function _get_state_ids(system)::Vector{String}
    ids = states(system)
    ids = string.(ids)
    ids = replace.(ids, "(t)" => "")
    return ids
end

"""
    set_parameters_to_file_values!(parameter_map, state_map, parameters_info::ParamData)

Function that sets the parameter and state values in parameter_map and state_map
to those in the PeTab parameters file.

Used when setting up the PeTab cost function, and when solving the ODE-system
for the values in the parameters-file.
"""
function set_parameters_to_file_values!(parameter_map, state_map,
                                        parameters_info::ParametersInfo)::Nothing
    parameter_names = string.(parameters_info.parameter_id)
    parameter_names_str = string.([parameter_map[i].first for i in eachindex(parameter_map)])
    state_names_str = replace.(string.([state_map[i].first for i in eachindex(state_map)]),
                               "(t)" => "")
    for i in eachindex(parameter_names)
        parameter_name = parameter_names[i]
        valChangeTo = parameters_info.nominal_value[i]

        # Check for value to change to in parameter file
        i_param = findfirst(x -> x == parameter_name, parameter_names_str)
        i_state = findfirst(x -> x == parameter_name, state_names_str)

        if !isnothing(i_param)
            parameter_map[i_param] = Pair(parameter_map[i_param].first, valChangeTo)
        elseif !isnothing(i_state)
            state_map[i_state] = Pair(state_map[i_state].first, valChangeTo)
        end
    end
    return nothing
end

function splitθ(θ_est::AbstractVector, θ_indices::ParameterIndices)
    @unpack xindices = θ_indices
    θ_dynamic = @view θ_est[xindices[:dynamic]]
    θ_observable = @view θ_est[xindices[:observable]]
    θ_sd = @view θ_est[xindices[:noise]]
    θ_non_dynamic = @view θ_est[xindices[:nondynamic]]
    return θ_dynamic, θ_observable, θ_sd, θ_non_dynamic
end

function splitθ!(θ_est::AbstractVector, θ_indices::ParameterIndices, petab_ODE_cache::PEtabODEProblemCache)::Nothing
    @unpack xindices = θ_indices
    @views petab_ODE_cache.θ_dynamic .= θ_est[xindices[:dynamic]]
    @views petab_ODE_cache.θ_observable .= θ_est[xindices[:observable]]
    @views petab_ODE_cache.θ_sd .= θ_est[xindices[:noise]]
    @views petab_ODE_cache.θ_non_dynamic .= θ_est[xindices[:nondynamic]]
    return nothing
end

function computeσ(u::AbstractVector{T1},
                  t::Float64,
                  θ_dynamic::AbstractVector,
                  θ_sd::AbstractVector,
                  θ_non_dynamic::AbstractVector,
                  petab_model::PEtabModel,
                  i_measurement::Int64,
                  measurement_info::MeasurementsInfo,
                  θ_indices::ParameterIndices,
                  parameter_info::ParametersInfo)::Real where {T1 <: Real}

    # Compute associated SD-value or extract said number if it is known
    mapθ_sd = θ_indices.mapθ_sd[i_measurement]
    if mapθ_sd.single_constant == true
        σ = mapθ_sd.constant_values[1]
    else
        σ = petab_model.compute_σ(u, t, θ_sd, θ_dynamic, θ_non_dynamic, parameter_info,
                                  measurement_info.observable_id[i_measurement], mapθ_sd)
    end
    return σ
end


# Compute observation function h
function computehT(u::AbstractVector{T1},
                   t::Float64,
                   θ_dynamic::AbstractVector,
                   θ_observable::AbstractVector,
                   θ_non_dynamic::AbstractVector,
                   petab_model::PEtabModel,
                   i_measurement::Int64,
                   measurement_info::MeasurementsInfo,
                   θ_indices::ParameterIndices,
                   parameter_info::ParametersInfo)::Real where {T1 <: Real}
    mapθ_observable = θ_indices.mapθ_observable[i_measurement]
    h = petab_model.compute_h(u, t, θ_dynamic, θ_observable, θ_non_dynamic, parameter_info,
                              measurement_info.observable_id[i_measurement],
                              mapθ_observable)
    # Transform y_model is necessary
    hT = transform_measurement_or_h(h,
                                    measurement_info.measurement_transformation[i_measurement])

    return hT
end

function computeh(u::AbstractVector{T1},
                  t::Float64,
                  θ_dynamic::AbstractVector,
                  θ_observable::AbstractVector,
                  θ_non_dynamic::AbstractVector,
                  petab_model::PEtabModel,
                  i_measurement::Int64,
                  measurement_info::MeasurementsInfo,
                  θ_indices::ParameterIndices,
                  parameter_info::ParametersInfo)::Real where {T1 <: Real}
    mapθ_observable = θ_indices.mapθ_observable[i_measurement]
    h = petab_model.compute_h(u, t, θ_dynamic, θ_observable, θ_non_dynamic, parameter_info,
                              measurement_info.observable_id[i_measurement],
                              mapθ_observable)
    return h
end

"""
    transform_measurement_or_h(val::Real, transformationArr::Array{Symbol, 1})

    Transform val using either :lin (identify), :log10 and :log transforamtions.
"""
function transform_measurement_or_h(val::T, transform::Symbol)::T where {T <: Real}
    if transform == :lin
        return val
    elseif transform == :log10
        return val > 0 ? log10(val) : Inf
    elseif transform == :log
        return val > 0 ? log(val) : Inf
    else
        throw(PEtabFormatError("Only lin, log or log10 is allowed for parameter transformation"))
    end
end

# Function to extract observable or noise parameters when computing h or σ
function get_obs_sd_parameter(x::AbstractVector, map::ObservableNoiseMap)
    map.nparameters == 0 && return nothing

    nestimate = sum(map.estimate)
    if nestimate == map.nparameters
        out = x[map.xindices]
    elseif nestimate == 0
        out = map.constant_values
    else
        out = similar(x, map.nparameters)
        for i in eachindex(out)
            if map.estimate[i] == true
                out[i] = x[map.xindices[i]]
            else
                out[i] = map.constant_values[i]
            end
        end
    end
    # TODO: Should be fixable when building observable function so that consistently a
    # vector can be returned
    return length(out) == 1 ? out[1] : out
end

# Transform parameter from log10 scale to normal scale, or reverse transform
function transformθ!(θ::AbstractVector,
                     n_parameters_estimate::Vector{Symbol},
                     θ_indices::ParameterIndices;
                     reverse_transform::Bool = false)::Nothing
    @inbounds for (i, θ_name) in pairs(n_parameters_estimate)
        θ[i] = transform_θ_element(θ[i], θ_indices.θ_scale[θ_name],
                                   reverse_transform = reverse_transform)
    end
    return nothing
end

# Transform parameter from log10 scale to normal scale, or reverse transform
function transformθ(θ::T,
                    n_parameters_estimate::Vector{Symbol},
                    θ_indices::ParameterIndices;
                    reverse_transform::Bool = false)::T where {T <: AbstractVector}
    if isempty(θ)
        return similar(θ)
    else
        out = [transform_θ_element(θ[i], θ_indices.θ_scale[θ_name],
                                   reverse_transform = reverse_transform)
               for (i, θ_name) in pairs(n_parameters_estimate)]
        return out
    end
end
function transformθ(θ::AbstractVector{T},
                    n_parameters_estimate::Vector{Symbol},
                    θ_indices::ParameterIndices,
                    whichθ::Symbol,
                    petab_ODE_cache::PEtabODEProblemCache;
                    reverse_transform::Bool = false)::AbstractVector{T} where {T}
    if whichθ === :θ_dynamic
        θ_out = get_tmp(petab_ODE_cache.θ_dynamicT, θ)
    elseif whichθ === :θ_sd
        θ_out = get_tmp(petab_ODE_cache.θ_sdT, θ)
    elseif whichθ === :θ_non_dynamic
        θ_out = get_tmp(petab_ODE_cache.θ_non_dynamicT, θ)
    elseif whichθ === :θ_observable
        θ_out = get_tmp(petab_ODE_cache.θ_observableT, θ)
    end

    @inbounds for (i, θ_name) in pairs(n_parameters_estimate)
        θ_out[i] = transform_θ_element(θ[i], θ_indices.θ_scale[θ_name],
                                       reverse_transform = reverse_transform)
    end

    return θ_out
end

function transform_θ_element(θ_element::T,
                             scale::Symbol;
                             reverse_transform::Bool = false)::T where {T <: Real}
    if scale === :lin
        return θ_element
    elseif scale === :log10
        return reverse_transform == true ? log10(θ_element) : exp10(θ_element)
    elseif scale === :log
        return reverse_transform == true ? log(θ_element) : exp(θ_element)
    end
end

function change_ode_parameters!(p_ode_problem::AbstractVector,
                                u0::AbstractVector,
                                θ::AbstractVector,
                                θ_indices::ParameterIndices,
                                petab_model::PEtabModel)::Nothing
    map_ode_problem = θ_indices.map_ode_problem
    p_ode_problem[map_ode_problem.dynamic_to_sys] .= θ[map_ode_problem.sys_to_dynamic]
    petab_model.compute_u0!(u0, p_ode_problem)

    return nothing
end

function change_ode_parameters(p_ode_problem::AbstractVector,
                               θ::AbstractVector,
                               θ_indices::ParameterIndices,
                               petab_model::PEtabModel)

    # Helper function to not-inplace map parameters
    function mapParamToEst(j::Integer, mapDynParam::MapODEProblem)
        which_index = findfirst(x -> x == j, mapDynParam.dynamic_to_sys)
        return map_ode_problem.sys_to_dynamic[which_index]
    end

    map_ode_problem = θ_indices.map_ode_problem
    outp_ode_problem = [i ∈ map_ode_problem.dynamic_to_sys ?
                        θ[mapParamToEst(i, map_ode_problem)] : p_ode_problem[i]
                        for i in eachindex(p_ode_problem)]
    outu0 = petab_model.compute_u0(outp_ode_problem)

    return outp_ode_problem, outu0
end

"""
    dual_to_float(x::ForwardDiff.Dual)::Real

Via recursion convert a Dual to a Float.
"""
function dual_to_float(x::ForwardDiff.Dual)::Real
    return dual_to_float(x.value)
end
"""
    dual_to_float(x::T)::T where T<:AbstractFloat
"""
function dual_to_float(x::T)::T where {T <: AbstractFloat}
    return x
end
