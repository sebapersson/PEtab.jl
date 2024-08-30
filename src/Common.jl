# Functions used by both the ODE-solvers and PeTab importer.

function _get_state_ids(system)::Vector{String}
    ids = states(system)
    ids = string.(ids)
    ids = replace.(ids, "(t)" => "")
    return ids
end

"""
    set_parameters_to_file_values!(parametermap, statemap, parameters_info::ParamData)

Function that sets the parameter and state values in parametermap and statemap
to those in the PeTab parameters file.

Used when setting up the PeTab cost function, and when solving the ODE-system
for the values in the parameters-file.
"""
function _set_constant_ode_parameters!(petab_model::PEtabModel,
                                       parameters_info::ParametersInfo)::Nothing
    # TODO: Refactor
    @unpack statemap, parametermap = petab_model
    parameter_names = string.(parameters_info.parameter_id)
    parameter_names_str = string.([parametermap[i].first for i in eachindex(parametermap)])
    state_names_str = replace.(string.([statemap[i].first for i in eachindex(statemap)]),
                               "(t)" => "")
    for i in eachindex(parameter_names)
        parameter_name = parameter_names[i]
        valChangeTo = parameters_info.nominal_value[i]

        # Check for value to change to in parameter file
        i_param = findfirst(x -> x == parameter_name, parameter_names_str)
        i_state = findfirst(x -> x == parameter_name, state_names_str)

        if !isnothing(i_param)
            parametermap[i_param] = Pair(parametermap[i_param].first, valChangeTo)
        elseif !isnothing(i_state)
            statemap[i_state] = Pair(statemap[i_state].first, valChangeTo)
        end
    end
    return nothing
end

function split_x(θ_est::AbstractVector, θ_indices::ParameterIndices)
    @unpack xindices = θ_indices
    xdynamic = @view θ_est[xindices[:dynamic]]
    xobservable = @view θ_est[xindices[:observable]]
    xnoise = @view θ_est[xindices[:noise]]
    xnondynamic = @view θ_est[xindices[:nondynamic]]
    return xdynamic, xobservable, xnoise, xnondynamic
end

function split_x!(x::AbstractVector, θ_indices::ParameterIndices, cache::PEtabODEProblemCache)::Nothing
    xindices = θ_indices.xindices
    @views cache.xdynamic .= x[xindices[:dynamic]]
    @views cache.xobservable .= x[xindices[:observable]]
    @views cache.xnoise .= x[xindices[:noise]]
    @views cache.xnondynamic .= x[xindices[:nondynamic]]
    return nothing
end

function computeσ(u::AbstractVector{T1},
                  t::Float64,
                  xdynamic::AbstractVector,
                  xnoise::AbstractVector,
                  xnondynamic::AbstractVector,
                  petab_model::PEtabModel,
                  i_measurement::Int64,
                  measurement_info::MeasurementsInfo,
                  θ_indices::ParameterIndices,
                  parameter_info::ParametersInfo)::Real where {T1 <: Real}

    # Compute associated SD-value or extract said number if it is known
    mapxnoise = θ_indices.mapxnoise[i_measurement]
    if mapxnoise.single_constant == true
        σ = mapxnoise.constant_values[1]
    else
        σ = petab_model.compute_σ(u, t, xnoise, xdynamic, xnondynamic, parameter_info,
                                  measurement_info.observable_id[i_measurement], mapxnoise)
    end
    return σ
end

# Compute observation function h
function computehT(u::AbstractVector{T1},
                   t::Float64,
                   xdynamic::AbstractVector,
                   xobservable::AbstractVector,
                   xnondynamic::AbstractVector,
                   petab_model::PEtabModel,
                   i_measurement::Int64,
                   measurement_info::MeasurementsInfo,
                   θ_indices::ParameterIndices,
                   parameter_info::ParametersInfo)::Real where {T1 <: Real}
    mapxobservable = θ_indices.mapxobservable[i_measurement]
    h = petab_model.compute_h(u, t, xdynamic, xobservable, xnondynamic, parameter_info,
                              measurement_info.observable_id[i_measurement],
                              mapxobservable)
    # Transform y_model is necessary
    hT = transform_measurement_or_h(h,
                                    measurement_info.measurement_transforms[i_measurement])

    return hT
end

function computeh(u::AbstractVector{T1},
                  t::Float64,
                  xdynamic::AbstractVector,
                  xobservable::AbstractVector,
                  xnondynamic::AbstractVector,
                  petab_model::PEtabModel,
                  i_measurement::Int64,
                  measurement_info::MeasurementsInfo,
                  θ_indices::ParameterIndices,
                  parameter_info::ParametersInfo)::Real where {T1 <: Real}
    mapxobservable = θ_indices.mapxobservable[i_measurement]
    h = petab_model.compute_h(u, t, xdynamic, xobservable, xnondynamic, parameter_info,
                              measurement_info.observable_id[i_measurement],
                              mapxobservable)
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
function transform_x!(x::AbstractVector, xids::Vector{Symbol}, θ_indices::ParameterIndices; reverse_transform::Bool = false)::Nothing
    @inbounds for (i, xid) in pairs(xids)
        x[i] = transform_θ_element(x[i], θ_indices.θ_scale[xid], reverse_transform = reverse_transform)
    end
    return nothing
end

function transform_x(x::AbstractVector, θ_indices::ParameterIndices, whichx::Symbol, cache::PEtabODEProblemCache; reverse_transform::Bool = false)::AbstractVector
    if whichx === :xdynamic
        xids = θ_indices.xids[:dynamic]
        x_ps = get_tmp(cache.xdynamic_ps, x)
    elseif whichx === :xnoise
        xids = θ_indices.xids[:noise]
        x_ps = get_tmp(cache.xnoise_ps, x)
    elseif whichx === :xnondynamic
        xids = θ_indices.xids[:nondynamic]
        x_ps = get_tmp(cache.xnondynamic_ps, x)
    elseif whichx === :xobservable
        xids = θ_indices.xids[:observable]
        x_ps = get_tmp(cache.xobservable_ps, x)
    end

    for (i, xid) in pairs(xids)
        x_ps[i] = transform_θ_element(x[i], θ_indices.θ_scale[xid],
                                      reverse_transform = reverse_transform)
    end
    return x_ps
end
function transform_x(θ::T,
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

function transform_θ_element(x::T, scale::Symbol; reverse_transform::Bool = false)::T where {T <: Real}
    if scale === :lin
        return x
    elseif scale === :log10
        return reverse_transform == true ? log10(x) : exp10(x)
    elseif scale === :log
        return reverse_transform == true ? log(x) : exp(x)
    end
end

function change_ode_parameters!(p_ode_problem::AbstractVector,
                                u0::AbstractVector,
                                θ::AbstractVector,
                                θ_indices::ParameterIndices,
                                petab_model::PEtabModel)::Nothing
    n_model_states = states(petab_model.sys_mutated) |> length
    map_ode_problem = θ_indices.map_ode_problem
    p_ode_problem[map_ode_problem.dynamic_to_sys] .= θ[map_ode_problem.sys_to_dynamic]
    u0change = @view u0[1:n_model_states]
    # TODO: Appearent I must refactor
    petab_model.compute_u0!(u0change, p_ode_problem)
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

"""
    is_number(x::String)::Bool

    Check if a string x is a number (Float) taking sciencetific notation into account.
"""
function is_number(x::Union{AbstractString, SubString{String}})::Bool
    x == "NaN" && return true
    re1 = r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)$" # Picks up scientific notation
    re2 = r"^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$"
    return (occursin(re1, x) || occursin(re2, x))
end
function is_number(x::Symbol)::Bool
    is_number(x |> string)
end

function _get_ixdynamic_simid(simid::Symbol, θ_indices::ParameterIndices;
                              full_x::Bool = false)::Vector{Integer}
    xmap_simid = θ_indices.maps_conidition_id[simid]
    if full_x == false
        ixdynamic = vcat(θ_indices.map_ode_problem.sys_to_dynamic, xmap_simid.ix_dynamic)
    else
        ixdynamic = vcat(θ_indices.map_ode_problem.sys_to_dynamic, xmap_simid.ix_dynamic,
                         θ_indices.xindices[:not_system])
    end
    return unique(ixdynamic)
end
