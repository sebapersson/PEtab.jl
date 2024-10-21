function _get_state_ids(system)::Vector{String}
    ids = unknowns(system)
    ids = string.(ids)
    ids = replace.(ids, "(t)" => "")
    return ids
end
function _get_state_ids(system::ODEProblem)::Vector{String}
    local ids
    try
        ids = keys(system.u0) .|>
            string |>
            collect
    catch
        throw(PEtabInputError("If the model is provided as an ODEProblem then the initial \
                               values (u0) must be a struct that supports the keys \
                               function, for example, a ComponentArray or a NamedTuple. \
                               This is because PEtab.jl must know the order of the species \
                               in the ODE model"))
    end
    return ids
end

function _set_const_parameters!(model::PEtabModel,
                                parameters_info::PEtabParameters)::Nothing
    @unpack speciemap, parametermap, sys_mutated = model
    @unpack nominal_value, parameter_id = parameters_info
    state_ids = _get_state_ids(sys_mutated)
    xids_sys = first.(parametermap) .|> string
    for (i, id) in pairs(parameter_id .|> string)
        # Check if values matches either state of parameter, and adjust value in the map
        ip = findfirst(x -> x == id, xids_sys)
        is = findfirst(x -> x == id, state_ids)
        if !isnothing(ip)
            parametermap[ip] = Pair(parametermap[ip].first, nominal_value[i])
        elseif !isnothing(is)
            speciemap[is] = Pair(speciemap[is].first, nominal_value[i])
        end
    end
    return nothing
end

function split_x(θ_est::AbstractVector, xindices::ParameterIndices)
    @unpack xindices = xindices
    xdynamic = @view θ_est[xindices[:dynamic]]
    xobservable = @view θ_est[xindices[:observable]]
    xnoise = @view θ_est[xindices[:noise]]
    xnondynamic = @view θ_est[xindices[:nondynamic]]
    return xdynamic, xobservable, xnoise, xnondynamic
end

function split_x!(x::AbstractVector, xindices::ParameterIndices,
                  cache::PEtabODEProblemCache)::Nothing
    xindices = xindices.xindices
    @views cache.xdynamic .= x[xindices[:dynamic]]
    @views cache.xobservable .= x[xindices[:observable]]
    @views cache.xnoise .= x[xindices[:noise]]
    @views cache.xnondynamic .= x[xindices[:nondynamic]]
    return nothing
end

# Transform parameter from log10 scale to normal scale, or reverse transform
function transform_x!(x::AbstractVector, xids::Vector{Symbol}, xindices::ParameterIndices;
                      to_xscale::Bool = false)::Nothing
    @inbounds for (i, xid) in pairs(xids)
        x[i] = transform_x(x[i], xindices.xscale[xid]; to_xscale = to_xscale)
    end
    return nothing
end

function transform_x(x::AbstractVector, xindices::ParameterIndices, whichx::Symbol,
                     cache::PEtabODEProblemCache; to_xscale::Bool = false)::AbstractVector
    if whichx === :xdynamic
        xids = xindices.xids[:dynamic]
        x_ps = get_tmp(cache.xdynamic_ps, x)
    elseif whichx === :xnoise
        xids = xindices.xids[:noise]
        x_ps = get_tmp(cache.xnoise_ps, x)
    elseif whichx === :xnondynamic
        xids = xindices.xids[:nondynamic]
        x_ps = get_tmp(cache.xnondynamic_ps, x)
    elseif whichx === :xobservable
        xids = xindices.xids[:observable]
        x_ps = get_tmp(cache.xobservable_ps, x)
    end
    for (i, xid) in pairs(xids)
        x_ps[i] = transform_x(x[i], xindices.xscale[xid]; to_xscale = to_xscale)
    end
    return x_ps
end
function transform_x(x::T, xnames::Vector{Symbol}, xindices::ParameterIndices;
                     to_xscale::Bool = false)::T where {T <: AbstractVector}
    out = similar(x)
    isempty(x) && return out
    for (i, xname) in pairs(xnames)
        out[i] = transform_x(x[i], xindices.xscale[xname]; to_xscale = to_xscale)
    end
    return out
end
function transform_x(x::T, scale::Symbol; to_xscale::Bool = false)::T where {T <: Real}
    if scale === :lin
        return x
    elseif scale === :log10
        return to_xscale == true ? log10(x) : exp10(x)
    elseif scale === :log
        return to_xscale == true ? log(x) : exp(x)
    end
end

function transform_observable(val::T, transform::Symbol)::T where {T <: Real}
    if transform == :lin
        return val
    elseif transform == :log10
        return val > 0 ? log10(val) : Inf
    elseif transform == :log
        return val > 0 ? log(val) : Inf
    end
end

function _sd(u::AbstractVector, t::Float64, p::AbstractVector, xnoise::T, xnondynamic::T,
             petab_sd::Function, mapxnoise::ObservableNoiseMap, observable_id::Symbol,
             nominal_values::Vector{Float64})::Real where {T <: AbstractVector}
    if mapxnoise.single_constant == true
        σ = mapxnoise.constant_values[1]
    else
        σ = petab_sd(u, t, p, xnoise, xnondynamic, nominal_values, observable_id, mapxnoise)
    end
    return σ
end

function _h(u::AbstractVector, t::Float64, p::AbstractVector, xobservable::T,
            xnondynamic::T, petab_h::Function, mapxobservable::ObservableNoiseMap,
            observable_id::Symbol,
            nominal_values::Vector{Float64})::Real where {T <: AbstractVector}
    return petab_h(u, t, p, xobservable, xnondynamic, nominal_values, observable_id,
                   mapxobservable)
end

# Function to extract observable or noise parameters when computing h or σ
function get_obs_sd_parameter(x::AbstractVector, map::ObservableNoiseMap)::AbstractVector
    out = similar(x, map.nparameters)
    map.nparameters == 0 && return out
    nestimate = sum(map.estimate)
    if nestimate == map.nparameters
        out .= x[map.xindices]
    elseif nestimate == 0
        out .= map.constant_values
    else
        for i in eachindex(out)
            if map.estimate[i] == true
                out[i] = x[map.xindices[i]]
            else
                out[i] = map.constant_values[i]
            end
        end
    end
    return out
end

"""
    is_number(x)::Bool

    Check if a string or symbol x is a number (Float).
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

function _get_ixdynamic_simid(simid::Symbol, xindices::ParameterIndices;
                              full_x::Bool = false)::Vector{Integer}
    xmap_simid = xindices.maps_conidition_id[simid]
    if full_x == false
        ixdynamic = vcat(xindices.map_odeproblem.dynamic_to_sys, xmap_simid.ix_dynamic)
    else
        ixdynamic = vcat(xindices.map_odeproblem.dynamic_to_sys, xmap_simid.ix_dynamic,
                         xindices.xindices[:not_system])
    end
    return unique(ixdynamic)
end

function _get_n_net_parameters(nn::Union{Dict, Nothing}, xids::Vector{Symbol})::Int64
    isnothing(nn) && return 0
    nparameters = 0
    for xid in xids
        netid = string(xid)[3:end] |> Symbol
        !haskey(nn, netid) && continue
        nparameters += Lux.LuxCore.parameterlength(nn[netid][2])
    end
    return nparameters
end
