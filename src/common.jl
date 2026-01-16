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

function split_x(x::AbstractVector, xindices::ParameterIndices, cache::PEtabODEProblemCache)
    split_x!(x, xindices, cache)
    @unpack xdynamic_mech, xobservable, xnoise, xnondynamic_mech, xnn_dict = cache
    return (
        get_tmp(xdynamic_mech, x), get_tmp(xobservable, x), get_tmp(xnoise, x),
        get_tmp(xnondynamic_mech, x), xnn_dict
    )
end

function split_x!(x::AbstractVector, xindices::ParameterIndices, cache::PEtabODEProblemCache; xdynamic_tot::Bool = false)::Nothing
    xi = xindices.xindices
    xdynamic = get_tmp(cache.xdynamic_mech, x)
    xdynamic .= @view x[xi[:dynamic_mech]]
    xobservable = get_tmp(cache.xobservable, x)
    xobservable .= @view x[xi[:observable]]
    xnoise = get_tmp(cache.xnoise, x)
    xnoise .= @view x[xi[:noise]]
    xnondynamic_mech = get_tmp(cache.xnondynamic_mech, x)
    xnondynamic_mech .= @view x[xi[:nondynamic_mech]]
    for (ml_id, xnn) in cache.xnn
        _xnn = get_tmp(xnn, x)
        _xnn .= @view x[xi[ml_id]]
        cache.xnn_dict[ml_id] = _xnn
    end
    if xdynamic_tot == true
        xdynamic_tot = get_tmp(cache.xdynamic_tot, x)
        xdynamic_tot .= @view x[xindices.xindices_dynamic[:xest_to_xdynamic]]
    end
    return nothing
end

function split_xdynamic(xdynamic::AbstractVector, xindices::ParameterIndices, cache::PEtabODEProblemCache)
    xdynamic_mech = get_tmp(cache.xdynamic_mech, xdynamic)
    xdynamic_mech .= @view xdynamic[xindices.xindices_dynamic[:xdynamic_to_mech]]
    for (ml_id, xnn) in cache.xnn
        ml_id in xindices.xids[:ml_nondynamic] && continue
        _xnn = get_tmp(xnn, xdynamic)
        _xnn .= @view xdynamic[xindices.xindices_dynamic[ml_id]]
        cache.xnn_dict[ml_id] = _xnn
    end
    return xdynamic_mech, cache.xnn_dict
end

function transform_x!(x::AbstractVector, xids::Vector{Symbol}, xindices::ParameterIndices;
                      to_xscale::Bool = false)::Nothing
    @inbounds for (i, xid) in pairs(xids)
        x[i] = transform_x(x[i], xindices.xscale[xid]; to_xscale = to_xscale)
    end
    return nothing
end

function transform_x(x::AbstractVector, xindices::ParameterIndices, whichx::Symbol,
                     cache::PEtabODEProblemCache; to_xscale::Bool = false)::AbstractVector
    if whichx === :xdynamic_mech || whichx === :xdynamic_tot
        xids = xindices.xids[:dynamic_mech]
        x_ps = get_tmp(cache.xdynamic_ps, x)
    elseif whichx === :xnoise
        xids = xindices.xids[:noise]
        x_ps = get_tmp(cache.xnoise_ps, x)
    elseif whichx === :xnondynamic_mech
        xids = xindices.xids[:nondynamic_mech]
        x_ps = get_tmp(cache.xnondynamic_mech_ps, x)
    elseif whichx === :xobservable
        xids = xindices.xids[:observable]
        x_ps = get_tmp(cache.xobservable_ps, x)
    end
    for (i, xid) in pairs(xids)
        x_ps[i] = transform_x(x[i], xindices.xscale[xid]; to_xscale = to_xscale)
    end
    # For xdynamic_tot (mechanistic + neural net parameters) it does not make sense to
    # transform the neural-net parameters, hence, in this approach only the mechanistic
    # parameters are transformed, then they are added to the total vector
    if whichx === :xdynamic_tot
        xdynamic_tot = get_tmp(cache.xdynamic_tot, x)
        xdynamic_tot[xindices.xindices_dynamic[:xdynamic_to_mech]] .= x_ps
        return xdynamic_tot
    else
        return x_ps
    end
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
    elseif scale === :log2
        return to_xscale == true ? log2(x) : exp2(x)
    end
end

function _sd(
        u::AbstractVector, t::Float64, p::AbstractVector, xnoise::T, xnondynamic_mech::T,
        xnn::Dict{Symbol, ComponentArray}, xnn_constant::Dict{Symbol, ComponentArray},
        model::PEtabModel, xnoise_maps::ObservableNoiseMap, observable_id::Symbol,
        nominal_values::Vector{Float64}
    )::Real where {T <: AbstractVector}
    if xnoise_maps.single_constant == true
        σ = xnoise_maps.constant_values[1]
    else
        σ = model.sd(
            u, t, p, xnoise, xnondynamic_mech, xnn, xnn_constant, nominal_values,
            observable_id, xnoise_maps, model.sys_observables, model.ml_models
        )
    end
    return σ
end

function _h(
        u::AbstractVector, t::Float64, p::AbstractVector, xobservable::T,
        xnondynamic_mech::T,  xnn::Dict{Symbol, ComponentArray},
        xnn_constant::Dict{Symbol, ComponentArray}, model::PEtabModel,
        xobservable_maps::ObservableNoiseMap, observable_id::Symbol,
        nominal_values::Vector{Float64},
    )::Real where {T <: AbstractVector}
    return model.h(
        u, t, p, xobservable, xnondynamic_mech, xnn, xnn_constant, nominal_values,
        observable_id, xobservable_maps, model.sys_observables, model.ml_models)
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

# TODO: Precompute only once
function _get_ixdynamic_simid(simid::Symbol, xindices::ParameterIndices; full_x::Bool = false, nn_preode::Bool = false)::Vector{Integer}
    @unpack isys_all_conditions, ix_condition = xindices.condition_maps[simid]
    if full_x == false
        ixdynamic = vcat(
            isys_all_conditions, ix_condition, xindices.xindices_dynamic[:nn_in_ode]
        )
    else
        ixdynamic = vcat(
            isys_all_conditions, ix_condition, xindices.xindices_dynamic[:nn_in_ode],
            xindices.xindices[:not_system_tot]
        )
    end
    # Include parameters that potentially appear as only input to a neural-net. These
    # parameters are by default included in xdynamic (as they gouvern model dynamics)
    if !isempty(xindices.maps_ml_preode)
        for map_ml_model in values(xindices.maps_ml_preode[simid])
            ixdynamic = vcat(ixdynamic, reduce(vcat, map_ml_model.ixdynamic_mech_inputs))
        end
    end
    if nn_preode == true || full_x == true
        ixdynamic = vcat(ixdynamic, xindices.xindices_dynamic[:nn_preode])
    end
    return unique(ixdynamic)
end

function _get_petab_tables(petab_tables::PEtabTables, table::Symbol)
    return _get_petab_tables(petab_tables, [table])
end
function _get_petab_tables(petab_tables::PEtabTables, tables::Vector{Symbol})
    return [petab_tables[table] for table in tables]
end
