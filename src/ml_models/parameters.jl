
function _set_ml_models_ps!(ml_models::MLModels, parameters::Vector)::Nothing
    for petab_parameter in parameters
        !(petab_parameter isa PEtabMLParameter) && continue

        @unpack value, ml_id = petab_parameter
        isnothing(value) && continue
        ml_models[ml_id].ps .= value
    end
    return nothing
end
function _set_ml_model_ps!(
        ps::ComponentArray, ml_model::MLModel, paths::Dict{Symbol, String}
    )::Nothing
    # Case when Julia provided parameter input
    if isempty(paths)
        ps .= ml_model.ps
        return nothing
    end
    ps_path = _get_ps_path(ml_model.ml_id, paths)
    _set_ml_model_ps!(ps, ps_path, ml_model.lux_model, ml_model.ml_id)
    return nothing
end
function _set_ml_model_ps!(
        ps::ComponentArray, ml_id::Symbol, ml_models, paths::Dict{Symbol, String},
        petab_tables::PEtabTables
    )::Nothing
    # Case when Julia provided parameter input
    if isempty(paths)
        ps .= ml_models[ml_id].ps
        return nothing
    end

    # Case for PEtab standard format provided
    ml_model = ml_models[ml_id]
    petab_ml_parameters = PEtabMLParameters(petab_tables, ml_models)

    # Set parameters for entire net, then set values for specific layers
    _set_ml_model_ps!(ps, ml_model, paths)

    i_parameters = _get_ml_model_indices(ml_id, petab_ml_parameters.mapping_table_id)
    length(i_parameters) == 1 && return nothing

    for ix in i_parameters
        mapping_table_id = string(petab_ml_parameters.mapping_table_id[ix])
        mapping_table_id == "$(ml_id).parameters" && continue

        value = petab_ml_parameters.nominal_value[ix]
        isempty(value) && continue

        layer_id = _get_layer_id(mapping_table_id)
        array_id = _get_array_id(mapping_table_id)
        if isempty(array_id)
            @views ps[Symbol(layer_id)] .= value
        else
            @views ps[Symbol(layer_id)][Symbol(array_id)] .= value
        end
    end
    return nothing
end

function _get_ml_model_pre_ode_x(
        ml_model_pre_simulate::MLModelPreSimulate, xdynamic_mech::AbstractVector,
        x_ml::ComponentArray, map_ml_model::MLModelPreSimulateMap
    )::AbstractVector
    x = get_tmp(ml_model_pre_simulate.x, xdynamic_mech)
    n_inputs = length(map_ml_model.ix_dynamic_mech)
    x[1:n_inputs] .= xdynamic_mech[map_ml_model.ix_dynamic_mech]
    @views x[(n_inputs+1):end] .= x_ml
    return x
end
function _get_ml_model_pre_ode_x(ml_model_pre_simulate
        ::MLModelPreSimulate, xdynamic_mech::AbstractVector,
        map_ml_model::MLModelPreSimulateMap
    )::AbstractVector
    x = get_tmp(ml_model_pre_simulate.x, xdynamic_mech)
    n_inputs = length(map_ml_model.ix_dynamic_mech)
    @views x[1:n_inputs] .= xdynamic_mech[map_ml_model.ix_dynamic_mech]
    return x
end

function _get_layer_id(s::AbstractString)::String
    m = match(r"\.parameters\[(?<layer>[^\]]+)\](?:\.(?<arr>[^.]+))?$", s)
    m === nothing && throw(ArgumentError("Invalid format (expected ...parameters[layer].array or ...parameters[layer]): $s"))
    return m["layer"]
end

function _get_array_id(s::AbstractString)::String
    m = match(r"\.parameters\[(?<layer>[^\]]+)\](?:\.(?<arr>[^.]+))?$", s)
    m === nothing && throw(ArgumentError("Invalid format (expected ...parameters[layer].array or ...parameters[layer]): $s"))
    return something(m["arr"], "")
end
