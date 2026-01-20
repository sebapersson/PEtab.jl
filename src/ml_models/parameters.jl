
function _set_ml_models_ps!(ml_models::MLModels, parameters)::Nothing
    for petab_parameter in parameters
        !(petab_parameter isa PEtabMLParameter) && continue
        @unpack ml_id, value = petab_parameter
        if !haskey(ml_models, ml_id)
            throw(PEtab.PEtabInputError("For neural network $(ml_id) a PEtabMLParameter \
                has been provided, but not as required a NetModel via the ml_models \
                keyword."))
        end
        isnothing(value) && continue
        ml_models[ml_id].ps .= value
    end
    return nothing
end

function set_ml_model_ps!(ps::ComponentArray, ml_id::Symbol, ml_model::MLModel, paths::Dict{Symbol, String})::Nothing
    # Case when Julia provided parameter input
    if isempty(paths)
        ps .= ml_model.ps
        return nothing
    end
    ps_path = _get_ps_path(ml_id, paths)
    set_ml_model_ps!(ps, ps_path, ml_model.model, ml_id)
    return nothing
end
function set_ml_model_ps!(ps::ComponentArray, ml_id::Symbol, ml_models, paths::Dict{Symbol, String}, petab_tables::PEtabTables)::Nothing
    # Case when Julia provided parameter input
    if isempty(paths)
        ps .= ml_models[ml_id].ps
        return nothing
    end

    # Case for PEtab standard format provided
    ml_model = ml_models[ml_id]
    petab_ml_parameters = PEtabMLParameters(petab_tables[:parameters], petab_tables[:mapping], ml_models)

    # Set parameters for entire net, then set values for specific layers
    PEtab.set_ml_model_ps!(ps, ml_id, ml_model, paths)

    netindices = _get_ml_model_indices(ml_id, petab_ml_parameters.mapping_table_id)
    length(netindices) == 1 && return nothing
    for netindex in netindices
        mapping_table_id = string(petab_ml_parameters.mapping_table_id[netindex])
        mapping_table_id == "$(ml_id).parameters" && continue

        value = petab_ml_parameters.nominal_value[netindex]
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

function _get_ml_model_pre_ode_x(ml_model_pre_simulate::MLModelPreSimulate, xdynamic_mech::AbstractVector, x_ml::ComponentArray, map_ml_model::MLModelPreSimulateMap)::AbstractVector
    x = get_tmp(ml_model_pre_simulate.x, xdynamic_mech)
    n_inputs = length(map_ml_model.ix_dynamic_mech)
    x[1:n_inputs] .= xdynamic_mech[map_ml_model.ix_dynamic_mech]
    @views x[(n_inputs+1):end] .= x_ml
    return x
end
function _get_ml_model_pre_ode_x(ml_model_pre_simulate::MLModelPreSimulate, xdynamic_mech::AbstractVector, map_ml_model::MLModelPreSimulateMap)::AbstractVector
    x = get_tmp(ml_model_pre_simulate.x, xdynamic_mech)
    n_inputs = length(map_ml_model.ix_dynamic_mech)
    x[1:n_inputs] .= xdynamic_mech[map_ml_model.ix_dynamic_mech]
    return x
end

function _get_n_ml_parameters(ml_models::MLModels, xids::Vector{Symbol})::Int64
    isnothing(ml_models) && return 0
    nparameters = 0
    for xid in xids
        !haskey(ml_models, xid) && continue
        nparameters += _get_n_ml_parameters(ml_models[xid])
    end
    return nparameters
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
