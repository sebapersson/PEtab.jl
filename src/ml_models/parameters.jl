
function _set_ml_models_ps!(ml_models::MLModels, parameters)::Nothing
    for petab_parameter in parameters
        !(petab_parameter isa PEtabMLParameter) && continue
        @unpack ml_model_id, value = petab_parameter
        if !haskey(ml_models, ml_model_id)
            throw(PEtab.PEtabInputError("For neural network $(ml_model_id) a PEtabMLParameter \
                has been provided, but not as required a NetModel via the ml_models \
                keyword."))
        end
        isnothing(value) && continue
        ml_models[ml_model_id].ps .= value
    end
    return nothing
end

function set_ml_model_ps!(ps::ComponentArray, ml_model_id::Symbol, ml_model::MLModel, paths::Dict{Symbol, String})::Nothing
    # Case when Julia provided parameter input
    if isempty(paths)
        ps .= ml_model.ps
        return nothing
    end
    ps_path = _get_ps_path(ml_model_id, paths)
    set_ml_model_ps!(ps, ps_path, ml_model.model, ml_model_id)
    return nothing
end
function set_ml_model_ps!(ps::ComponentArray, ml_model_id::Symbol, ml_models, paths::Dict{Symbol, String}, petab_tables::PEtabTables)::Nothing
    # Case when Julia provided parameter input
    if isempty(paths)
        ps .= ml_models[ml_model_id].ps
        return nothing
    end

    # Case for PEtab standard format provided
    ml_model = ml_models[ml_model_id]
    petab_net_parameters = PEtabMLParameters(petab_tables[:parameters], petab_tables[:mapping], ml_models)
    netindices = _get_ml_model_indices(ml_model_id, petab_net_parameters.mapping_table_id)

    # Set parameters for entire net, then set values for specific layers
    PEtab.set_ml_model_ps!(ps, ml_model_id, ml_model, paths)
    length(netindices) == 1 && return nothing
    for netindex in netindices
        mapping_table_id = string(petab_net_parameters.mapping_table_id[netindex])
        mapping_table_id == "$(ml_model_id).parameters" && continue

        value = petab_net_parameters.nominal_value[netindex]
        isempty(value) && continue

        @assert count(".", mapping_table_id) ≤ 2 "Only two . are allowed when specifying network layer"
        if count('[', mapping_table_id) == 1 && count('.', mapping_table_id) == 1
            layerid = match(r"parameters\[(\w+)\]", mapping_table_id).captures[1] |>
                Symbol
            @views ps[layerid] .= value
        else
            layerid = match(r"parameters\[(\w+)\]", mapping_table_id).captures[1] |>
                Symbol
            arrayid = Symbol(split(mapping_table_id, ".")[3])
            @views ps[layerid][arrayid] .= value
        end
    end
    return nothing
end

function _get_ml_model_pre_ode_x(nnpre::MLModelPreODE, xdynamic_mech::AbstractVector, pnn::ComponentArray, map_ml_model::MLModelPreODEMap)::AbstractVector
    x = get_tmp(nnpre.x, xdynamic_mech)
    x[1:map_ml_model.nxdynamic_inputs] = xdynamic_mech[map_ml_model.ixdynamic_mech_inputs]
    @views x[(map_ml_model.nxdynamic_inputs+1):end] .= pnn
    return x
end
function _get_ml_model_pre_ode_x(nnpre::MLModelPreODE, xdynamic_mech::AbstractVector, map_ml_model::MLModelPreODEMap)::AbstractVector
    x = get_tmp(nnpre.x, xdynamic_mech)
    x[1:map_ml_model.nxdynamic_inputs] = xdynamic_mech[map_ml_model.ixdynamic_mech_inputs]
    return x
end

function _get_n_ml_model_parameters(ml_models::Union{MLModels, Nothing}, xids::Vector{Symbol})::Int64
    isnothing(ml_models) && return 0
    nparameters = 0
    for xid in xids
        !haskey(ml_models, xid) && continue
        nparameters += _get_n_ml_model_parameters(ml_models[xid])
    end
    return nparameters
end
