function _get_indices(ids_subset::Vector{Symbol}, ids_est::Vector{Symbol})::Vector{Int32}
    return Int32[findfirst(x -> x == id, ids_est) for id in ids_subset]
end

function _get_indices_ml_model(i_start::Integer, ml_model::MLModel)::Vector{Int64}
    np = _get_n_ml_parameters(ml_model)
    return (i_start+1):(i_start + np)
end

function _get_ids_sys_order_in_xdynamic(xindices::ParameterIndices, conditions_df::DataFrame)::Vector{String}
    ids_sys = xindices.ids[:sys]
    ids_sys_in_xdynamic = filter(x -> x in ids_sys, xindices.ids[:est_to_dynamic_mech])
    # Extract sys parameters where an xdynamic via the condition table maps to a parameter
    # in the ODE
    ids_condition = filter(x -> !(x in ids_sys), xindices.ids[:est_to_dynamic_mech])
    for variable in propertynames(conditions_df)
        !(variable in ids_sys) && continue
        for xid_condition in string.(ids_condition)
            if xid_condition in conditions_df[!, variable]
                push!(ids_sys_in_xdynamic, variable)
            end
        end
    end
    unique!(ids_sys_in_xdynamic)
    return ids_sys_in_xdynamic .|> string
end
