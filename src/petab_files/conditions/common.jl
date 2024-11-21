function _get_xindices(xids_subset::Vector{Symbol}, xids_est::Vector{Symbol})::Vector{Int32}
    return Int32[findfirst(x -> x == id, xids_est) for id in xids_subset]
end

function _get_xindices_net(pid::Symbol, istart::Integer, nn)::Vector{Int64}
    np = _get_n_net_parameters(nn, [pid])
    return (istart+1):(istart + np)
end

function _get_xids_sys_order_in_xdynamic(xindices::ParameterIndices, conditions_df::DataFrame)::Vector{String}
    xids_sys = xindices.xids[:sys]
    xids_sys_in_xdynamic = filter(x -> x in xids_sys, xindices.xids[:dynamic_mech])
    # Extract sys parameters where an xdynamic via the condition table maps to a parameter
    # in the ODE
    xids_condition = filter(x -> !(x in xids_sys), xindices.xids[:dynamic_mech])
    for variable in propertynames(conditions_df)
        !(variable in xids_sys) && continue
        for xid_condition in string.(xids_condition)
            if xid_condition in conditions_df[!, variable]
                push!(xids_sys_in_xdynamic, variable)
            end
        end
    end
    unique!(xids_sys_in_xdynamic)
    return xids_sys_in_xdynamic .|> string
end
