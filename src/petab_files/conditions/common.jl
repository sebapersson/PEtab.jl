function _get_xindices(xids_subset::Vector{Symbol}, xids_est::Vector{Symbol})::Vector{Int32}
    return Int32[findfirst(x -> x == id, xids_est) for id in xids_subset]
end

function _get_xindices_net(pid::Symbol, istart::Int64, nn)::Vector{Int64}
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

# TODO: Move to net common
function _get_nn_input_variables(inputs::Vector{Symbol}, conditions_df::DataFrame, petab_parameters::PEtabParameters, sys::ModelSystem; keep_numbers::Bool = false)::Vector{Symbol}
    state_ids = _get_state_ids(sys) .|> Symbol
    xids_sys = _get_xids_sys(sys)
    input_variables = Symbol[]
    for input in inputs
        if is_number(input)
            if keep_numbers == true
                push!(input_variables, input)
            end
            continue
        end
        if input in petab_parameters.parameter_id
            push!(input_variables, input)
            continue
        end
        if input in Iterators.flatten((state_ids, xids_sys))
            push!(input_variables, input)
            continue
        end
        if input in propertynames(conditions_df)
            for condition_value in Symbol.(conditions_df[!, input])
                _input_variables = _get_nn_input_variables([condition_value], conditions_df, petab_parameters, sys; keep_numbers = keep_numbers)
                input_variables = vcat(input_variables, _input_variables)
            end
            continue
        end
        throw(PEtabInputError("Input $input to neural-network cannot be found among ODE \
                               variables, PEtab parameters, or in the conditions table"))
    end
    return input_variables
end
