function _get_f_nn_preode_x(nnpre::NNPreODE, xdynamic_mech::AbstractVector, pnn::ComponentArray, map_nn::NNPreODEMap)::AbstractVector
    x = get_tmp(nnpre.x, xdynamic_mech)
    x[1:map_nn.nxdynamic_inputs] = xdynamic_mech[map_nn.ixdynamic_mech_inputs]
    @views x[(map_nn.nxdynamic_inputs+1):end] .= pnn
    return x
end

function _get_net_values(mapping_table::DataFrame, netid::Symbol, type::Symbol)::Vector{String}
    entity_col = string.(mapping_table[!, "petab.MODEL_ENTITY_ID"])
    if type == :outputs
        idf = startswith.(entity_col, "$(netid).output")
    elseif type == :inputs
        idf = startswith.(entity_col, "$(netid).input")
    end
    df = mapping_table[idf, :]
    # Sort to get inputs in order output1, output2, ...
    is = sortperm(string.(df[!, "petab.MODEL_ENTITY_ID"]),
                  by = x -> parse(Int, match(r"\d+$", x).match))
    return string.(df[is, "petab.PETAB_ENTITY_ID"])
end

function _get_nn_input_variables(inputs::Vector{Symbol}, netid::Symbol, nnmodel::NNModel, conditions_df::DataFrame, petab_parameters::PEtabParameters, sys::ModelSystem; keep_numbers::Bool = false)::Vector{Symbol}
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
        # When building ParameterIndices somtimes only the relative input is provided for
        # the path of a potential input file. To ease downstream processing the complete
        # path is provided for downstream processing
        if isfile(string(input))
            push!(input_variables, input)
            continue
        end
        if isfile(joinpath(nnmodel.dirdata, string(input)))
            push!(input_variables, Symbol(joinpath(nnmodel.dirdata, string(input))))
            continue
        end
        if input in propertynames(conditions_df)
            for condition_value in Symbol.(conditions_df[!, input])
                _input_variables = _get_nn_input_variables([condition_value], netid, nnmodel, conditions_df, petab_parameters, sys; keep_numbers = keep_numbers)
                input_variables = vcat(input_variables, _input_variables)
            end
            continue
        end
        throw(PEtabInputError("Input $input to neural-network cannot be found among ODE \
                               variables, PEtab parameters, or in the conditions table"))
    end
    return input_variables
end

function _get_nnmodels_inode(nnmodels::Union{Dict, Nothing})::Dict{Symbol, <:NNModel}
    out = Dict{Symbol, NNModel}()
    isnothing(nnmodels) && return out
    for (netid, nnmodel) in nnmodels
        if nnmodel.input_info[1] == "ode" && nnmodel.output_info[1] == "ode"
            out[netid] = nnmodel
        end
    end
    return out
end

function _get_n_net_parameters(nnmodels::Union{Dict{Symbol, <:NNModel}, Nothing}, xids::Vector{Symbol})::Int64
    isnothing(nnmodels) && return 0
    nparameters = 0
    for xid in xids
        !haskey(nnmodels, xid) && continue
        nparameters += _get_n_net_parameters(nnmodels[xid])
    end
    return nparameters
end

function _get_netids(mapping_table::DataFrame)::Vector{String}
    isempty(mapping_table) && return String[]
    return split.(string.(mapping_table[!, "petab.MODEL_ENTITY_ID"]), ".") .|>
        first .|>
        string
end
