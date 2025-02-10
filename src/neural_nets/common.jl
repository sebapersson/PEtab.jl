function _get_f_nn_preode_x(nnpre::NNPreODE, xdynamic_mech::AbstractVector, pnn::ComponentArray, map_nn::NNPreODEMap)::AbstractVector
    x = get_tmp(nnpre.x, xdynamic_mech)
    x[1:map_nn.nxdynamic_inputs] = xdynamic_mech[map_nn.ixdynamic_mech_inputs]
    @views x[(map_nn.nxdynamic_inputs+1):end] .= pnn
    return x
end
function _get_f_nn_preode_x(nnpre::NNPreODE, xdynamic_mech::AbstractVector, map_nn::NNPreODEMap)::AbstractVector
    x = get_tmp(nnpre.x, xdynamic_mech)
    x[1:map_nn.nxdynamic_inputs] = xdynamic_mech[map_nn.ixdynamic_mech_inputs]
    return x
end

function set_ps_net!(ps::ComponentArray, netid::Symbol, nnmodel::NNModel, model::PEtabModel, petab_net_parameters::PEtabNetParameters)::Nothing
    inet = findfirst(x -> x == netid, petab_net_parameters.parameter_id)
    psfile_path = joinpath(model.paths[:dirmodel], petab_net_parameters.nominal_value[inet])
    set_ps_net!(ps, psfile_path, nnmodel.nn)
    return nothing
end
function set_ps_net!(ps::ComponentArray, netid::Symbol, model_info::ModelInfo)::Nothing
    @unpack model, petab_net_parameters = model_info
    dirmodel, nnmodel = model.paths[:dirmodel], model.nnmodels[netid]
    netindices = _get_netindices(netid, petab_net_parameters.parameter_id)
    netfile = joinpath(dirmodel, petab_net_parameters.nominal_value[1])
    @assert isfile(netfile) "Parameter values for net $netid must be a file"

    # Set parameters for entire net, then set values for specific layers
    PEtab.set_ps_net!(ps, netid, nnmodel, model, petab_net_parameters)
    length(netindices) == 1 && return nothing

    for netindex in netindices
        id = string(petab_net_parameters.parameter_id[netindex])
        value = petab_net_parameters.nominal_value[netindex]
        if value isa String
            @assert joinpath(dirmodel, value) == netfile "A separate file for a layer is not allowed"
            continue
        end

        @assert count(".", id) ≤ 2 "Only two . are allowed when specifaying network layer"
        if count(".", id) == 1
            layerid = Symbol(split(id, ".")[2])
            @views ps[layerid] .= value
        else
            layerid = Symbol(split(id, ".")[2])
            pid = Symbol(split(id, ".")[3])
            @views ps[layerid][pid] .= value
        end
    end
    return nothing
end

function _get_net_values(mapping_table::DataFrame, netid::Symbol, type::Symbol)::Vector{String}
    entity_col = string.(mapping_table[!, "modelEntityId"])
    if type == :outputs
        idf = startswith.(entity_col, "$(netid).output")
    elseif type == :inputs
        idf = startswith.(entity_col, "$(netid).input")
    end
    df = mapping_table[idf, :]
    # Sort to get inputs in order output1, output2, ...
    is = sortperm(string.(df[!, "modelEntityId"]),
                  by = x -> parse(Int, match(r"\d+$", x).match))
    return string.(df[is, "petabEntityId"])
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
    return split.(string.(mapping_table[!, "modelEntityId"]), ".") .|>
        first .|>
        string
end

function _get_netindices(netid::Symbol, netids::Vector{Symbol})::Vector{Int64}
    filtered_indices = findall(x -> startswith(x, string(netid)), string.(netids))
    return sort(filtered_indices, by = i -> count(==('.'), string(netids[i])))
end

function _get_xnames_nn(xnames::Vector{Symbol}, model_info::ModelInfo)::Vector{Symbol}
    ix_mech = _get_ixnames_mech(xnames, model_info.petab_parameters)
    return xnames[setdiff(1:length(xnames), ix_mech)]
end
