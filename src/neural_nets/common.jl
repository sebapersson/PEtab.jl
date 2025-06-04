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

function set_ps_net!(ps::ComponentArray, netid::Symbol, nnmodel::NNModel, paths::Dict{Symbol, String}, petab_net_parameters::PEtabNetParameters)::Nothing
    netindices = _get_netindices(netid, petab_net_parameters.mapping_table_id)
    ps_path = _get_ps_path(netid, paths, petab_net_parameters.nominal_value[netindices[1]])
    set_ps_net!(ps, ps_path, nnmodel.nn)
    return nothing
end
function set_ps_net!(ps::ComponentArray, netid::Symbol, nnmodels, paths::Dict{Symbol, String}, petab_tables::PEtabTables)::Nothing
    # Case when Julia provided parameter input
    if isempty(paths)
        ps .= nnmodels[netid].ps
        return nothing
    end

    # Case for PEtab standard format provided
    nnmodel = nnmodels[netid]
    petab_net_parameters = PEtabNetParameters(petab_tables[:parameters], petab_tables[:mapping], nnmodels)
    netindices = _get_netindices(netid, petab_net_parameters.mapping_table_id)
    ps_path = _get_ps_path(netid, paths, petab_net_parameters.nominal_value[netindices[1]])
    @assert isfile(ps_path) "Parameter values for net $netid must be a file"

    # Set parameters for entire net, then set values for specific layers
    PEtab.set_ps_net!(ps, netid, nnmodel, paths, petab_net_parameters)
    length(netindices) == 1 && return nothing
    for netindex in netindices
        mapping_table_id = string(petab_net_parameters.mapping_table_id[netindex])
        value = petab_net_parameters.nominal_value[netindex]
        if value isa String
            _path = _get_ps_path(netid, paths, value)
            @assert _path == ps_path "A separate file for a layer is not allowed"
            continue
        end

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

function _get_ps_path(netid::Symbol, paths::Dict{Symbol, String}, nominal_value::String)
    yaml_file = YAML.load_file(paths[:yaml])
    array_files = yaml_file["extensions"]["sciml"]["array_files"]
    if !haskey(array_files, nominal_value)
        throw(PEtab.PEtabInputError("For neural network $netid the parameter file \
            $(nominal_value) has not been defined in the YAML problem file under \
            array_files"))
    end
    return joinpath(paths[:dirmodel], array_files[nominal_value]["location"])
end

function _get_net_petab_variables(mappings_df::DataFrame, netid::Symbol, type::Symbol)::Vector{String}
    entity_col = string.(mappings_df[!, "modelEntityId"])
    if type == :outputs
        idf = startswith.(entity_col, "$(netid).outputs")
    elseif type == :inputs
        idf = startswith.(entity_col, "$(netid).inputs")
    elseif type == :parameters
        idf = startswith.(entity_col, "$(netid).parameters")
        return mappings_df[idf, :petabEntityId]
    end
    df = mappings_df[idf, :]
    # Sort to get inputs in order output1, output2, ...
    is = sortperm(string.(df[!, "modelEntityId"]),
                  by = x -> parse(Int, match(r".*\[(\d+)\]$", x).captures[1]))
    return string.(df[is, "petabEntityId"])
end

function _get_net_input_values(input_variables::Vector{Symbol}, netid::Symbol, nnmodel::NNModel, conditions_df::DataFrame, petab_tables::PEtabTables, petab_parameters::PEtabParameters, sys::ModelSystem; keep_numbers::Bool = false)::Vector{Symbol}
    input_values = Symbol[]
    hybridization_df = petab_tables[:hybridization]
    for input_variable in input_variables
        # This can be triggered via recursion (condition table can have numbers)
        if is_number(input_variable)
            if keep_numbers == true
                push!(input_values, input_variable)
            end
            continue
        end

        if input_variable in petab_parameters.parameter_id
            push!(input_values, input_variable)
            continue
        end

        if input_variable in Symbol.(hybridization_df.targetId)
            ix = findfirst(x -> x == input_variable, Symbol.(hybridization_df.targetId))
            push!(input_values, Symbol.(hybridization_df.targetValue[ix]))
            continue
        end

        # When input is assigned via the conditions table. Recursion needed to find the
        # the potential parameter assigning the input
        if input_variable in propertynames(conditions_df)
            for condition_value in Symbol.(conditions_df[!, input_variable])
                _input_values = _get_net_input_values([condition_value], netid, nnmodel, conditions_df, petab_tables, petab_parameters, sys; keep_numbers = keep_numbers)
                input_values = vcat(input_values, _input_values)
            end
            continue
        end

        # If the input variable is a file, the complete path is added here, which simplifies
        # downstream processing
        if _input_isfile(input_variable, petab_tables[:yaml])
            path = _get_input_path(input_variable, petab_tables[:yaml], nnmodel.dirdata)
            push!(input_values, path)
            continue
        end

        throw(PEtabInputError("Input $(input_variable) to neural-network cannot be found \
            among ODE variables, PEtab parameters, array files or in the conditions table"))
    end
    return input_values
end

"""
    _get_nnmodels_in_ode(nnmodels, path_SBML, petab_tables)

Identify which neural-network models appear in the ODE right-hand-side

For this to hold, the following must hold:

1. Neural network has static = false
2. All neural network inputs and outputs appear in the hybridization table
3. All neural network outputs should assign to SBML model parameters

In case 3 does not hold, an error should be thrown as something is wrong with the PEtab
problem.
"""
function _get_nnmodels_in_ode(nnmodels::Dict{Symbol, <:NNModel}, path_SBML::String, petab_tables::PEtabTables)::Dict{Symbol, <:NNModel}
    out = Dict{Symbol, NNModel}()
    isempty(nnmodels) && return out

    libsbml_model = SBMLImporter.SBML.readSBML(path_SBML)
    hybridization_df = petab_tables[:hybridization]
    mappings_df = petab_tables[:mapping]
    # First sanity check mapping table column names
    # Sanity check that columns in mapping table are correctly named
    pattern = r"(.inputs|.outputs|.parameters)"
    for io_id in string.(mappings_df[!, "modelEntityId"])
        if !occursin(pattern, io_id)
            throw(PEtabInputError("In mapping table, in modelEntityId column allowed \
                                   values are only netid.inputs..., netid.outputs... \
                                   and netid.parameters... $io_id is invalid"))
        end
    end

    for (netid, nnmodel) in nnmodels
        nnmodel.static == true && continue

        input_variables = _get_net_petab_variables(mappings_df, netid, :inputs)
        if !all([x in hybridization_df.targetId for x in input_variables])
            throw(PEtab.PEtabInputError("For a static=false neural network all input \
                must be assigned value in the hybridization table. This does not hold for \
                $netid"))
        end

        output_variables = _get_net_petab_variables(mappings_df, netid, :outputs)
        outputs_df = filter(row -> row.targetValue in output_variables, hybridization_df)
        isempty(outputs_df) && continue
        if !all([x in keys(libsbml_model.parameters) for x in outputs_df.targetId])
            throw(PEtab.PEtabInputError("For a static=false neural network all output \
                variables in hybridization table must map to SBML model parameters. This does
                not hold for $netid"))
        end
        out[netid] = nnmodel
    end
    return out
end

function _get_nnmodels_in_ode_ids(nnmodels::Dict{Symbol, <:NNModel}, path_SBML::String, petab_tables::PEtabTables)::Vector{Symbol}
    nnmodels_in_ode = _get_nnmodels_inode_ids(nnmodels, path_SBML, petab_tables)
    return collect(keys(nnmodels_in_ode))
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

function _get_netids(mappings_df::DataFrame)::Vector{String}
    isempty(mappings_df) && return String[]
    return split.(string.(mappings_df[!, "modelEntityId"]), ".") .|>
        first .|>
        string
end

function _get_netindices(netid::Symbol, mapping_table_ids::Vector{String})::Vector{Int64}
    ix = findall(x -> startswith(x, string(netid)), mapping_table_ids)
    return sort(ix, by = i -> count(c -> c == '[' || c == '.', mapping_table_ids[i]))
end

function _get_xnames_nn(xnames::Vector{Symbol}, model_info::ModelInfo)::Vector{Symbol}
    ix_mech = _get_ixnames_mech(xnames, model_info.petab_parameters)
    return xnames[setdiff(1:length(xnames), ix_mech)]
end

function _input_isfile(input_variable::Union{String, Symbol}, yaml_file::Dict)::Bool
    yaml_file_extensions = yaml_file["extensions"]["sciml"]
    !haskey(yaml_file_extensions, "array_files") && return false
    return haskey(yaml_file_extensions["array_files"], string(input_variable))
end

function _get_input_path(input_variable::Union{String, Symbol}, yaml_file::Dict, dir::String)::Symbol
    _input_variable = string(input_variable)
    filename = yaml_file["extensions"]["sciml"]["array_files"][_input_variable]["location"]
    path = joinpath(dir, filename)
    if !isfile(path)
        throw(PEtab.PEtabInputError("$path is not a valid file path to the file input \
            for PEtab neural network input variable $(input_variable)"))
    end
    return Symbol(path)
end

function _set_nn_parameters!(nnmodels::Dict{Symbol, <:NNModel}, parameters)::Nothing
    for petab_parameter in parameters
        !(petab_parameter isa PEtabNetParameter) && continue
        @unpack netid, value = petab_parameter
        if !haskey(nnmodels, netid)
            throw(PEtab.PEtabInputError("For neural network $(netid) a PEtabNetParameter \
                has been provided, but not as required a NetModel via the net_models \
                keyword."))
        end
        isnothing(value) && continue
        nnmodels[netid].ps .= value
    end
    return nothing
end
