"""
    _get_ml_models_in_ode(ml_models, path_SBML, petab_tables)

Identify which neural-network models appear in the ODE right-hand-side

For this to hold, the following must hold:

1. Neural network has static = false
2. All neural network inputs and outputs appear in the hybridization table
3. All neural network outputs should assign to SBML model parameters

In case 3 does not hold, an error should be thrown as something is wrong with the PEtab
problem.
"""
function _get_ml_models_in_ode(ml_models::MLModels, path_SBML::String, petab_tables::PEtabTables)::MLModels
    out = Dict{Symbol, MLModel}()
    isempty(ml_models) && return out

    libsbml_model = SBMLImporter.SBML.readSBML(path_SBML)
    hybridization_df = petab_tables[:hybridization]
    mappings_df = petab_tables[:mapping]
    # Sanity check that columns in mapping table are correctly named
    pattern = r"(.inputs|.outputs|.parameters)"
    for io_id in string.(mappings_df[!, "modelEntityId"])
        if !occursin(pattern, io_id)
            throw(PEtabInputError("In mapping table, in modelEntityId column allowed \
                                   values are only ml_model_id.inputs..., ml_model_id.outputs... \
                                   and ml_model_id.parameters... $io_id is invalid"))
        end
    end

    for (ml_model_id, ml_model) in ml_models
        ml_model.static == true && continue

        input_variables = get_ml_model_petab_variables(mappings_df, ml_model_id, :inputs) |>
            Iterators.flatten
        if !all([x in hybridization_df.targetId for x in input_variables])
            throw(PEtab.PEtabInputError("For a static=false neural network all input \
                must be assigned value in the hybridization table. This does not hold for \
                $ml_model_id"))
        end

        output_variables = get_ml_model_petab_variables(mappings_df, ml_model_id, :outputs) |>
            Iterators.flatten
        outputs_df = filter(row -> row.targetValue in output_variables, hybridization_df)
        isempty(outputs_df) && continue
        if !all([x in keys(libsbml_model.parameters) for x in outputs_df.targetId])
            throw(PEtab.PEtabInputError("For a static=false neural network all output \
                variables in hybridization table must map to SBML model parameters. This does
                not hold for $ml_model_id"))
        end
        out[ml_model_id] = ml_model
    end
    return out
end

function _get_ml_models_in_ode_ids(ml_models::MLModels, path_SBML::String, petab_tables::PEtabTables)::Vector{Symbol}
    ml_models_in_ode = _get_ml_models_inode_ids(ml_models, path_SBML, petab_tables)
    return collect(keys(ml_models_in_ode))
end

function _get_ml_model_ids(mappings_df::DataFrame)::Vector{String}
    isempty(mappings_df) && return String[]
    return split.(string.(mappings_df[!, "modelEntityId"]), ".") .|>
        first .|>
        string
end

function _get_ml_model_indices(ml_model_id::Symbol, mapping_table_ids::Vector{String})::Vector{Int64}
    ix = findall(x -> startswith(x, string(ml_model_id)), mapping_table_ids)
    return sort(ix, by = i -> count(c -> c == '[' || c == '.', mapping_table_ids[i]))
end

function _get_xnames_ml_models(xnames::Vector{Symbol}, model_info::ModelInfo)::Vector{Symbol}
    ix_mech = _get_ixnames_mech(xnames, model_info.petab_parameters)
    return xnames[setdiff(1:length(xnames), ix_mech)]
end
