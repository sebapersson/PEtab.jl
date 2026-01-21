function _check_condition_table(petab_tables::PEtabTables, petab_measurements::PEtabMeasurements)::Nothing
    conditions_df, hybridization_df = _get_petab_tables(
        petab_tables, [:conditions, :hybridization]
    )
    ncol(conditions_df) == 1 && return nothing

    @unpack pre_equilibration_condition_id, simulation_condition_id = petab_measurements
    measurement_ids = unique(vcat(pre_equilibration_condition_id, simulation_condition_id))
    for id in (Symbol.(conditions_df[!, :conditionId]))
        id in measurement_ids && continue
        @warn "Simulation condition id $id does not appear in the measurements \
            table. Therefore no measurement corresponds to this id."
    end

    isempty(hybridization_df) && return nothing
    for target_id in hybridization_df.targetId
        !(target_id in names(conditions_df)) && continue
        throw(PEtab.PEtabInputError("Variable $(target_id) is assigned in both the \
            hybridization and conditions table. This is not allowed."))
    end
    return nothing
end

function _check_mapping_table(
        petab_tables::PEtabTables, paths::Dict{Symbol, String}, ml_models::MLModels,
        petab_parameters::PEtabParameters, sys::ModelSystem
    )::Nothing
    mappings_df = _get_petab_tables(petab_tables, :mapping)
    if isempty(mappings_df) || isempty(ml_models)
        return nothing
    end

    for (ml_id, ml_model) in ml_models
        ml_model.static == false && continue
        _check_static_net_inputs(petab_tables, paths, petab_parameters, ml_id)
        _check_static_net_outputs(petab_tables, petab_parameters, sys, ml_id)
    end
    return nothing
end

function _check_static_net_inputs(
        petab_tables::PEtabTables, paths::Dict{Symbol, String},
        petab_parameters::PEtabParameters, ml_id::Symbol
    )::Nothing
    conditions_df, hybridization_df, mappings_df, yaml_file = _get_petab_tables(
        petab_tables, [:conditions, :hybridization, :mapping, :yaml]
    )

    input_ids = Iterators.flatten(
        _get_ml_model_io_petab_ids(mappings_df, ml_id, :inputs)
    )
    for input_id in input_ids
        input_id in hybridization_df.targetId && continue
        input_id in names(conditions_df) && continue
        input_id == "_ARRAY_INPUT" && continue
        _input_isfile(input_id, yaml_file, paths) && continue
        Symbol(input_id) in petab_parameters.parameter_id && continue

        throw(PEtab.PEtabInputError("For a static neural network, input variables in \
            assigned to in the mapping table must be assigned value by a parameter in \
            the parameter table, or in the conditions table, or be assigned to a file \
            variable defined in the YAML file. This does not hold for $(input_id)"))
    end
    return nothing
end

function _check_static_net_outputs(
        petab_tables::PEtabTables, petab_parameters::PEtabParameters, sys::ModelSystem,
        ml_id::Symbol
    )::Nothing
    hybridization_df, mappings_df = _get_petab_tables(
        petab_tables, [:hybridization, :mapping]
    )
    state_ids = Symbol.(_get_state_ids(sys))
    xids_sys = _get_xids_sys(sys)

    output_ids = Iterators.flatten(
        _get_ml_model_io_petab_ids(mappings_df, ml_id, :outputs)
    )
    outputs_df = filter(row -> row.targetValue in output_ids, hybridization_df)
    for (i, output_id) in pairs(Symbol.(outputs_df.targetId))
        ml_output_id = outputs_df.targetValue[i]

        output_id in state_ids && continue
        if !(output_id in xids_sys)
            throw(PEtab.PEtabInputError("For a static neural network, output variables in \
                the hybridization table can only assign to a specie id or to a
                non-estimated model parameter. This does not hold for $ml_output_id \
                assigning to $output_id, as $output_id does not fulfill these \
                criteria"))
        end

        if _estimate_parameter(output_id, petab_parameters)
            throw(PEtab.PEtabInputError("For a static neural network, output variables in \
                the hybridization table cannot assign to model parameters that are \
                estimated. This does not hold for $ml_output_id assigning to \
                $output_id, as $output_id is set to be estimated in the parameters \
                table"))
        end
    end
    return nothing
end
