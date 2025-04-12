function _check_conditionids(petab_tables::Dict{Symbol, DataFrame}, petab_measurements::PEtabMeasurements)::Nothing
    conditions_df = petab_tables[:conditions]
    ncol(conditions_df) == 1 && return nothing
    @unpack pre_equilibration_condition_id, simulation_condition_id = petab_measurements
    conditions_df = petab_tables[:conditions]
    hybridization_df = petab_tables[:hybridization]
    measurementids = unique(vcat(pre_equilibration_condition_id, simulation_condition_id))
    for conditionid in (conditions_df[!, :conditionId] .|> Symbol)
        conditionid in measurementids && continue
        @warn "Simulation condition id $conditionid does not appear in the measurements " *
              "table. Therefore no measurement corresponds to this id."
    end
    # No variables are allowed to be double assigned in conditions and hybridization tables
    for target_id in hybridization_df.targetId
        !(target_id in names(conditions_df)) && continue
        throw(PEtab.PEtabInputError("Variable $(target_id) is assigned in both the
            hybridization and conditions table. This is not allowed."))
    end
    return nothing
end

function _check_mapping_table(petab_tables::Dict{Symbol, DataFrame}, nnmodels::Union{Nothing, Dict{Symbol, <:NNModel}}, petab_parameters::PEtabParameters, sys::ModelSystem)::Nothing
    mappings_df = petab_tables[:mapping_table]
    isempty(mappings_df) && return nothing
    isnothing(nnmodels) && return nothing

    # Sanity check assigned assignments for static neural networks
    conditions_df = petab_tables[:conditions]
    hybridization_df = petab_tables[:hybridization]
    for (netid, nnmodel) in nnmodels
        nnmodel.static == false && continue
        _check_static_net_inputs(nnmodels, mappings_df, hybridization_df, conditions_df, petab_parameters, sys, netid)
        _check_static_net_outputs(nnmodels, mappings_df, hybridization_df, petab_parameters, sys, netid)
    end
    return nothing
end

function _check_static_net_outputs(mappings_df::DataFrame, hybridization_df::DataFrame, conditions_df::DataFrame, petab_parameters::PEtabParameters, netid::Symbol)::Nothing
    input_variables = _get_net_petab_variables(mappings_df, netid, :inputs)
    for input_variable in input_variables
        input_variable in hybridization_df.targetId && continue
        input_variable in names(conditions_df) && continue
        Symbol(input_variable) in petab_parameters.parameter_id && continue
        throw(PEtab.PEtabInputError("For a static neural network, input variables in \
            assigned to in the mapping table must be assigned value by a parameter in \
            the parameter table, or in the conditions table. This does not hold for \
            $(input_variable)"))
    end
    return nothing
end

function _check_static_net_outputs(mappings_df::DataFrame, hybridization_df::DataFrame, petab_parameters::PEtabParameters, sys::ModelSystem, netid::Symbol)::Nothing
    state_ids = _get_state_ids(sys) .|> Symbol
    xids_sys = _get_xids_sys(sys)
    x_estimate = petab_parameters.parameter_id[petab_parameters.estimate]

    output_variables = _get_net_petab_variables(mappings_df, netid, :outputs)
    outputs_df = filter(row -> row.targetValue in output_variables, hybridization_df)
    for i in 1:nrow(outputs_df)
        output_variable = outputs_df.targetValue
        output_value = Symbol(outputs_df.targetId[i])

        output_value in state_ids && continue
        if !(output_value in xids_sys)
            throw(PEtab.PEtabInputError("For a static neural network, output variables in \
                the hybridization table can only assign to a specie id or to a
                non-estimated model parameter. This does not hold for $output_variable \
                assigning to $output_value, as $output_value does not fulfill these \
                criteria"))
        end
        if output_value in x_estimate
            throw(PEtab.PEtabInputError("For a static neural network, output variables in \
                the hybridization table cannot assign to model parameters that are \
                estimated. This does not hold for $output_variable assigning to \
                $output_value, as $output_value is set to be estimated in the parameters \
                table"))
        end
    end
    return nothing
end
