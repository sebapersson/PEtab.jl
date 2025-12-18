"""
    v2_to_v1_tables(path_yaml::String, ifelse_to_callback::Bool)

Load PEtab v2 tables referenced by `path_yaml` and convert them to a v1-compatible
tables.

# Returns

- `petab_tables::Dict{Symbol, DataFrame}`: Dict holding `:parameters`, `:conditions`,
    `:observables`, and `:measurements` v1 tables.
- `petab_events::Vector{PEtabEvent}`: Events/callbacks constructed from v2 experiments
    (non-t0 triggers) so v1-style event handling can be used.

# Implementation note

The main aim with PEtab v2 problems is to rewrite them into v1 problems, as this entails
only minor changes to rest of the code-base, while maintaining v1 compatibility. In PEtab
v2, a key difference compared to v1 is that conditions may trigger at times other
than the simulation start (t ≠ t0). To preserve v1 compatibility, such non-t0 condition rows
are converted into SBML events and subsequently into Julia callbacks. Most other v2 features
can be represented using v1 tables, except non-zero simulation start times which are
captured by adding a `simulationStartTime` column to the measurements table in the v1
representation.
"""
function v2_to_v1_tables(path_yaml::String, ifelse_to_callback::Bool)
    petab_paths = PEtab._get_petab_paths(path_yaml)

    model_SBML = SBMLImporter.parse_SBML(petab_paths[:SBML], false; model_as_string = false,
        ifelse_to_callback = ifelse_to_callback, inline_assignment_rules = false)

    experiments_v2_df = _read_table(petab_paths[:experiment], :experiments_v2)
    conditions_v2_df = _read_table(petab_paths[:conditions], :conditions_v2)
    measurements_v2_df = _read_table(petab_paths[:measurements], :measurements_v2)
    observables_v2_df = _read_table(petab_paths[:observables], :observables_v2)
    parameters_v2_df = _read_table(petab_paths[:parameters], :parameters_v2)

    parameters_v1_df = _parameters_v2_v1(parameters_v2_df)
    observables_v1_df = _observables_v2_to_v1(observables_v2_df)
    conditions_v1_df, petab_events = _conditions_v2_to_v1(experiments_v2_df, conditions_v2_df, model_SBML)
    measurements_v1_df = _measurements_v2_to_v1(measurements_v2_df, experiments_v2_df, conditions_v2_df)

    petab_v1_tables = Dict(:parameters => parameters_v1_df, :conditions => conditions_v1_df, :observables => observables_v1_df, :measurements => measurements_v1_df)
    return petab_v1_tables, petab_events
end

function _parameters_v2_v1(parameters_v2_df::DataFrame)::DataFrame
    parameters_v1_df = deepcopy(parameters_v2_df)
    if !(:parameterScale in propertynames(parameters_v2_df))
        parameters_v1_df[:, :parameterScale] .= "lin"
    end

    if (:priorDistribution in propertynames(parameters_v2_df) &&
        any(.!ismissing.(parameters_v2_df.priorDistribution)))

        rename!(parameters_v1_df, Dict(:priorDistribution => "objectivePriorType"))
        rename!(parameters_v1_df, Dict(:priorParameters => "objectivePriorParameters"))

        # In v2, if priors are provided for one parameters all parameters with missing
        # priors are assigned Uniform(lb, ub)
        for row_idx in 1:nrow(parameters_v1_df)
            !ismissing(parameters_v1_df[row_idx, :objectivePriorType]) && continue
            parameters_v1_df[row_idx, :estimate] == false && continue

            lb = parameters_v1_df[row_idx, :lowerBound]
            ub = parameters_v1_df[row_idx, :upperBound]
            parameters_v1_df[row_idx, :objectivePriorType] = "uniform"
            parameters_v1_df[row_idx, :objectivePriorParameters] = "$(lb);$(ub)"
        end
    end

    return parameters_v1_df
end

function _observables_v2_to_v1(observables_v2_df::DataFrame)::DataFrame
    observables_v1_df = deepcopy(observables_v2_df)
    _observables_formulas_v2_to_v1!(observables_v1_df, observables_v2_df, :observable)
    _observables_formulas_v2_to_v1!(observables_v1_df, observables_v2_df, :noise)
    _observables_distribution_v2_to_v1!(observables_v1_df)
    _observables_into_noise_formulas!(observables_v1_df)
    return observables_v1_df
end

function _observables_formulas_v2_to_v1!(observables_v1_df::DataFrame, observables_v2_df::DataFrame, formula_kind::Symbol)::Nothing
    @assert formula_kind in [:observable, :noise]

    placeholder_col = formula_kind == :noise ? :noisePlaceholders : :observablePlaceholders
    formula_col = formula_kind == :noise ? :noiseFormula : :observableFormula
    new_param_prefix = formula_kind == :noise ? :noiseParameter : :observableParameter

    if !(placeholder_col in propertynames(observables_v2_df))
        return nothing
    end

    for row_idx in 1:nrow(observables_v2_df)
        ismissing(observables_v2_df[row_idx, placeholder_col]) && continue

        observable_id = observables_v2_df.observableId[row_idx]
        rewritten_formula = observables_v2_df[row_idx, formula_col]
        placeholder_parameters = split(observables_v2_df[row_idx, placeholder_col], ';')
        for (param_idx, parameter) in pairs(string.(placeholder_parameters))
            new_parameter_name = "$(new_param_prefix)$(param_idx)_$(observable_id)"
            rewritten_formula = SBMLImporter._replace_variable(rewritten_formula, parameter, new_parameter_name)
        end
        observables_v1_df[row_idx, formula_col] = rewritten_formula
    end
    return nothing
end

function _observables_distribution_v2_to_v1!(observables_v1_df::DataFrame)::Nothing
    # Nothing to do if the v1 table doesn't have any noise distribution column
    if !(:noiseDistribution in propertynames(observables_v1_df))
        return nothing
    end

    # Ensure transformation column exists (default to lin) for any potential edits
    if !(:observableTransformation in propertynames(observables_v1_df))
        observables_v1_df[!, :observableTransformation] .= "lin"
    end

    for row_idx in 1:nrow(observables_v1_df)
        noise_distribution = observables_v1_df.noiseDistribution[row_idx]
        noise_distribution == "normal" && continue
        @assert noise_distribution == "log-normal" "Currently only support normal and log-normal noise distributions"

        observables_v1_df[row_idx, :observableTransformation] = "log"
        observables_v1_df[row_idx, :noiseDistribution] = "normal"
    end
    return nothing
end

function _observables_into_noise_formulas!(observables_v1_df::DataFrame)::Nothing
    if !(:noiseFormula in propertynames(observables_v1_df))
        return nothing
    end

    for row_idx in 1:nrow(observables_v1_df)
        noise_formula = observables_v1_df.noiseFormula[row_idx]
        ismissing(noise_formula) && continue
        noise_formula isa Real && continue

        observable_formula = string(observables_v1_df.observableFormula[row_idx])
        observable_id = string(observables_v1_df.observableId[row_idx])
        updated_noise_formula = SBMLImporter._replace_variable(noise_formula, observable_id, "($(observable_formula))")
        observables_v1_df.noiseFormula[row_idx] = updated_noise_formula
    end
    return nothing
end

function _conditions_v2_to_v1(experiments_df::DataFrame, conditions_v2_df::DataFrame, model_SBML::SBMLImporter.ModelSBML)::Tuple{DataFrame, Vector{PEtabEvent}}
    conditions_v1_df = DataFrame()
    petab_events = PEtabEvent[]
    for experiment_id in unique(experiments_df.experimentId)
        experiment_df = filter(row -> row.experimentId == experiment_id, experiments_df)

        conditions_v1_pre_eq_row = _get_v1_condition(experiment_df, conditions_v2_df, true)
        conditions_v1_row = _get_v1_condition(experiment_df, conditions_v2_df, false)
        conditions_v1_df = DataFrames.vcat(conditions_v1_df, conditions_v1_pre_eq_row, cols = :union)
        conditions_v1_df = DataFrames.vcat(conditions_v1_df, conditions_v1_row, cols = :union)

        if !isempty(conditions_v1_pre_eq_row) && isempty(conditions_v1_row)
            simulation_condition_id = conditions_v1_pre_eq_row.conditionId[1]
        else
            simulation_condition_id = conditions_v1_row.conditionId[1]
        end
        _parse_petab_v2_events!(petab_events, experiment_df, conditions_v2_df, simulation_condition_id)
    end

    # Conditions and experiment tables are optional in v2, but not in v1. Therefore, must
    # add a dummy condition if conditions_v1_df is empty
    if isempty(conditions_v1_df)
        conditions_v1_df[!, :conditionId] = ["__c0__"]
    end

    # In PEtab v1 condition table, columns assigning species cannot be empty. They must
    # have a value or NaN
    for model_id in names(conditions_v1_df)
        model_id in ["conditionId"] && continue
        if !(haskey(model_SBML.species, model_id) || model_id in model_SBML.rate_rule_variables)
            continue
        end

        for row_idx in 1:nrow(conditions_v1_df)
            !ismissing(conditions_v1_df[row_idx, model_id]) && continue
            conditions_v1_df[!, model_id] .= string.(conditions_v1_df[!, model_id])
            conditions_v1_df[row_idx, model_id] = "NaN"
        end
    end

    return conditions_v1_df, petab_events
end

function _get_v1_condition(experiment_df::DataFrame, conditions_v2_df::DataFrame, pre_equilibration::Bool)::DataFrame
    if pre_equilibration == true
        experiment_t0_df = filter(r -> r.time == -Inf, experiment_df)
    else
        t0 = _get_t0_experiment(experiment_df)
        experiment_t0_df = filter(r -> r.time == t0, experiment_df)
    end

    if isempty(experiment_t0_df)
        return DataFrame()
    end

    # In PEtab v2 an experiment can have an empty conditionId, in this case a row with
    # only a conditionId should be returned
    if all(ismissing.(experiment_t0_df.conditionId))
        return DataFrame(conditionId = experiment_df.experimentId[1])
    end

    condition_ids = experiment_t0_df.conditionId
    experiment_id = unique(experiment_df.experimentId)[1]
    condition_v1_id = "$(experiment_id)_" * prod(condition_ids .* "_")[1:end-1]

    conditions_v1_row = DataFrame(conditionId = condition_v1_id)
    condition_experiment_df = filter(r -> r.conditionId in condition_ids, conditions_v2_df)
    for row_idx in 1:nrow(condition_experiment_df)
        @unpack targetId, targetValue = condition_experiment_df
        conditions_v1_row[:, Symbol(targetId[row_idx])] = [targetValue[row_idx]]
    end
    return conditions_v1_row
end

function _get_t0_experiment(experiment_df::DataFrame)::Float64
    experiment_t0_df = filter(r -> r.time != -Inf, experiment_df)
    if isempty(experiment_t0_df)
        return 0.0
    else
        return minimum(experiment_t0_df.time)
    end
end

function _parse_petab_v2_events!(petab_events::Vector{PEtabEvent}, experiment_df::DataFrame, conditions_v2_df::DataFrame, simulation_condition_id::String)::Nothing
    t0 = _get_t0_experiment(experiment_df)
    experiment_events_df = filter(r -> r.time ∉ [-Inf, t0], experiment_df)

    isempty(experiment_events_df) && return nothing

    # PEtab events must not change the same target. Therefore, events that trigger at the
    # same time can be merged into a single `PEtabEvent`. Merging ensures outputs are saved
    # after the final event in the group (which matters when events coincide
    # with measurement times) and thus simplifies downstream processing.
    for t in unique(experiment_events_df.time)
        experiments_event_t_df = filter(r -> r.time == t, experiment_events_df)

        trigger_time = experiments_event_t_df.time[1]
        condition = "t == $(trigger_time)"

        target_ids = String[]
        target_values = String[]
        for row_idx in 1:nrow(experiments_event_t_df)
            condition_id = experiment_events_df.conditionId[row_idx]
            condition_event_df = filter(r -> r.conditionId == condition_id, conditions_v2_df)

            target_values = vcat(target_values, string.(condition_event_df.targetValue))
            target_ids = vcat(target_ids, condition_event_df.targetId)
        end

        event = PEtabEvent(condition, target_ids, target_values, trigger_time, [Symbol(simulation_condition_id)])
        push!(petab_events, event)
    end
    return nothing
end

function _measurements_v2_to_v1(measurements_v2_df::DataFrame, experiments_v2_df::DataFrame, conditions_v2_df::DataFrame)::DataFrame
    measurements_v1_df = deepcopy(measurements_v2_df)
    measurements_v1_df[!, :simulationStartTime] .= 0.0
    rename!(measurements_v1_df, Dict(:experimentId => "simulationConditionId"))
    measurements_v1_df[!, :preequilibrationConditionId] .= ""
    allowmissing!(measurements_v1_df, :preequilibrationConditionId)

    if isempty(experiments_v2_df)
        measurements_v1_df[!, :preequilibrationConditionId] .= missing
        measurements_v1_df.simulationConditionId .= "__c0__"
        return measurements_v1_df
    end

    @unpack simulationConditionId, preequilibrationConditionId, simulationStartTime = measurements_v1_df
    for experiment_id in unique(experiments_v2_df.experimentId)
        experiment_df = filter(row -> row.experimentId == experiment_id, experiments_v2_df)
        conditions_v1_pre_eq_row = _get_v1_condition(experiment_df, conditions_v2_df, true)
        conditions_v1_row = _get_v1_condition(experiment_df, conditions_v2_df, false)

        idx_rename = findall(x -> x == experiment_id, simulationConditionId)
        if isempty(conditions_v1_pre_eq_row) && !isempty(conditions_v1_row)
            simulationConditionId[idx_rename] .= conditions_v1_row.conditionId[1]
            simulationStartTime[idx_rename] .= _get_t0_experiment(experiment_df)

        elseif !isempty(conditions_v1_pre_eq_row) && !isempty(conditions_v1_row)
            pre_equilibration_id = conditions_v1_pre_eq_row.conditionId[1]
            preequilibrationConditionId[idx_rename] .= pre_equilibration_id
            simulationConditionId[idx_rename] .= conditions_v1_row.conditionId[1]
            simulationStartTime[idx_rename] .= _get_t0_experiment(experiment_df)

        elseif !isempty(conditions_v1_pre_eq_row) && isempty(conditions_v1_row)
            pre_equilibration_id = conditions_v1_pre_eq_row.conditionId[1]
            preequilibrationConditionId[idx_rename] .= pre_equilibration_id
            simulationConditionId[idx_rename] .= pre_equilibration_id
            simulationStartTime[idx_rename] .= 0.0
        end
    end

    idx_empty = findall(isempty, preequilibrationConditionId)
    measurements_v1_df[idx_empty, :preequilibrationConditionId] .= missing

    return measurements_v1_df
end
