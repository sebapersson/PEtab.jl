function _parameters_to_table(parameters::Vector, ml_models::MLModels)::DataFrame
    # Most validity check occurs later during table parsing
    parameters_df = DataFrame()
    for petab_parameter in parameters
        if !(petab_parameter isa Union{PEtabParameter, PEtabMLParameter})
            throw(PEtab.PEtabInputError("Input parameters to a PEtabModel must either \
                be a PEtabParameter or a PEtabMLParameter."))
        end

        row = _parse_petab_parameter(petab_parameter, ml_models)
        parameters_df = DataFrames.vcat(parameters_df, row; cols = :union)
    end

    # Each ML model must be assigned a PEtabMLParameter
    if !isempty(ml_models)
        ml_parameters = parameters[findall(x -> x isa PEtabMLParameter, parameters)]
        ml_ids = isempty(ml_parameters) ? Symbol[] : getfield.(ml_parameters, :ml_id)
        for ml_model in ml_models.ml_models
            ml_model.ml_id in ml_ids && continue
            throw(PEtabInputError("Each declared MLModel must have an associated \
                PEtabMLParameter. MLModel $(ml_model.ml_id) does not have one."))
        end
    end

    _check_table(parameters_df, :parameters_v1)
    return parameters_df
end

function _observables_to_table(observables::Vector{PEtabObservable})::DataFrame
    observable_ids = getfield.(observables, :observable_id)
    if observable_ids != unique(observable_ids)
        throw(PEtabFormatError("Observable ids ($(observable_ids)) are not \
            unique; each PEtabObservable must have a unique id."))
    end

    observables_df = DataFrame()

    for observable in observables
        @unpack observable_id, observable_formula, noise_formula, distribution = observable

        if distribution in [Distributions.Normal, Distributions.Laplace]
            _transformation = "lin"
            _dist = distribution == Distributions.Laplace ? "laplace" : "normal"

        elseif distribution in [Distributions.LogNormal, LogLaplace]
            _transformation = "log"
            _dist = distribution == Distributions.Laplace ? "laplace" : "normal"

        elseif distribution == Log10Normal
            _transformation = "log10"
            _dist = "normal"

        elseif distribution == Log2Normal
            _transformation = "log2"
            _dist = "normal"
        end

        row = DataFrame(
            observableId = observable_id,
            observableFormula = replace(observable_formula, "(t)" => ""),
            observableTransformation = _transformation,
            noiseFormula = replace(noise_formula, "(t)" => ""),
            noiseDistribution = _dist
        )
        append!(observables_df, row)
    end
    _check_table(observables_df, :observables_v1)
    return observables_df
end

function _conditions_to_table(
        conditions::Vector{PEtabCondition}, sys::ModelSystem, ml_models::MLModels
    )::DataFrame
    condition_ids = getfield.(conditions, :condition_id)
    if condition_ids != unique(condition_ids)
        throw(PEtabFormatError("Simulation condition ids ($(condition_ids)) are not \
            unique; each PEtabCondition must have a unique id."))
    end

    specie_ids = _get_state_ids(sys)
    conditions_df = DataFrame()

    for condition in conditions
        @unpack condition_id, target_ids, target_values = condition
        target_ids = replace.(target_ids, "(t)" => "")
        conditions_row = DataFrame(conditionId = condition_id)
        for (i, target_id) in pairs(target_ids)
            isempty(target_id) && continue
            conditions_row[!, target_id] .= _parse_target_value(
                target_values[i], target_id, condition_id, i, ml_models
            )
        end
        conditions_df = DataFrames.vcat(conditions_df, conditions_row, cols = :union)
    end

    for model_id in names(conditions_df)
        !(model_id in specie_ids) && continue

        for row_idx in 1:nrow(conditions_df)
            !ismissing(conditions_df[row_idx, model_id]) && continue
            conditions_df[!, model_id] .= string.(conditions_df[!, model_id])
            conditions_df[row_idx, model_id] = "NaN"
        end
    end

    _check_table(conditions_df, :conditions_v1)
    return conditions_df
end

function _measurements_to_table(
        measurements::DataFrame, conditions::Vector{PEtabCondition}
    )::DataFrame
    measurements_df = deepcopy(measurements)
    # Reformat column names to follow PEtab standard
    if "pre_eq_id" in names(measurements_df)
        rename!(measurements_df, "pre_eq_id" => "preequilibrationConditionId")
    end
    if "pre_equilibration_id" in names(measurements_df)
        rename!(measurements_df, "pre_equilibration_id" => "preequilibrationConditionId")
    end
    if "obs_id" in names(measurements_df)
        rename!(measurements_df, "obs_id" => "observableId")
    end
    if "observable_parameters" in names(measurements_df)
        rename!(measurements_df, "observable_parameters" => "observableParameters")
    end
    if "noise_parameters" in names(measurements_df)
        rename!(measurements_df, "noise_parameters" => "noiseParameters")
    end

    if "simulation_id" in names(measurements_df)
        rename!(measurements_df, "simulation_id" => "simulationConditionId")
    elseif !("simulationConditionId" in names(measurements_df))
        measurements_df[!, "simulationConditionId"] .= "__c0__"
    end

    measurements_df[!, :simulationStartTime] .= 0.0
    for condition in conditions
        @unpack condition_id, t0 = condition
        row_idx = findall(x -> x == condition_id, measurements_df.simulationConditionId)
        measurements_df.simulationStartTime[row_idx] .= t0

        if !all(measurements_df.time[row_idx] .≥ t0)
            throw(PEtabFormatError("Measurements for simulation condition $(condition_id) \
                contain time points before its simulation start time ($(t0)). All \
                measurement times for a condition must be ≥ the condition's start time."))
        end
    end

    # As the measurement table now follows the PEtab standard it can be checked with the
    # standard validation function
    _check_table(measurements_df, :measurements_v1)
    return measurements_df
end

function _mapping_to_table(
        ml_models::MLModels, conditions_df::DataFrame, parameters::Vector
    )::DataFrame
    isempty(ml_models) && return DataFrame()
    mappings_df = DataFrame()
    for ml_model in ml_models.ml_models
        ml_id = ml_model.ml_id
        _inputs_df = _get_mapping_table_io(
            ml_model.inputs, conditions_df, ml_id, ml_model, :inputs
        )
        _outputs_df = _get_mapping_table_io(
            ml_model.outputs, conditions_df, ml_id, ml_model, :outputs
        )
        mappings_df = reduce(vcat, (mappings_df, _inputs_df, _outputs_df))
    end

    for parameter in parameters
        parameter isa PEtabParameter && continue

        @unpack ml_id, priors = parameter
        _parameters_df = DataFrame(
            modelEntityId = "$(ml_id).parameters",
            petabEntityId = "$(ml_id)_parameters"
        )
        for prior_id in first.(priors)
            row = DataFrame(
                modelEntityId = _get_nested_parameter_id(
                    prior_id, ml_models[ml_id]; model_entity = true
                ),
                petabEntityId = _get_nested_parameter_id(
                    prior_id, ml_models[ml_id]; model_entity = false
                )
            )
            DataFrames.append!(_parameters_df, row)
        end
        mappings_df = vcat(mappings_df, _parameters_df)
    end

    # Unlike standard PEtab variables, ML-input variables do not have a default value, so
    # if it is assigned in the condition table, it needs to be assigned for all conditions
    ml_input_ids = _get_ml_model_io_petab_ids(ml_models, mappings_df)
    for condition_variable in names(conditions_df)
        !in(condition_variable, ml_input_ids) && continue
        !any(ismissing.(conditions_df[:, condition_variable])) && continue

        idx = findfirst(x -> ismissing(x), conditions_df[:, condition_variable])
        condition_id = conditions_df.conditionId[idx]
        throw(PEtabInputError("ML input variable '$(condition_variable)' is not assigned \
            for PEtabCondition '$(condition_id)'. ML input variables require a value for \
            every simulation condition (they have no default value)."))
    end
    # If one input is assigned array, all need to be
    for condition_variable in names(conditions_df)
        !in(condition_variable, ml_input_ids) && continue

        condition_values = conditions_df[:, condition_variable]
        !any(condition_values .== "array") && continue
        all(condition_values .== "array") && continue

        idx = findfirst(x -> x != "array", condition_values)
        condition_id = conditions_df.conditionId[idx]
        throw(PEtabInputError("ML input variable '$(condition_variable)' has array input \
            for some PEtab conditions but not others. If a variable uses array input it \
            must be provided for every condition. Specifically PEtabCondition \
            '$(condition_id)' has no array input."))
    end

    if !isempty(mappings_df)
        _check_table(mappings_df, :mapping)
    end
    return mappings_df
end

function _get_mapping_table_io(
        io_arguments::Vector{Vector{Symbol}}, conditions_df::DataFrame, ml_id::Symbol,
        ml_model::MLModel, io_type::Symbol
    )::DataFrame
    mappings_df = DataFrame()
    for (arg_idx, io_argument) in pairs(io_arguments)
        _mappings_df = _get_mapping_table_io(
            io_argument, conditions_df, ml_id, ml_model, io_type; arg_idx = (arg_idx - 1)
        )
        mappings_df = vcat(mappings_df, _mappings_df)
    end
    return mappings_df
end
function _get_mapping_table_io(
        io_argument::Vector{Symbol}, conditions_df::DataFrame, ml_id::Symbol,
        ml_model::MLModel, io_type::Symbol; arg_idx = 0
    )::DataFrame
    mappings_df = DataFrame()
    isempty(io_argument) && return mappings_df

    if io_type == :inputs && io_argument[1] == :_ARRAY_INPUT
        _mappings_df = DataFrame(
            modelEntityId = "$(ml_id).$(io_type)[$(arg_idx)]",
            petabEntityId = "__$(ml_id)__$(io_type)$(arg_idx)"
        )
        return _mappings_df
    end

    for (i, io_id) in pairs(io_argument)
        if (
            io_type == :inputs && ml_model.pre_initialization) || (io_type == :outputs &&
                !ml_model.pre_initialization
            )
            _mappings_df = DataFrame(
                modelEntityId = "$(ml_id).$(io_type)[$(arg_idx)][$(i - 1)]",
                petabEntityId = "$(io_id)"
            )

        else
            # Check if array input
            if !_is_array_input(io_id, conditions_df)
                _mappings_df = DataFrame(
                    modelEntityId = "$(ml_id).$(io_type)[$(arg_idx)][$(i - 1)]",
                    petabEntityId = "__$(ml_id)__$(io_type)$(arg_idx)__$(i - 1)"
                )
            else
                _mappings_df = DataFrame(
                    modelEntityId = "$(ml_id).$(io_type)[$(arg_idx)][$(i - 1)]",
                    petabEntityId = "$(io_id)"
                )
            end
        end
        mappings_df = vcat(mappings_df, _mappings_df)
    end
    return mappings_df
end

function _hybridization_to_table(
        ml_models::MLModels, parameters_df::DataFrame, conditions_df::DataFrame,
        mappings_df::DataFrame
    )::DataFrame
    hybridization_df = DataFrame()
    for ml_model in ml_models.ml_models
        ml_id = ml_model.ml_id
        _inputs_df = _get_hybridization_table_io(
            ml_model.inputs, ml_id, ml_model, parameters_df, conditions_df, :inputs
        )
        _outputs_df = _get_hybridization_table_io(
            ml_model.outputs, ml_id, ml_model, parameters_df, conditions_df, :outputs
        )
        hybridization_df = reduce(vcat, (hybridization_df, _inputs_df, _outputs_df))
    end

    # In case any input is assigned array input, a correction is needed in the mapping
    # table
    for i in 1:nrow(mappings_df)
        isempty(hybridization_df) && continue

        idx = findfirst(x -> x == mappings_df.petabEntityId[i], hybridization_df.targetId)
        isnothing(idx) && continue
        if hybridization_df.targetValue[idx] == "array"
            petab_id = replace(
                mappings_df[i, :modelEntityId], r"(\[\d+\])\[\d+\]$" => s"\1"
            )
            mappings_df[i, :modelEntityId] = petab_id
        end
    end

    # Finally, any array variable assigned in the condition table should not be removed,
    # as array assignments occur in hybridization table (and all internal mapping has
    # at this stage been done to properly deal with array input, as array inputs live in
    # MLModel
    for row_idx in 1:nrow(hybridization_df)
        hybridization_df.targetValue[row_idx] != "array" && continue
        if hybridization_df.targetId[row_idx] in names(conditions_df)
            DataFrames.select!(
                conditions_df, Not(Symbol(hybridization_df.targetId[row_idx]))
            )
        end
    end
    return hybridization_df
end

function _get_hybridization_table_io(
        io_arguments::Vector{Vector{Symbol}}, ml_id::Symbol, ml_model::MLModel,
        parameters_df::DataFrame, conditions_df::DataFrame, io_type::Symbol
    )
    hybridization_df = DataFrame()
    for (i, io_argument) in pairs(io_arguments)
        _hybridization_df = _get_hybridization_table_io(
            io_argument, ml_id, ml_model, parameters_df, conditions_df, io_type;
            arg_idx = (i - 1)
        )
        hybridization_df = vcat(hybridization_df, _hybridization_df)
    end
    return hybridization_df
end
function _get_hybridization_table_io(
        io_argument::Vector{Symbol}, ml_id::Symbol, ml_model::MLModel,
        parameters_df::DataFrame, conditions_df::DataFrame, io_type::Symbol; arg_idx = 0
    )::DataFrame

    if (ml_model.pre_initialization == false && io_type == :outputs) || isempty(io_argument)
        return DataFrame()
    end

    # Global array input
    if io_type == :inputs && io_argument[1] == :_ARRAY_INPUT
        hybridization_df = DataFrame(
            targetId = "__$(ml_id)__$(io_type)$(arg_idx)",
            targetValue = "array"
        )
        return hybridization_df
    end

    hybridization_df = DataFrame()
    for (i, io_id) in pairs(string.(io_argument))
        if (io_type == :inputs && ml_model.pre_initialization) || io_type == :outputs
            io_id in parameters_df.parameterId && continue

            if !in(io_id, names(conditions_df))
                _hybridization_df = DataFrame(
                    targetId = io_id,
                    targetValue = "__$(ml_id)__$(io_type)$(arg_idx)__$(i - 1)"
                )
            else
                io_value = conditions_df[1, io_id]
                io_value != "array" && continue
                _hybridization_df = DataFrame(
                    targetId = io_id,
                    targetValue = "array"
                )
            end

        else
            if !in(io_id, names(conditions_df))
                _hybridization_df = DataFrame(
                    targetId = "__$(ml_id)__$(io_type)$(arg_idx)__$(i - 1)",
                    targetValue = io_id
                )
            else
                _hybridization_df = DataFrame(
                    targetId = io_id,
                    targetValue = "array"
                )
            end
        end
        hybridization_df = vcat(hybridization_df, _hybridization_df)
    end
    return hybridization_df
end

function _parse_petab_parameter(petab_parameter::PEtabParameter, ::MLModels)::DataFrame
    @unpack parameter_id, scale, lb, ub, value, estimate, prior = petab_parameter

    parameterScale = isnothing(scale) ? "lin" : string(scale)
    if !(parameterScale in VALID_SCALES)
        throw(PEtabFormatError("Scale $parameterScale is not allowed for parameter \
            $parameter. Allowed scales are $(VALID_SCALES)"))
    end

    lowerBound = isnothing(lb) ? 1.0e-3 : lb
    upperBound = isnothing(ub) ? 1.0e3 : ub
    if lowerBound > upperBound
        throw(PEtabFormatError("Lower bound $lowerBound is larger than upper bound \
            $upperBound for parameter $parameter"))
    end

    nominalValue = isnothing(value) ? (lowerBound + upperBound) / 2.0 : value
    should_estimate = estimate == true ? 1 : 0
    row = DataFrame(
        parameterId = parameter_id, parameterScale = parameterScale, lowerBound = lowerBound,
        upperBound = upperBound, nominalValue = nominalValue, estimate = should_estimate
    )

    if !isnothing(prior)
        row[!, :objectivePriorType] .= "__Julia__" * string(prior)
    end
    return row
end
function _parse_petab_parameter(
        petab_parameter::PEtabMLParameter, ml_models::MLModels
    )::DataFrame
    @unpack ml_id, estimate, value, prior, priors = petab_parameter

    if isnothing(value)
        nominal_value = "$(ml_id)_julia_random"
    else
        nominal_value = "$(ml_id)_julia_provided"
    end
    should_estimate = estimate == true ? 1 : 0
    rows = DataFrame(
        parameterId = "$(ml_id)_parameters", parameterScale = "lin", lowerBound = -Inf,
        upperBound = Inf, nominalValue = nominal_value, estimate = should_estimate
    )

    if !isnothing(prior)
        rows[!, :objectivePriorType] .= "__Julia__" * string(prior)
    end

    for (id, prior) in priors
        parameter_id = _get_nested_parameter_id(id, ml_models[ml_id])
        row = DataFrame(
            parameterId = parameter_id, parameterScale = "lin", lowerBound = -Inf,
            upperBound = Inf, nominalValue = nominal_value, estimate = should_estimate,
            objectivePriorType = "__Julia__" * string(prior)
        )
        DataFrames.append!(rows, row; cols = :union)
    end
    return rows
end

function _parse_target_value(
        target_value::Array{<:Real}, target_id::String, condition_id::String, ::Integer,
        ml_models::MLModels
    )::String
    # Only allowed for ML model inputs
    _ml_models = ml_models.ml_models
    ml_model_inputs = reduce(vcat, _to_vec.(getfield.(_ml_models, :inputs)))
    if !in(Symbol(target_id), ml_model_inputs)
        throw(PEtabInputError("Assigning an array in a PEtabCondition is only valid \
            when the target variable is an ML model input. For condition '$condition_id', \
            the target '$target_id' is not an ML model input variable."))
    end

    for ml_model in ml_models.ml_models
        if ml_model.inputs isa Vector{Symbol}
            !in(Symbol(target_id), ml_model_inputs) && continue
            arg_idx = 1
        else
            !in(Symbol(target_id), reduce(vcat, ml_model.inputs)) && continue
            arg_idx = findfirst(x -> x == [Symbol(target_id)], ml_model.inputs)
        end
        ml_model.array_inputs[Symbol("__arg$(arg_idx)_$(condition_id)")] = target_value
    end
    return "array"
end
function _parse_target_value(
        target_value, ::String, condition_id::String, i::Integer, ::MLModels
    )::String
    _check_target_value(target_value, i, condition_id)
    return string(target_value)
end

function _get_nested_parameter_id(id, ml_model::MLModel; model_entity::Bool = false)
    if count('.', id) > 1
        throw(PEtabInputError("Invalid nested ML identifier format: '$id'. Expected \
            layerId' or 'layerId.arrayId'."))
    end

    @unpack ps, ml_id = ml_model
    if count('.', id) == 0
        if !haskey(ps, Symbol(id))
            throw(PEtabInputError("For setting priors, layer ID '$id' does not exist in \
                MLModel $(ml_id)."))
        end

        if model_entity == true
            parameter_id = "$(ml_id).parameters[$(id)]"
        else
            parameter_id = "$(ml_id)_parameters_$(id)"
        end
    else
        layer_id, array_id = split(id, '.')
        if !haskey(ps, Symbol(layer_id))
            throw(PEtabInputError("For setting priors, layer ID '$layer_id' does not exist \
                in MLModel $(ml_id)."))

        elseif !haskey(ps[Symbol(layer_id)], Symbol(array_id))
            throw(PEtabInputError("For priors, array id '$(array_id)' does not exist in \
                layer $(layer_id) for MLModel $(ml_id)."))
        end

        if model_entity == true
            parameter_id = "$(ml_id).parameters[$(layer_id)].$(array_id)"
        else
            parameter_id = "$(ml_id)_parameters_$(layer_id)_$(array_id)"
        end
    end
    return parameter_id
end

function _is_array_input(io_id::Symbol, conditions_df::DataFrame)::Bool
    !in(io_id, propertynames(conditions_df)) && return false
    return conditions_df[1, io_id] == "array"
end
