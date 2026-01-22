function _parameters_to_table(parameters::Vector)::DataFrame
    # Most validity check occurs later during table parsing
    parameters_df = DataFrame()
    for petab_parameter in parameters
        if !(petab_parameter isa Union{PEtabParameter, PEtabMLParameter})
            throw(PEtab.PEtabInputError("Input parameters to a PEtabModel must either \
                be a PEtabParameter or a PEtabMLParameter."))
        end

        row = _parse_petab_parameter(petab_parameter)
        parameters_df = DataFrames.vcat(parameters_df, row; cols = :union)
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

        row = DataFrame(observableId = observable_id,
                        observableFormula = replace(observable_formula, "(t)" => ""),
                        observableTransformation = _transformation,
                        noiseFormula = replace(noise_formula, "(t)" => ""),
                        noiseDistribution = _dist)
        append!(observables_df, row)
    end
    _check_table(observables_df, :observables_v1)
    return observables_df
end

function _conditions_to_table(conditions::Vector{PEtabCondition}, sys::ModelSystem, ml_models::MLModels)::DataFrame
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

function _measurements_to_table(measurements::DataFrame, conditions::Vector{PEtabCondition})::DataFrame
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

function _mapping_to_table(ml_models::MLModels, parameters::Vector)::DataFrame
    isempty(ml_models) && return DataFrame()
    mappings_df = DataFrame()
    for (ml_id, ml_model) in ml_models
        _inputs_df = _get_mapping_table_io(ml_model.inputs, ml_id, ml_model, :inputs)
        _outputs_df = _get_mapping_table_io(ml_model.outputs, ml_id, ml_model, :outputs)
        mappings_df = reduce(vcat, (mappings_df, _inputs_df, _outputs_df))
    end

    for parameter in parameters
        parameter isa PEtabParameter && continue

        @unpack ml_id, priors = parameter
        _parameters_df = DataFrame(
            modelEntityId = "$(ml_id).parameters",
            petabEntityId = "$(ml_id)_parameters"
        )
        for prior_id in keys(priors)
            row = DataFrame(
                modelEntityId = _get_nested_parameter_id(prior_id, ml_id; model_entity = true),
                petabEntityId = _get_nested_parameter_id(prior_id, ml_id; model_entity = false)
            )
            DataFrames.append!(_parameters_df, row)
        end
        mappings_df = vcat(mappings_df, _parameters_df)
    end


    if !isempty(mappings_df)
        _check_table(mappings_df, :mapping)
    end
    return mappings_df
end

function _get_mapping_table_io(io_arguments::Vector{Vector{Symbol}}, ml_id::Symbol, ml_model::MLModel, io_type::Symbol)::DataFrame
    mappings_df = DataFrame()
    for (i, io_argument) in pairs(io_arguments)
        _mappings_df = _get_mapping_table_io(io_argument, ml_id, ml_model, io_type; i_arg=(i-1))
        mappings_df = vcat(mappings_df, _mappings_df)
    end
    return mappings_df
end
function _get_mapping_table_io(io_argument::Vector{Symbol}, ml_id::Symbol, ml_model::MLModel, io_type::Symbol; i_arg=0)::DataFrame
    mappings_df = DataFrame()
    for (i, io_id) in pairs(io_argument)
        if (io_type == :inputs && ml_model.static) || (io_type == :outputs && !ml_model.static)
            _mappings_df = DataFrame(Dict(
                "modelEntityId" => "$(ml_id).$(io_type)[$(i_arg)][$(i-1)]",
                "petabEntityId" => "$(io_id)"))
        else
            _mappings_df = DataFrame(Dict(
                "modelEntityId" => "$(ml_id).$(io_type)[$(i_arg)][$(i-1)]",
                "petabEntityId" => "__$(ml_id)__$(io_type)$(i_arg)__$(i-1)"))
        end
        mappings_df = vcat(mappings_df, _mappings_df)
    end
    return mappings_df
end

function _hybridization_to_table(ml_models::MLModels, parameters_df::DataFrame, conditions_df::DataFrame)::DataFrame
    hybridization_df = DataFrame()
    for (ml_id, ml_model) in ml_models
        _inputs_df = _get_hybridization_table_io(ml_model.inputs, ml_id, ml_model, parameters_df, conditions_df, :inputs)
        _outputs_df = _get_hybridization_table_io(ml_model.outputs, ml_id, ml_model, parameters_df, conditions_df, :outputs)
        hybridization_df = reduce(vcat, (hybridization_df, _inputs_df, _outputs_df))
    end
    return hybridization_df
end

function _get_hybridization_table_io(io_arguments::Vector{Vector{Symbol}}, ml_id::Symbol, ml_model::MLModel, parameters_df::DataFrame, conditions_df::DataFrame, io_type::Symbol)
    hybridization_df = DataFrame()
    for (i, io_argument) in pairs(io_arguments)
        _hybridization_df = _get_hybridization_table_io(io_argument, ml_id, ml_model, parameters_df, conditions_df, io_type; i_arg=(i-1))
        hybridization_df = vcat(hybridization_df, _hybridization_df)
    end
    return hybridization_df
end
function _get_hybridization_table_io(io_argument::Vector{Symbol}, ml_id::Symbol, ml_model::MLModel, parameters_df::DataFrame, conditions_df::DataFrame, io_type::Symbol; i_arg=0)::DataFrame
    if (ml_model.static == false && io_type == :outputs)
        return DataFrame()
    end
    hybridization_df = DataFrame()
    for (i, io_id) in pairs(string.(io_argument))
        if io_type == :inputs && io_id == "_ARRAY_INPUT"
            continue
        end

        if (io_type == :inputs && ml_model.static) || io_type == :outputs
            io_id in parameters_df.parameterId && continue
            io_id in names(conditions_df) && continue
            _hybridization_df = DataFrame(
                targetId = io_id,
                targetValue = "__$(ml_id)__$(io_type)$(i_arg)__$(i-1)")
        else
            _hybridization_df = DataFrame(
                targetId = "__$(ml_id)__$(io_type)$(i_arg)__$(i-1)",
                targetValue = io_id)
        end
        hybridization_df = vcat(hybridization_df, _hybridization_df)
    end
    return hybridization_df
end

function _parse_petab_parameter(petab_parameter::PEtabParameter)::DataFrame
    @unpack parameter_id, scale, lb, ub, value, estimate, prior = petab_parameter

    parameterScale = isnothing(scale) ? "lin" : string(scale)
    if !(parameterScale in VALID_SCALES)
        throw(PEtabFormatError("Scale $parameterScale is not allowed for parameter \
            $parameter. Allowed scales are $(VALID_SCALES)"))
    end

    lowerBound = isnothing(lb) ? 1e-3 : lb
    upperBound = isnothing(ub) ? 1e3 : ub
    if lowerBound > upperBound
        throw(PEtabFormatError("Lower bound $lowerBound is larger than upper bound \
            $upperBound for paramter $parameter"))
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
function _parse_petab_parameter(petab_parameter::PEtabMLParameter)::DataFrame
    @unpack ml_id, estimate, value, prior, priors = petab_parameter

    if isnothing(value)
        nominal_value = "$(ml_id)_julia_random"
    else
        nominal_value = "$(ml_id)_julia_provided"
    end
    should_estimate = estimate == true ? 1 : 0
    rows =  DataFrame(
        parameterId = "$(ml_id)_parameters", parameterScale = "lin", lowerBound = -Inf,
        upperBound = Inf, nominalValue = nominal_value, estimate = should_estimate
    )

    if !isnothing(prior)
        rows[!, :objectivePriorType] .= "__Julia__" * string(prior)
    end

    for (id, prior) in priors
        parameter_id = _get_nested_parameter_id(id, ml_id)
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
    _ml_models = values(ml_models)
    ml_model_inputs = reduce(vcat, getfield.(_ml_models, :inputs))
    if !in(Symbol(target_id), ml_model_inputs)
        throw(PEtabInputError("Assigning an array in a PEtabCondition is only valid \
            when the target variable is an ML model input. For condition '$condition_id', \
            the target '$target_id' is not an ML model input variable."))
    end

    for ml_model in values(ml_models)
        ml_model.static == false && continue
        if ml_model.inputs isa Vector{Symbol}
            !in(Symbol(target_id), ml_model_inputs) && continue
            arg_idx = 1
        else
            !in(Symbol(target_id), reduce(vcat, ml_model.inputs)) && continue
            arg_idx = findfirst(x -> x == [Symbol(target_id)], ml_model_inputs)
        end
        ml_model.array_inputs[Symbol("$(condition_id)_$(arg_idx)")] = target_value
    end
    return "_ARRAY_INPUT"
end
function _parse_target_value(
        target_value, ::String, condition_id::String, i::Integer, ml_models::MLModels
    )::String
    _check_target_value(target_value, i, condition_id)
    return string(target_value)
end

function _get_nested_parameter_id(id, ml_id; model_entity::Bool = false)
    if count('.', id) > 1
        throw(PEtabInputError("When specifying a prior for a nested ML identifier, \
            the input must be in the format 'layerId.arrayId'. '$id' does not match \
            this format."))
    elseif count('.', id) == 0
        if model_entity == true
            parameter_id = "$(ml_id).parameters[$(id)]"
        else
            parameter_id = "$(ml_id)_parameters_$(id)"
        end
    else
        layer_id, array_id = split(id, '.')
        if model_entity == true
            parameter_id = "$(ml_id).parameters[$(layer_id)].$(array_id)"
        else
            parameter_id = "$(ml_id)_parameters_$(layer_id)_$(array_id)"
        end
    end
    return parameter_id
end
