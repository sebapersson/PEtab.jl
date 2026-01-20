function _parameters_to_table(parameters::Vector{PEtabParameter})::DataFrame
    # Most validity check occurs later during table parsing
    parameters_df = DataFrame()
    for petab_paramter in parameters
        @unpack parameter_id, scale, lb, ub, value, estimate = petab_paramter

        parameterScale = isnothing(scale) ? "lin" : string(scale)
        if !(parameterScale in VALID_SCALES)
            throw(PEtabFormatError("Scale $parameterScale is not allowed for parameter " *
                                   "$parameter. Allowed scales are $(VALID_SCALES)"))
        end

        lowerBound = isnothing(lb) ? 1e-3 : lb
        upperBound = isnothing(ub) ? 1e3 : ub
        if lowerBound > upperBound
            throw(PEtabFormatError("Lower bound $lowerBound is larger than upper bound " *
                                   "$upperBound for paramter $parameter"))
        end

        nominalValue = isnothing(value) ? (lowerBound + upperBound) / 2.0 : value
        should_estimate = estimate == true ? 1 : 0
        row = DataFrame(parameterId = parameter_id,
                        parameterScale = parameterScale,
                        lowerBound = lowerBound,
                        upperBound = upperBound,
                        nominalValue = nominalValue,
                        estimate = should_estimate)
        append!(parameters_df, row)
    end

    if all(isnothing.(getfield.(parameters, :prior)))
        _check_table(parameters_df, :parameters_v1)
        return parameters_df
    end

    priors = fill("", length(parameters))
    initialization_priors = fill("", length(parameters))
    for (i, petab_parameter) in pairs(parameters)
        @unpack prior, sample_prior = petab_parameter
        isnothing(prior) && continue

        priors[i] = "__Julia__" * string(prior)
        if sample_prior == true
            initialization_priors[i] = priors[i]
        end
    end
    parameters_df[!, :objectivePriorType] .= priors
    parameters_df[!, :initializationPriorType] .= initialization_priors

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

function _conditions_to_table(conditions::Vector{PEtabCondition}, sys::ModelSystem)::DataFrame
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
        for i in eachindex(target_ids)
            isempty(target_ids[i]) && continue
            conditions_row[!, target_ids[i]] .= target_values[i]
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
