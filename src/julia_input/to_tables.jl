function _parameters_to_table(parameters::Vector{PEtabParameter})::DataFrame
    # Most validity check occurs later during table parsing
    parameters_df = DataFrame()
    for petab_paramter in parameters
        @unpack parameter, scale, lb, ub, value, estimate = petab_paramter

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
        row = DataFrame(parameterId = parameter |> string,
                        parameterScale = parameterScale,
                        lowerBound = lowerBound,
                        upperBound = upperBound,
                        nominalValue = nominalValue,
                        estimate = should_estimate)
        append!(parameters_df, row)
    end
    # Add potential prior columns
    if !all(isnothing.(getfield.(parameters, :prior)))
        priors = fill("", length(parameters))
        initialisation_priors = priors |> deepcopy
        priors_on_linear_scale = Vector{Union{Bool, String}}(undef, length(priors))
        fill!(priors_on_linear_scale, "")
        for (i, petab_parameter) in pairs(parameters)
            @unpack prior, prior_on_linear_scale, sample_prior = petab_parameter
            isnothing(prior) && continue
            priors[i] = "__Julia__" * string(prior)
            priors_on_linear_scale[i] = prior_on_linear_scale
            sample_prior == false && continue
            initialisation_priors[i] = priors[i]
        end
        parameters_df[!, :objectivePriorType] .= priors
        parameters_df[!, :priorOnLinearScale] .= priors_on_linear_scale
        parameters_df[!, :initializationPriorType] .= initialisation_priors
    end
    _check_table(parameters_df, :parameters_v1)
    return parameters_df
end

function _observables_to_table(observables::Dict{String, <:PEtabObservable})::DataFrame
    observables_df = DataFrame()
    for (id, observable) in observables
        @unpack transformation, obs, noise_formula = observable
        _transformation = isnothing(transformation) ? "lin" : string(transformation)
        if !(_transformation in VALID_SCALES)
            throw(PEtabFormatError("Transformation $_transformation is not an allowed " *
                                   "for a PEtabObservable. Allowed transformations " *
                                   "are $(VALID_SCALES)"))
        end
        row = DataFrame(observableId = id,
                        observableFormula = replace(string(obs), "(t)" => ""),
                        observableTransformation = _transformation,
                        noiseFormula = replace(string(noise_formula), "(t)" => ""),
                        noiseDistribution = "normal")
        append!(observables_df, row)
    end
    _check_table(observables_df, :observables_v1)
    return observables_df
end

function _conditions_to_table(conditions::Dict, sys::ModelSystem)::DataFrame
    specie_ids = _get_state_ids(sys)

    # Check that for each condition the same states/parameters are assigned values.
    # Required by the PEtab standard to avoid default values problems. Other checks
    # like validating conditions are assigned to model paramters are handled later after
    # transformation to a table
    if length(conditions) > 1
        conditionids = keys(conditions) |> collect
        reference_variables = conditions[conditionids[1]] |> keys |> collect .|> string
        for conditionid in conditionids
            condition_variables = conditions[conditionid] |> keys |> collect .|> string
            if all(sort(condition_variables) .== sort(reference_variables))
                continue
            end
            throw(PEtab.PEtabFormatError("Not all simulation conditions assign values " *
                                         "to the same variables. If for example " *
                                         "parameter c₁ is assigned for condition cond1 " *
                                         "it must also be assigned for condition cond2 " *
                                         ", as well as any other condition"))
        end
    end

    conditions_df = DataFrame()
    for (condition_id, condition_variables) in conditions
        row = DataFrame(conditionId = condition_id)
        for (variable_id, variable_value) in condition_variables
            row[!, variable_id] = [variable_value |> string]
        end
        conditions_df = DataFrames.vcat(conditions_df, row, cols = :union)
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

function _measurements_to_table(measurements::DataFrame, conditions::Dict)::DataFrame
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
        defaultcond = conditions == Dict("__c0__" => Dict())
        if defaultcond == true && !all(measurements_df[!, :simulation_id] .== "__c0__")
            throw(PEtab.PEtabFormatError("Simulation conditions have been provided, but " *
                                         "the simulation condition ids do not appear in " *
                                         "in the measurement table"))
        end
        rename!(measurements_df, "simulation_id" => "simulationConditionId")
    elseif !("simulationConditionId" in names(measurements_df))
        measurements_df[!, "simulationConditionId"] .= "__c0__"
    end

    # As the measurement table now follows the PEtab standard it can be checked with the
    # standard validation function
    _check_table(measurements_df, :measurements_v1)
    return measurements_df
end
