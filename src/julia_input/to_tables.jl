function _parameters_to_table(parameters::Vector)::DataFrame
    # Most validity check occurs later during table parsing
    parameters_df = DataFrame()
    for petab_paramter in parameters
        if !(petab_paramter isa Union{PEtabParameter, PEtabNetParameter})
            throw(PEtab.PEtabInputError("Input parameters to a PEtabModel must either \
                be a PEtabParameter or a PEtabNetParameter."))
        end
        row = _parse_petab_parameter(petab_paramter)
        parameters_df = vcat(parameters_df, row)
    end
    # Add potential prior columns
    ip = findall(x -> x isa PEtabParameter, parameters)
    if !all(isnothing.(getfield.(parameters[ip], :prior)))
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
    _check_table(parameters_df, :parameters)
    return parameters_df
end

function _parse_petab_parameter(petab_parameter::PEtabParameter)::DataFrame
    @unpack parameter, scale, lb, ub, value, estimate = petab_parameter

    parameter_scale = isnothing(scale) ? "lin" : string(scale)
    if !(parameter_scale in VALID_SCALES)
        throw(PEtabFormatError("Scale $parameter_scale is not allowed for parameter \
            $parameter. Allowed scales are $(VALID_SCALES)"))
    end

    lower_bound = isnothing(lb) ? 1e-3 : lb
    upper_bound = isnothing(ub) ? 1e3 : ub
    if lower_bound > upper_bound
        throw(PEtabFormatError("Lower bound $lower_bound is larger than upper bound \
            $upper_bound for paramter $parameter"))
    end

    nominal_value = isnothing(value) ? (lower_bound + upper_bound) / 2.0 : value
    should_estimate = estimate == true ? 1 : 0
    row = DataFrame(parameterId = "$(parameter)",
                    parameterScale = parameter_scale,
                    lowerBound = lower_bound,
                    upperBound = upper_bound,
                    nominalValue = nominal_value,
                    estimate = should_estimate)
    return row
end
function _parse_petab_parameter(petab_parameter::PEtabNetParameter)::DataFrame
    @unpack netid, estimate, value = petab_parameter

    if isnothing(value)
        nominal_value = "$(netid)_julia_random"
    else
        nominal_value = "$(netid)_julia_provided"
    end
    should_estimate = estimate == true ? 1 : 0
    row = DataFrame(parameterId = "$(netid)_parameters",
                    parameterScale = "lin",
                    lowerBound = -Inf,
                    upperBound = Inf,
                    nominalValue = nominal_value,
                    estimate = should_estimate)
    return row
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
    _check_table(observables_df, :observables)
    return observables_df
end

function _conditions_to_table(conditions::Dict)::DataFrame
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
        append!(conditions_df, row)
    end

    _check_table(conditions_df, :conditions)
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
    _check_table(measurements_df, :measurements)
    return measurements_df
end

function _mapping_to_table(nnmodels::Dict{Symbol, <:NNModel})::DataFrame
    isempty(nnmodels) && return DataFrame()
    mappings_df = DataFrame()
    for (netid, nnmodel) in nnmodels
        for (i, input_id) in pairs(nnmodel.inputs)
            if nnmodel.static == true
                dftmp = DataFrame(Dict(
                    "modelEntityId" => "$(netid).inputs[0][$(i-1)]",
                    "petabEntityId" => string(input_id)))
                mappings_df = vcat(mappings_df, dftmp)
            else
                dftmp = DataFrame(Dict(
                    "modelEntityId" => "$(netid).inputs[0][$(i-1)]",
                    "petabEntityId" => "__$(netid)__input$(i-1)"))
                mappings_df = vcat(mappings_df, dftmp)
            end
        end
        for (i, output_id) in pairs(nnmodel.outputs)
            if nnmodel.static == true
                dftmp = DataFrame(Dict(
                    "modelEntityId" => "$(netid).outputs[0][$(i-1)]",
                    "petabEntityId" => "__$(netid)__output$(i-1)"))
                mappings_df = vcat(mappings_df, dftmp)
            else
                dftmp = DataFrame(Dict(
                    "modelEntityId" => "$(netid).outputs[0][$(i-1)]",
                    "petabEntityId" => string(output_id)))
                mappings_df = vcat(mappings_df, dftmp)
            end
        end
        dftmp = DataFrame(Dict(
                "modelEntityId" => "$(netid).parameters",
                "petabEntityId" => "$(netid)_parameters"))
        mappings_df = vcat(mappings_df, dftmp)
    end
    if !isempty(mappings_df)
        _check_table(mappings_df, :mapping)
    end
    return mappings_df
end

function _hybridization_to_table(nnmodels::Dict{Symbol, <:NNModel}, parameters_df::DataFrame, conditions_df::DataFrame)::DataFrame
    hybridization_df = DataFrame()
    for (netid, nnmodel) in nnmodels

        for (i, input_id) in pairs(string.(nnmodel.inputs))
            if nnmodel.static == true
                input_id in parameters_df.parameterId && continue
                input_id in names(conditions_df) && continue
                dftmp = DataFrame(targetId = input_id, targetValue = "__$(netid)__input$(i-1)")
                hybridization_df = vcat(hybridization_df, dftmp)
            else
                dftmp = DataFrame(targetId = "__$(netid)__input$(i-1)", targetValue = input_id)
                hybridization_df = vcat(hybridization_df, dftmp)
            end
        end

        for (i, output_id) in pairs(string.(nnmodel.outputs))
            nnmodel.static == false && continue
            dftmp = DataFrame(targetId = output_id, targetValue = "__$(netid)__output$(i-1)")
            hybridization_df = vcat(hybridization_df, dftmp)
        end
    end
    return hybridization_df
end
