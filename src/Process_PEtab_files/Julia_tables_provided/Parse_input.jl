
# Parse PEtabParameter into PEtab-parameters file. In case of nothing a default value is set
# for a specific row.
function parse_petab_parameters(petab_parameters::Vector{PEtabParameter},
                                system,
                                simulation_conditions::Dict{String, T},
                                observables::Dict{String, PEtabObservable},
                                measurements::DataFrame)::DataFrame where T<:Dict

    # Extract any parameter that appears in the model or in simulation_conditions
    model_parameters = string.(parameters(system))
    condition_values = unique(reduce(vcat, (string.(collect(values(dict))) for dict in values(simulation_conditions))))
    non_dynamic_parameters = vcat([string(obs.obs) for obs in values(observables)], [string(obs.noise_formula) for obs in values(observables)])
    if "observable_parameters" ∈ names(measurements)
        observable_parameters = reduce(vcat, split.(string.(measurements[!, "observable_parameters"]), ';'))
    else
        observable_parameters = [""]
    end
    if "noise_parameters" ∈ names(measurements)
        noise_parameters = reduce(vcat, split.(string.(measurements[!, "noise_parameters"]), ';'))
    else
        noise_parameters = [""]
    end


    df = DataFrame()
    for parameter in petab_parameters
        parameter_id = string(parameter.parameter)

        # Often performance is improvied by estimating parameters on log-scale
        if isnothing(parameter.scale)
            parameterScale = "log10"
        else
            parameterScale = string(parameter.scale)
            @assert parameterScale ∈ ["lin", "log", "log10"]
        end

        # Default upper and lowers bounds are [1e-3, 1e3]
        if isnothing(parameter.lb)
            lowerBound = 1e-3
        else
            lowerBound = parameter.lb
        end
        if isnothing(parameter.ub)
            upperBound = 1e3
        else
            upperBound = parameter.ub
        end

        # Symbolically setting nominal value if not given
        if isnothing(parameter.value)
            nominalValue = (lowerBound + upperBound) / 2.0
        else
            nominalValue = parameter.value
        end

        estimate = parameter.estimate == true ? 1 : 0

        if (parameter_id ∉ model_parameters &&
            parameter_id ∉ condition_values &&
            parameter_id ∉ noise_parameters &&
            parameter_id ∉ observable_parameters &&
            !any(occursin.(parameter_id, non_dynamic_parameters)))

            if parameter.estimate == true
                str_write = "Parameter $parameter_id set to be estimated does not appear as a parameter in the reaction system, "
                str_write *= "as a parameter in the simulation-conditions dict, or as a noise/observable parameter in the  "
                str_write *= "measurement data. The parameter does not have any effect on the objective function."
                throw(PEtab.PEtabFormatError(str_write))
            end
            str_write = "Parameter $parameter_id set to not be estimated does not appear as a parameter in the reaction system "
            str_write *= "as a parameter in the simulation-conditions dict, or as a noise/observable parameter in the  "
            str_write *= "measurement data. The parameter does not have any effect on the objective function."
            @warn "$str_write"
        end

        row = DataFrame(parameterId = parameter_id,
                        parameterScale = parameterScale,
                        lowerBound = lowerBound,
                        upperBound = upperBound,
                        nominalValue = nominalValue,
                        estimate = estimate)
        append!(df, row)
    end

    has_priors = !all([isnothing(petab_parameters[i].prior) for i in eachindex(petab_parameters)])
    if has_priors == false
        return df
    end

    priors = Vector{String}(undef, length(petab_parameters))
    prior_on_linear_scale = Vector{Union{Bool, String}}(undef, length(petab_parameters))
    for i in eachindex(priors)
        if isnothing(petab_parameters[i].prior)
            priors[i] = ""
            prior_on_linear_scale[i] = ""
            continue
        end

        priors[i] = "__Julia__" * string(petab_parameters[i].prior)
        prior_on_linear_scale[i] = petab_parameters[i].prior_on_linear_scale
    end
    df[!, :objectivePriorType] = priors
    df[!, :priorOnLinearScale] = prior_on_linear_scale
    return df
end


# Parse PEtabExperimentalCondition into PEtab conditions file. A lot of work is going to be needed
# here to make sure that the user does not provide any bad values.
function parse_petab_conditions(simulation_conditions::Dict{String, T},
                                petab_parameters::Vector{PEtabParameter},
                                observables::Dict{String, PEtabObservable},
                                system)::DataFrame where {T<:Dict}

    # Sanity check that parameters/states set for a specific experimental condition correspond
    # to states or parameters that appear in the model
    model_state_names = replace.(string.(states(system)), "(t)" => "")
    model_parameter_names = string.(parameters(system))
    petab_parameterIds = [string(petab_parameter.parameter) for petab_parameter in petab_parameters]
    non_dynamic_parameters = vcat([string(obs.obs) for obs in values(observables)], [string(obs.noise_formula) for obs in values(observables)])

    for condition_id in keys(simulation_conditions)
        for id in keys(simulation_conditions[condition_id])
            id_str = string(id)
            if (id_str ∈ model_state_names ||
                id_str ∈ model_parameter_names ||
                any(occursin.(id_str, non_dynamic_parameters)))
                continue
            end
            str_write = "Parameter/state $id_str specifed to change between simulation-conditions (set as dictionary "
            str_write *= "key for simulation-conditions) does not as required correspond to one of the states or "
            str_write *= "parameters in the model"
            throw(PEtab.PEtabFormatError(str_write))
        end
    end

    # Check that for each condition the same states/parameters are assigned values
    if length(keys(simulation_conditions)) > 1
        _keys = collect(keys(simulation_conditions))
        reference_keys = keys(simulation_conditions[_keys[1]])
        for i in 2:length(keys(simulation_conditions))
            keysCheck = collect(keys(simulation_conditions[_keys[i]]))
            if all(sort(string.(reference_keys)) .== sort(string.(keysCheck))) == true
                continue
            end
            str_write = "Not all simulation conditions have the same parameter/states which are assigned. If "
            str_write *= "for example parameter is assigned under condition c1, it must also be assigned under "
            str_write *= "condition c2"
            throw(PEtab.PEtabFormatError(str_write))
        end
    end

    df = DataFrame()
    for (condition_id, simulation_condition) in simulation_conditions
        row = DataFrame(conditionId = condition_id)
        for (id, parameterOrSpeciesOrCompartment) in simulation_condition
            parameterOrSpeciesOrCompartmentStr = string(parameterOrSpeciesOrCompartment)
            if PEtab.is_number(parameterOrSpeciesOrCompartmentStr)
                row[!, string(id)] = [string(parameterOrSpeciesOrCompartment)]
                continue
            end

            tmp = parameterOrSpeciesOrCompartmentStr
            if tmp ∈ model_parameter_names || tmp ∈ petab_parameterIds
                row[!, string(id)] = [string(parameterOrSpeciesOrCompartment)]
                continue
            end

            if parameterOrSpeciesOrCompartmentStr == "NaN" && string(id) ∈ model_state_names
                row[!, string(id)] = [string(parameterOrSpeciesOrCompartment)]
                continue
            end

            str_write = "For simulation-condition $condition_id $id is set to map to $tmp. However, as"
            str_write *= " required $tmp does not correspond to a number, model parameter or a PEtabParameter."
            throw(PEtab.PEtabFormatError(str_write))
        end
        append!(df, row)
    end

    # Check if initial value is set in condition table (must then add a parameter to the reaction-system
    # to correctly compute gradients)
    state_names = replace.(string.(states(system)), "(t)" => "")
    for id in names(df)
        if string(id) ∈ state_names
            new_parameter = "__init__" * string(id) * "__"
            add_model_parameter!(system, new_parameter)
        end
    end

    return df
end


function add_model_parameter!(system, new_parameter)
    return nothing
end


function update_state_map(state_map, system, experimental_conditions::CSV.File)

    # Check if initial value is set in condition table (must then add a parameter to the reaction-system
    # to correctly compute gradients)
    state_names = replace.(string.(states(system)), "(t)" => "")
    for id in experimental_conditions.names
        if string(id) ∉ state_names
            continue
        end
        new_parameter = "__init__" * string(id) * "__"
        if isnothing(state_map)
            state_map = [Symbol(id) => Symbol(new_parameter)]
        else
            state_map = vcat(state_map, Symbol(id) => Symbol(new_parameter))
        end
    end
    return state_map
end


# The measurements will be rewritten into a DataFrame which follows the correct format
function parse_petab_measurements(petab_measurements::DataFrame,
                                  observables::Dict{String,PEtabObservable},
                                  simulation_conditions,
                                  petab_parameters::Vector{PEtabParameter})::DataFrame

    allowed_names = ["time", "obs_id", "observable_id", "noise_parameters", "measurement",
                     "simulation_id", "pre_equilibration_id", "pre_eq_id", "observable_parameters"]
    column_names = names(petab_measurements)
    for name in column_names
        if name ∈ allowed_names
            continue
        end
        tmp = prod([tmp * ", " for tmp in allowed_names])[1:end-2]
        str_write = "$name is not an allowed column name for PEtab measurements, allowed names are : $tmp"
        throw(PEtab.PEtabFormatError(str_write))
    end
    df = DataFrame()

    #=
        Check that input data is valid
    =#
    check_measurement_column(petab_measurements, "measurement", simulation_conditions, observables, petab_parameters)
    df[!, "measurement"] = petab_measurements[!, "measurement"]

    check_measurement_column(petab_measurements, "time", simulation_conditions, observables, petab_parameters)
    df[!, "time"] = petab_measurements[!, "time"]

    # Parse optionally provided simulation conditions 
    conditions_provided = !(length(simulation_conditions) == 1 && collect(keys(simulation_conditions))[1] == "__c0__")
    if "simulation_id" ∈ names(petab_measurements) && conditions_provided == false
        str_write = "As simulation conditions are specified in the measurement data, simulation conditions must also "
        str_write *= "be provided when building the PEtabModel"
        throw(PEtab.PEtabFormatError(str_write))

    elseif "simulation_id" ∉ names(petab_measurements) && conditions_provided == true
        str_write = "As simulation conditions are not specified in the measurement data, simulation conditions must "
        str_write *= "be provided when building the PEtabModel"
        throw(PEtab.PEtabFormatError(str_write))

    elseif "simulation_id" ∉ names(petab_measurements) && conditions_provided == false
        df[!, "simulationConditionId"] .= "__c0__"

    elseif "simulation_id" ∈ names(petab_measurements) && conditions_provided == true
        check_measurement_column(petab_measurements, "simulation_id", simulation_conditions, observables, petab_parameters)
        df[!, "simulationConditionId"] = petab_measurements[!, "simulation_id"]
    end

    _name = "observable_id" ∈ column_names ? "observable_id" : "obs_id"
    check_measurement_column(petab_measurements, _name, simulation_conditions, observables, petab_parameters)
    df[!, "observableId"] = petab_measurements[!, _name]

    if "pre_equilibration_id" ∈ column_names || "pre_eq_id" ∈ column_names # Optional column
        _name = "pre_equilibration_id" ∈ column_names ? "pre_eqparameterIduilibration_id" : "pre_eq_id"
        check_measurement_column(petab_measurements, _name, simulation_conditions, observables, petab_parameters)
        df[!, "preequilibrationConditionId"] = petab_measurements[!, _name]
    end

    if "noise_parameters" ∈ column_names
        check_measurement_column(petab_measurements, "noise_parameters", simulation_conditions, observables, petab_parameters)
        df[!, "noiseParameters"] = replace.(string.(petab_measurements[!, "noise_parameters"]), "missing" => "")
    end

    if "observable_parameters" ∈ column_names
        check_measurement_column(petab_measurements, "observable_parameters", simulation_conditions, observables, petab_parameters)
        df[!, "observableParameters"] = replace.(string.(petab_measurements[!, "observable_parameters"]), "missing" => "")
    end

    return df
end


function check_measurement_column(measurements::DataFrame,
                                    column_name::String,
                                    simulation_conditions,
                                    observables::Dict{String,PEtabObservable},
                                    petab_parameters::Vector{PEtabParameter})::Bool

    if column_name == "time" || column_name == "measurement"
        column = measurements[!, column_name]
        if typeof(column) <: Vector{<:Real}
            return true
        end
        str_write = "In the measurement data the $column_name column must only contain numerical values (e.g. Floats) "
        str_write *= "Currently column $column_name contains a none-numerical value"
        throw(PEtab.PEtabFormatError(str_write))
    end

    if column_name == "simulation_id" || column_name == "pre_equilibration_id" || column_name == "pre_eq_id"
        column = measurements[!, column_name]
        if !(typeof(column) <: Vector{<:AbstractString})
            str_write = "In the measurement data the $column_name column must only contain names (non-numerical values). "
            str_write *= "Currently column $column_name contains a numerical value"
            throw(PEtab.PEtabFormatError(str_write))
        end

        # Check that each simulation condition also is defined in the simulation-conditions struct
        conditionsIds = string.(collect(keys(simulation_conditions)))
        local _id
        id_in_simulation_conditions = true
        for id in column
            if id ∉ conditionsIds
                id_in_simulation_conditions = false
                _id = id
                break
            end
        end
        if id_in_simulation_conditions == true
            return true
        end
        str_write = "In the measurement data in the column $column_name the condition id $_id is not specifed "
        str_write *= "in the simulation-conditions. For each measurement a valid simulation-condition must be specified."
        throw(PEtab.PEtabFormatError(str_write))
    end

    if column_name == "observable_id" || column_name == "obs_id"
        column = measurements[!, column_name]
        if !(typeof(column) <: Vector{<:AbstractString})
            str_write = "In the measurement data the $column_name column must only contain names (non-numerical values). "
            str_write *= "Currently column $column_name contains a numerical value"
            throw(PEtab.PEtabFormatError(str_write))
        end

        # Check that each simulation condition also is defined in the simulation-conditions struct
        obs_ids = collect(keys(observables))
        local _id
        id_in_observables = true
        for id in column
            if id ∉ obs_ids
                id_in_observables = false
                _id = id
                break
            end
        end
        if id_in_observables == true
            return true
        end
        str_write = "In the measurement data in the column $column_name the observable id $_id is not specifed "
        str_write *= "as a PEtab observable. For each measurement a valid observable must be specified."
        throw(PEtab.PEtabFormatError(str_write))
    end

    if column_name == "observable_parameters" || column_name == "noise_parameters"
        column = measurements[!, column_name]
        if !all(typeof.(column) .<: Union{<:Real, <:AbstractString, <:Missing})
            str_write = "In the measurement data the $column_name column must only contain names or numerical values "
            str_write *= "Currently column $column_name does not conform to this."
            throw(PEtab.PEtabFormatError(str_write))
        end

        parameter_ids = string.([parameter.parameter for parameter in petab_parameters])
        for value in string.(column)

            if isempty(value) || value == "missing"
                continue
            end

            for part in split(value, ';')
                if PEtab.is_number(part)
                    continue
                end

                if part ∈ parameter_ids
                    continue
                end

                str_write = "In column $column_name the parameter $part is specifed as a noise or observable parameter. "
                str_write *= "However, as required the parameter is not specifed as a model PEtabParameter."
                throw(PEtab.PEtabFormatError(str_write))
            end
        end
        return true
    end
end


function parse_petab_observables(observables::Dict{String, PEtabObservable})::DataFrame
    df = DataFrame()

    for (observableId, observable) in observables

        if isnothing(observable.transformation)
            transformation = "lin"
        else
            transformation = string(observable.transformation)
            @assert transformation ∈ ["lin", "log", "log10"]
        end
        row = DataFrame(observableId = observableId,
                        observableFormula = replace(string(observable.obs), "(t)" => ""),
                        observableTransformation = transformation,
                        noiseFormula = replace(string(observable.noise_formula), "(t)" => ""),
                        noiseDistribution = "normal")
        append!(df, row)
    end
    return df
end


function dataframe_to_CSVFile(df::DataFrame)
    io = IOBuffer()
    io = CSV.write(io, df)
    str = String(take!(io))
    return CSV.File(IOBuffer(str), stringtype=String)
end