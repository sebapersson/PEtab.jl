
# Parse PEtabParameter into PEtab-parameters file. In case of nothing a default value is set
# for a specific row.
function parse_petab_parameters(petab_parameters::Vector{PEtabParameter},
                                system,
                                simulation_conditions::Dict,
                                observables::Dict,
                                measurements::DataFrame,
                                state_map,
                                parameter_map)::DataFrame

    # Extract any parameter that appears in the model or in simulation_conditions
    model_parameters = string.(parameters(system))
    condition_values = unique(reduce(vcat,
                                     (string.(collect(values(dict)))
                                      for dict in values(simulation_conditions))))
    condition_parameters = unique(reduce(vcat,
                                         (string.(collect(keys(dict)))
                                          for dict in values(simulation_conditions))))
    non_dynamic_parameters = vcat([string(obs.obs) for obs in values(observables)],
                                  [string(obs.noise_formula) for obs in values(observables)])
    if "observable_parameters" ∈ names(measurements)
        observable_parameters = reduce(vcat,
                                       split.(string.(measurements[!,
                                                                   "observable_parameters"]),
                                              ';'))
    else
        observable_parameters = [""]
    end
    if "noise_parameters" ∈ names(measurements)
        noise_parameters = reduce(vcat,
                                  split.(string.(measurements[!, "noise_parameters"]), ';'))
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

    # Sanity check if all model have been defined anywhere 
    for model_parameter in model_parameters
        cond1 = model_parameter ∉ df[!, :parameterId]
        cond2 = model_parameter ∉ condition_parameters
        cond3 = isnothing(parameter_map) ? true :
                model_parameter ∉ string.(first.(parameter_map))
        cond4 = isnothing(state_map) ? true : model_parameter ∉ string.(first.(state_map))
        if length(model_parameter) > 2 && model_parameter[1:2] == "__" &&
           model_parameter[(end - 1):end] == "__"
            continue
        end
        if cond1 && cond2 && cond3 && cond4
            @warn "No value has been specified for model parameters $model_parameter, it defaults to zero"
        end
    end

    has_priors = !all([isnothing(petab_parameters[i].prior)
                       for i in eachindex(petab_parameters)])
    if has_priors == false
        return df
    end

    priors = Vector{String}(undef, length(petab_parameters))
    initialisation_priors = similar(priors)
    prior_on_linear_scale = Vector{Union{Bool, String}}(undef, length(petab_parameters))
    for i in eachindex(priors)
        if isnothing(petab_parameters[i].prior)
            priors[i] = ""
            prior_on_linear_scale[i] = ""
            initialisation_priors[i] = ""
            continue
        end

        priors[i] = "__Julia__" * string(petab_parameters[i].prior)
        prior_on_linear_scale[i] = petab_parameters[i].prior_on_linear_scale
        if petab_parameters[i].sample_from_prior == true
            initialisation_priors[i] = priors[i]
        else
            initialisation_priors[i] = ""
        end
    end
    df[!, :objectivePriorType] = priors
    df[!, :priorOnLinearScale] = prior_on_linear_scale
    df[!, :initializationPriorType] = initialisation_priors
    return df
end

# Parse PEtabExperimentalCondition into PEtab conditions file. A lot of work is going to be needed
# here to make sure that the user does not provide any bad values.
function parse_petab_conditions(simulation_conditions::Dict,
                                petab_parameters::Vector{PEtabParameter},
                                observables::Dict,
                                system)::DataFrame

    # Sanity check that parameters/states set for a specific experimental condition correspond
    # to states or parameters that appear in the model
    model_state_names = replace.(string.(states(system)), "(t)" => "")
    model_parameter_names = string.(parameters(system))
    petab_parameterIds = [string(petab_parameter.parameter)
                          for petab_parameter in petab_parameters]
    non_dynamic_parameters = vcat([string(obs.obs) for obs in values(observables)],
                                  [string(obs.noise_formula) for obs in values(observables)])

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
                                  observables::Dict,
                                  simulation_conditions,
                                  petab_parameters::Vector{PEtabParameter})::DataFrame
    allowed_names = ["time", "obs_id", "observable_id", "noise_parameters", "measurement",
                     "simulation_id", "pre_equilibration_id", "pre_eq_id",
                     "observable_parameters"]
    column_names = names(petab_measurements)
    for name in column_names
        if name ∈ allowed_names
            continue
        end
        tmp = prod([tmp * ", " for tmp in allowed_names])[1:(end - 2)]
        str_write = "$name is not an allowed column name for PEtab measurements, allowed names are : $tmp"
        throw(PEtab.PEtabFormatError(str_write))
    end
    df = DataFrame()

    #=
        Check that input data is valid
    =#
    check_measurement_column(petab_measurements, "measurement", simulation_conditions,
                             observables, petab_parameters)
    df[!, "measurement"] = petab_measurements[!, "measurement"]

    check_measurement_column(petab_measurements, "time", simulation_conditions, observables,
                             petab_parameters)
    df[!, "time"] = petab_measurements[!, "time"]

    # Parse optionally provided simulation conditions 
    conditions_provided = !(length(simulation_conditions) == 1 &&
                            collect(keys(simulation_conditions))[1] == "__c0__")
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
        check_measurement_column(petab_measurements, "simulation_id", simulation_conditions,
                                 observables, petab_parameters)
        df[!, "simulationConditionId"] = petab_measurements[!, "simulation_id"]
    end

    _name = "observable_id" ∈ column_names ? "observable_id" : "obs_id"
    check_measurement_column(petab_measurements, _name, simulation_conditions, observables,
                             petab_parameters)
    df[!, "observableId"] = petab_measurements[!, _name]

    if "pre_equilibration_id" ∈ column_names || "pre_eq_id" ∈ column_names # Optional column
        _name = "pre_equilibration_id" ∈ column_names ? "pre_eqparameterIduilibration_id" :
                "pre_eq_id"
        check_measurement_column(petab_measurements, _name, simulation_conditions,
                                 observables, petab_parameters)
        df[!, "preequilibrationConditionId"] = petab_measurements[!, _name]
    end

    if "noise_parameters" ∈ column_names
        check_measurement_column(petab_measurements, "noise_parameters",
                                 simulation_conditions, observables, petab_parameters)
        df[!, "noiseParameters"] = replace.(string.(petab_measurements[!,
                                                                       "noise_parameters"]),
                                            "missing" => "")
    end

    if "observable_parameters" ∈ column_names
        check_measurement_column(petab_measurements, "observable_parameters",
                                 simulation_conditions, observables, petab_parameters)
        df[!, "observableParameters"] = replace.(string.(petab_measurements[!,
                                                                            "observable_parameters"]),
                                                 "missing" => "")
    end

    return df
end

function check_measurement_column(measurements::DataFrame,
                                  column_name::String,
                                  simulation_conditions,
                                  observables::Dict,
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

    if column_name == "simulation_id" || column_name == "pre_equilibration_id" ||
       column_name == "pre_eq_id"
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

function parse_petab_observables(observables::Dict)::DataFrame
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
                        noiseFormula = replace(string(observable.noise_formula),
                                               "(t)" => ""),
                        noiseDistribution = "normal")
        append!(df, row)
    end
    return df
end

function process_petab_events(events::Union{PEtabEvent, AbstractVector, Nothing},
                              system,
                              θ_indices::ParameterIndices)

    # Must be a vector for downstream processing 
    if events isa PEtabEvent
        events = [events]
    end

    if !isnothing(events)
        callbacks = Vector{SciMLBase.DECallback}(undef, length(events))
        write_tstops = "\nfunction computeTstops(u::AbstractVector, p::AbstractVector)\n"
        for (i, event) in pairs(events)
            event_name = "event" * string(i)
            _affect, _condition, _callback = process_petab_event(event, event_name, system)
            affect! = @RuntimeGeneratedFunction(Meta.parse(_affect))
            condition = @RuntimeGeneratedFunction(Meta.parse(_condition))
            callback = @RuntimeGeneratedFunction(Meta.parse(_callback))
            callbacks[i] = callback(affect!, condition)
        end
        _get_cbset = "function get_cbset(cbs)\n\treturn CallbackSet(" *
                     prod("cbs[$i], " for i in 1:length(events))[1:(end - 2)] * ")\nend"
        get_cbset = @RuntimeGeneratedFunction(Meta.parse(_get_cbset))
        cbset = get_cbset(callbacks)
    else
        cbset = CallbackSet()
    end

    write_tstops = "\nfunction computeTstops(u::AbstractVector, p::AbstractVector)\n"
    if !isnothing(events)
        _write_tstops, convert_tspan = PEtab.create_tstops_function(events, system,
                                                                    θ_indices)
    else
        _write_tstops, convert_tspan = "Float64[]", false
    end
    write_tstops *= "\treturn " * _write_tstops * "\n" * "end"
    get_tstops = @RuntimeGeneratedFunction(Meta.parse(write_tstops))

    return cbset, get_tstops, convert_tspan
end

function process_petab_event(event::PEtabEvent, event_name,
                             system)::Tuple{String, String, String}
    state_names = replace.(string.(states(system)), "(t)" => "")
    parameter_names = string.(parameters(system))

    # Sanity check input, trigger 
    condition = replace(string(event.condition), "(t)" => "")
    if PEtab.is_number(condition) || condition ∈ parameter_names
        condition = "t == " * condition
    elseif condition ∈ state_names
        str_write = "A PEtab event trigger cannot be a model state as condition. It must be a Boolean expression, or a "
        str_write *= "single constant value or parameter"
        throw(PEtabFormatError(str_write))
    elseif !any(occursin.(["==", "!=", ">", "<", "≥", "≤"], condition))
        str_write = "A PEtab event trigger must be a Boolean expression (contain ==, !=, >, <, ≤, or ≥), or a single "
        str_write *= "value or parameter. This does not hold for condition: $condition"
        throw(PEtabFormatError(str_write))
    end

    # Sanity check, target
    if typeof(event.target) <: Vector{<:Any}
        if !(typeof(event.affect) <: Vector{<:Any}) ||
           length(event.target) != length(event.affect)
            str_write = "In case of several event targets (targets provided as vector) the affect vector must match the length of the affect vector"
            throw(PEtabFormatError(str_write))
        end
        targets = event.target
    else
        # Input needs to be a Vector for downstream processing
        targets = [event.target]
    end

    # Sanity check affect 
    if typeof(event.affect) <: Vector{<:Any}
        if !(typeof(event.target) <: Vector{<:Any}) ||
           length(event.target) != length(event.affect)
            str_write = "In case of several event targets (targets provided as vector) the affect vector must match the length of the affect vector"
            throw(PEtabFormatError(str_write))
        end
        affects = event.affect
    else
        # Input needs to be a Vector for downstream processing
        affects = [event.affect]
    end

    targets = replace.(string.(targets), "(t)" => "")
    for target in targets
        if target ∉ state_names && target ∉ parameter_names
            str_write = "Event target must be either a model parameter or model state. This does not hold for $target"
            throw(PEtabFormatError(str_write))
        end
    end

    condition_has_states = check_condition_has_states(condition, state_names)
    discrete_event = condition_has_states == false

    if discrete_event == true
        # Only for time-triggered events, here we can help the user to replace any 
        # in-equality signs used 
        condition = replace(condition, r"≤|≥|<=|>=|<|>" => "==")

    elseif discrete_event == false
        # If we have a trigger on the form a ≤ b then event should only be 
        # activated when crossing the condition from left -> right. Reverse
        # holds for ≥
        affect_neg = any(occursin.(["≤", "<", "=<"], condition))
        affect_equality = occursin.("==", condition)
        condition = replace(condition, r"≤|≥|<=|>=|<|>|==" => "-")
    end

    # Building the condition syntax for the event 
    for i in eachindex(state_names)
        condition = PEtab.SBMLImporter.replace_variable(condition, state_names[i],
                                                        "u[" * string(i) * "]")
    end
    for i in eachindex(parameter_names)
        condition = PEtab.SBMLImporter.replace_variable(condition, parameter_names[i],
                                                        "integrator.p[" * string(i) * "]")
    end
    condition_str = "\nfunction condition_" * event_name * "(u, t, integrator)\n\t" *
                    condition * "\nend\n"

    # Build the affect syntax for the event. Note, a tmp variable is used in case of several affects. For example, if the 
    # event affects u[1] and u[2], then I do not want that a change in u[1] should affect the value for u[2], similar holds 
    # for parameters 
    affect_str = "function affect_" * event_name *
                 "!(integrator)\n\tu_tmp = similar(integrator.u)\n\tu_tmp .= integrator.u\n\tp_tmp = similar(integrator.p)\n\tp_tmp .= integrator.p\n\n"
    affects = replace.(string.(affects), "(t)" => "")
    for (i, affect) in pairs(affects)
        _affect = targets[i] * " = " * affect
        _affect1, _affect2 = split(_affect, "=")
        for j in eachindex(state_names)
            _affect1 = PEtab.SBMLImporter.replace_variable(_affect1, state_names[j],
                                                           "integrator.u[" * string(j) *
                                                           "]")
            _affect2 = PEtab.SBMLImporter.replace_variable(_affect2, state_names[j],
                                                           "u_tmp[" * string(j) * "]")
        end
        for j in eachindex(parameter_names)
            _affect1 = PEtab.SBMLImporter.replace_variable(_affect1, parameter_names[j],
                                                           "integrator.p[" * string(j) *
                                                           "]")
            _affect2 = PEtab.SBMLImporter.replace_variable(_affect2, parameter_names[j],
                                                           "p_tmp[" * string(j) * "]")
        end
        affect_str *= "\t\t" * _affect1 * " = " * _affect2 * '\n'
    end
    affect_str *= '\n' * "\tend"

    # Build the callback 
    callback_str = "function get_callback" * event_name * "(affect!, cond)\n"
    if discrete_event == false
        if affect_equality == true
            callback_str *= "\tcb = ContinuousCallback(cond, affect!, "
        elseif affect_neg == true
            callback_str *= "\tcb = ContinuousCallback(cond, nothing, affect!, "
        else
            callback_str *= "\tcb = ContinuousCallback(cond, affect!, nothing, "
        end
    else
        callback_str *= "\tcb = DiscreteCallback(cond, affect!, "
    end
    callback_str *= "save_positions=(false, false))\n" # So we do not get problems with saveat in the ODE solver 
    callback_str *= "\treturn cb\nend\n"

    return affect_str, condition_str, callback_str
end

function dataframe_to_CSVFile(df::DataFrame)
    io = IOBuffer()
    io = CSV.write(io, df)
    str = String(take!(io))
    return CSV.File(IOBuffer(str), stringtype = String)
end
