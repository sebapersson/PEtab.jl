# Function generating callbacksets for time-depedent SBML piecewise expressions, as callbacks are more efficient than
# using ifelse (for example better integration stability, faster runtimes etc...)
function create_callbacks_SBML(system::ODESystem,
                               parameter_map,
                               state_map,
                               model_SBML::SBMLImporter.ModelSBML,
                               model_name::String,
                               path_yaml::String,
                               dir_julia::String;
                               custom_parameter_values::Union{Nothing, Dict}=nothing,
                               write_to_file::Bool=true)::String

    p_ode_problem_names = string.(parameters(system))
    model_specie_names = replace.(string.(states(system)), "(t)" => "")

    # Compute indices tracking parameters (needed as down the line we need to know if a parameter should be estimated
    # or not)
    experimental_conditions, measurements_data, parameters_data, observables_data = read_petab_files(path_yaml)
    parameter_info = process_parameters(parameters_data, custom_parameter_values=custom_parameter_values)
    measurement_info = process_measurements(measurements_data, observables_data)
    θ_indices = compute_θ_indices(parameter_info, measurement_info, system, parameter_map, state_map, experimental_conditions)

    # Set function names
    model_name = replace(model_name, "-" => "_")
    write_affects_conds = ""
    write_callbacks = ""
    write_active_at_t0 = ""
    write_initial_function = ""
    write_tstops = "\nfunction compute_tstops(u::AbstractVector, p::AbstractVector)\n"

    # In case we do not have any SBML related events
    if isempty(model_SBML.ifelse_parameters) && isempty(model_SBML.events)
        write_tstops *= "\t return Float64[]\nend\n"
        n_callbacks = 0
        convert_tspan = false
    else

        # For ifelse parameter
        n_callbacks = 0
        for parameter in keys(model_SBML.ifelse_parameters)
            affects_conds, callback, active_t0 = create_callback_ifelse(parameter, model_SBML, p_ode_problem_names, model_specie_names)
            write_affects_conds *= affects_conds
            write_callbacks *= callback
            write_active_at_t0 *= active_t0
            n_callbacks += 1
        end
        # For classical SBML events
        for key in keys(model_SBML.events)
            affects_conds, callback, initial_function = create_callback_SBML_event(key, model_SBML, p_ode_problem_names, model_specie_names)
            write_affects_conds *= affects_conds
            write_callbacks *= callback
            write_initial_function *= initial_function
            n_callbacks += 1
        end

        # Only relevant for picewise expressions
        if !isempty(model_SBML.ifelse_parameters)
            check_activated_t0_names = prod(["is_active_t0_" * key * "!, " for key in keys(model_SBML.ifelse_parameters)])[1:end-2]
        else
            check_activated_t0_names = ""
        end

        _write_tstops, convert_tspan = create_tstops_function(model_SBML, model_specie_names, p_ode_problem_names, θ_indices)
        write_tstops *= "\treturn" * _write_tstops  * "\n" * "end"
    end

    # Function for whether or not timespan should be converted
    write_convert_tspan = "function(foo)\n\treturn $convert_tspan \nend\n"

    # Write callback to file if required, otherwise just return the string for the callback and tstops functions
    path_save = joinpath(dir_julia, model_name * "_callbacks.jl")
    isfile(path_save) && rm(path_save)
    io = IOBuffer()
    if write_to_file == true
        write(io, write_affects_conds * "\n\n")
        write(io, write_callbacks * "\n\n")
        write(io, write_active_at_t0 * "\n\n")
        write(io, write_initial_function * "\n\n")
        write(io, write_convert_tspan * "\n\n")
        write(io, write_tstops)
    end
    callback_str = String(take!(io))
    close(io)
    if write_to_file == true
        open(path_save, "w") do f
            write(f, callback_str)
        end
    end

    return callback_str
end


function get_n_callbacks(model_SBML::SBMLImporter.ModelSBML)::Int64
    return length(keys(model_SBML.ifelse_parameters)) + length(keys(model_SBML.events))
end


function get_callback_functions(callback_str::String,
                                n_callbacks::Int64,
                                model_SBML::Union{SBMLImporter.ModelSBML, Nothing})

    if isnothing(model_SBML)
        n_ifelse_events = 0
    else
        n_ifelse_events = length(keys(model_SBML.ifelse_parameters))
    end

    if n_callbacks == 0
        cbset = CallbackSet()
        active_t0 = Function[]
        _get_tstops = "function compute_tstops(u::AbstractVector, p::AbstractVector)\nreturn Float64[]\nend"
        get_tstops = @RuntimeGeneratedFunction(Meta.parse(_get_tstops))
        convert_tspan = false

    else 
        # Number of events + tstops and convert timespan functions
        callback_functions = get_function_str(callback_str, n_ifelse_events+n_callbacks*3+2; as_str=true)

        affects = Vector{Function}(undef, n_callbacks)
        conds = Vector{Function}(undef, n_callbacks)
        cbs = Vector{SciMLBase.DECallback}(undef, n_callbacks)
        active_t0 = Vector{Function}(undef, n_ifelse_events)
        k = 1
        while k ≤ n_callbacks
            j = (k-1)*2 + 1
            affects[k] = @RuntimeGeneratedFunction(Meta.parse(callback_functions[j+1]))
            # Check if we have event where we need to track from_neg parameter, can happen 
            # with SBML events, as in order for an event to be triggered we must go form 
            # false to true
            println("callback_functions[j] = ", callback_functions[j])
            if occursin("from_neg", callback_functions[j])
                event = collect(keys(model_SBML.events))[k - n_ifelse_events]
                _cond = @RuntimeGeneratedFunction(Meta.parse(callback_functions[j]))
                conds[k] = let from_neg = [!model_SBML.events[event].trigger_initial_value]
                    (u, t, integrator) -> _cond(u, t, integrator, from_neg)
                end
                println("Got here")
            else
                println("Here instead")
                conds[k] = @RuntimeGeneratedFunction(Meta.parse(callback_functions[j]))
            end
            k += 1
        end
        k = 1
        for i in (n_callbacks*2+1):(n_callbacks*3)
            _cb = @RuntimeGeneratedFunction(Meta.parse(callback_functions[i]))
            cbs[k] = _cb(affects[k], conds[k])
            k += 1
        end
        for i in eachindex(active_t0)
            j = n_callbacks*3 + i
            active_t0[i] = @RuntimeGeneratedFunction(Meta.parse(callback_functions[j]))
        end
        _get_cbset = "function get_cbset(cbs)\n\treturn CallbackSet(" * prod("cbs[$i], " for i in 1:n_callbacks)[1:end-2] * ")\nend"
        get_cbset = @RuntimeGeneratedFunction(Meta.parse(_get_cbset))
        cbset = get_cbset(cbs)
        get_tstops = @RuntimeGeneratedFunction(Meta.parse(callback_functions[end]))
        get_convert_tspan = @RuntimeGeneratedFunction(Meta.parse(callback_functions[end-1]))
        convert_tspan = get_convert_tspan("https://xkcd.com/2694/")
    end
    
    return cbset, get_tstops, convert_tspan, active_t0
end


function create_callback_ifelse(parameter_name::String,
                                model_SBML::SBMLImporter.ModelSBML,
                                p_ode_problem_names::Vector{String},
                                model_specie_names::Vector{String})::Tuple{String, String, String}

    # Check if the event trigger depend on parameters which are to be i) estimated, or ii) if it depend on models state.
    # For i) we need to convert tspan. For ii) we cannot compute tstops (the event times) prior to starting to solve
    # the ODE so it most be cont. callback
    _condition, side_activated_with_time = model_SBML.ifelse_parameters[parameter_name]
    discrete_event = !(check_condition_has_states(_condition, model_specie_names))

    # Replace any state or parameter with their corresponding index in the ODE system to be comaptible with event
    # syntax
    for (i, specie_name) in pairs(model_specie_names)
        _condition = SBMLImporter.replace_variable(_condition, specie_name, "u["*string(i)*"]")
    end
    for (i, p_name) in pairs(p_ode_problem_names)
        _condition = SBMLImporter.replace_variable(_condition, p_name, "integrator.p["*string(i)*"]")
    end

    # Replace inequality with - (root finding cont. event) or with == in case of
    # discrete event
    replace_with = discrete_event == true ? "==" : "-"
    _condition_for_t0 = deepcopy(_condition) # Needed for checking active at t0 function
    _condition = replace(_condition, r"<=|>=|>|<" => replace_with)

    # Build the condition function
    condition_function = "\nfunction condition_" * parameter_name * "(u, t, integrator)\n"
    condition_function *= "\t" * _condition * "\nend\n"

    # Build the affect function
    i_ifelse_parameter = findfirst(x -> x == parameter_name, p_ode_problem_names)
    affect_function = "function affect_" * parameter_name * "!(integrator)\n"
    affect_function *= "\tintegrator.p[" * string(i_ifelse_parameter) * "] = 1.0\nend\n"

    # Build the callback formula
    callback_formula = "function get_callback" * parameter_name * "(affect!, cond)\n"
    if discrete_event == false
        callback_formula *= "\tcb = ContinuousCallback(cond, affect!, "
    else
        callback_formula *= "\tcb = DiscreteCallback(cond, affect!, "
    end
    callback_formula *= "save_positions=(false, false))\n" # So we do not get problems with saveat in the ODE solver
    callback_formula *= "\treturn cb\nend\n"

    # Building a function which check if a callback is activated at time zero (as this is not something Julia will
    # check for us so must be done here)
    side_inequality = side_activated_with_time == "right" ? "!" : "" # Check if true or false evaluates expression to true
    active_t0_function = "function is_active_t0_" * parameter_name * "!(u, p)\n"
    active_t0_function *= "\tt = 0.0 # Used to check conditions activated at t0=0\n" * "\tp[" * string(i_ifelse_parameter) * "] = 0.0 # Default to being off\n"
    condition_active_t0 = replace(_condition_for_t0, "integrator." => "")
    active_t0_function *= "\tif " * side_inequality *"(" * condition_active_t0 * ")\n" * "\t\tp[" * string(i_ifelse_parameter) * "] = 1.0\n\tend\nend\n"

    # Gather all the functions needed by the callback
    callback_functions = condition_function * '\n' * affect_function * '\n'

    return callback_functions, callback_formula, active_t0_function
end


function create_callback_SBML_event(event_name::String,
                                    model_SBML::SBMLImporter.ModelSBML,
                                    p_ode_problem_names::Vector{String},
                                    model_specie_names::Vector{String})::Tuple{String, String, String}

    event = model_SBML.events[event_name]
    _condition = event.trigger
    affects = event.formulas
    initial_value_cond = event.trigger_initial_value

    discrete_event = !(check_condition_has_states(_condition, model_specie_names))

    # If the event trigger does not contain a model state but fixed parameters it can at a maximum be triggered once.
    if discrete_event == false
        # If we have a trigger on the form a ≤ b then event should only be
        # activated when crossing the condition from left -> right. Reverse
        # holds for ≥
        affect_neg = occursin("≤", _condition)
    
    elseif discrete_event == true
        # Build the SBML activation, which has a check to see that the condition crosses from false to
        # true, per SBML standard
        _condition = "\tcond = " * _condition * " && from_neg[1] == true\n\tfrom_neg[1] = !(" * _condition * ")\n\treturn cond"
    end

    # Replace any state or parameter with their corresponding index in the ODE system to be comaptible with event
    # syntax
    for (i, specie_name) in pairs(model_specie_names)
        _condition = SBMLImporter.replace_variable(_condition, specie_name, "u["*string(i)*"]")
    end
    for (i, p_name) in pairs(p_ode_problem_names)
        _condition = SBMLImporter.replace_variable(_condition, p_name, "integrator.p["*string(i)*"]")
    end
    # Build the condition function used in Julia file, for discrete checking that event indeed is coming from negative
    # direction
    if discrete_event == false
        _condition_at_t0 = deepcopy(_condition)
        _condition = replace(_condition, r"≤|≥" => "-")
        condition_function = "\nfunction condition_" * event_name * "(u, t, integrator)\n\t" * _condition * "\nend\n"
        
    elseif discrete_event == true
        condition_function = "\nfunction _condition_" * event_name * "(u, t, integrator, from_neg)\n"
        condition_function *= "\t" * _condition * "\nend\n"
    end

    # Building the affect function (which can act on multiple states and/or parameters)
    affect_function = "function affect_" * event_name * "!(integrator)\n\tu_tmp = similar(integrator.u)\n\tu_tmp .= integrator.u\n"
    for (i, affect) in pairs(affects)
        # In RHS we use u_tmp to not let order affects, while in assigning LHS we use u
        affect_function1, affect_function2 = split(affect, "=")
        for j in eachindex(model_specie_names)
            affect_function1 = SBMLImporter.replace_variable(affect_function1, model_specie_names[j], "integrator.u["*string(j)*"]")
            affect_function2 = SBMLImporter.replace_variable(affect_function2, model_specie_names[j], "u_tmp["*string(j)*"]")
        end
        affect_function *= "\t" * affect_function1 * " = " * affect_function2 * '\n'
    end
    affect_function *= "end"
    for i in eachindex(p_ode_problem_names)
        affect_function = SBMLImporter.replace_variable(affect_function, p_ode_problem_names[i], "integrator.p["*string(i)*"]")
    end

    # In case the event can be activated at time zero build an initialisation function
    if discrete_event == true && initial_value_cond == false
        initial_value_str = "function init_" * event_name * "(c,u,t,integrator)\n"
        initial_value_str *= "\tcond = condition_" * event_name * "(u, t, integrator)\n"
        initial_value_str *= "\tif cond == true\n"
        initial_value_str *= "\taffect_" * event_name * "!(integrator)\n\tend\n"
        initial_value_str *= "end"
    elseif discrete_event == false && initial_value_cond == false
        initial_value_str = "function init_" * event_name * "(c,u,t,integrator)\n"
        initial_value_str *= "\tcond = " * _condition_at_t0 * "\n" # We need a Bool not minus (-) condition
        initial_value_str *= "\tif cond == true\n"
        initial_value_str *= "\taffect_" * event_name * "!(integrator)\n\tend\n"
        initial_value_str *= "end"
    else
        initial_value_str = ""
    end
    if !isempty(initial_value_str)
        throw(PEtabFileError("SBML events firing at time zero are  not supported"))
    end

    # Build the callback, consider initialisation if needed and direction for ContinuousCallback
    callback_formula = "function get_callback" * event_name * "(affect!, cond)\n"
    if discrete_event == false
        if affect_neg == true
            callback_formula *= "\tcb = ContinuousCallback(cond, nothing, affect!, "
        else
            callback_formula *= "\tcb = ContinuousCallback(cond, affect!, nothing, "
        end
    elseif discrete_event == true
        callback_formula *= "\tcb = DiscreteCallback(cond, affect!, "
    end
    callback_formula *= "save_positions=(false, false))\n\treturn cb\nend\n" 

    return condition_function * affect_function, callback_formula, initial_value_str
end


# Function computing t-stops (time for events) for piecewise expressions using the symbolics package
# to symboically solve for where the condition is zero.
function create_tstops_function(model_SBML::SBMLImporter.ModelSBML,
                                model_specie_names::Vector{String},
                                p_ode_problem_names::Vector{String},
                                θ_indices::Union{ParameterIndices, Nothing})::Tuple{String, Bool}

    conditions = string.(vcat([model_SBML.ifelse_parameters[key][1] for key in keys(model_SBML.ifelse_parameters)], [e.trigger for e in values(model_SBML.events)]))
    return _create_tstops_function(conditions, model_specie_names, p_ode_problem_names, θ_indices)
end
function create_tstops_function(events::Vector{T},
                                system,
                                θ_indices::Union{ParameterIndices, Nothing}) where T<:PEtabEvent

    model_specie_names = replace.(string.(states(system)), "(t)" => "")
    p_ode_problem_names = string.(parameters(system))
    conditions = [string(event.condition) for event in events]
    for (i, condition) in pairs(conditions)
        if PEtab.is_number(condition) || condition ∈ p_ode_problem_names
            conditions[i] = "t == " * condition
        end
    end
    return _create_tstops_function(conditions, model_specie_names, p_ode_problem_names, θ_indices)
end


function _create_tstops_function(conditions::Vector{String},
                                 model_specie_names::Vector{String},
                                 p_ode_problem_names::Vector{String},
                                 θ_indices::Union{ParameterIndices, Nothing})::Tuple{String, Bool}

    convert_tspan::Bool = false
    tstops = Vector{String}(undef, length(conditions))
    tstops_to_float = Vector{String}(undef, length(conditions))
    for (i, condition) in pairs(conditions)

        # In case the activation formula contains a state we cannot precompute the t-stop time as it depends on
        # the actual ODE solution.
        if check_condition_has_states(condition, model_specie_names)
            tstops[i] = ""
            tstops_to_float[i] = ""
            continue
        end
        # If condition contains parameters to estimate the tspan must not be converted to floats, rather kept as
        # Duals (slower, but yields accurate gradients)
        if !isnothing(θ_indices) && check_has_parameter_to_estimate(condition, p_ode_problem_names, θ_indices)
            convert_tspan = true
        end
        if isnothing(θ_indices)
            convert_tspan = true
        end

        # We need to make the parameters and states symbolic in order to solve the condition expression
        # using the Symbolics package.
        _variables = "@variables t, "
        _variables *= prod(string.(collect(p_ode_problem_names)) .* ", " )[1:end-2] * " "
        _variables *= prod(string.(collect(model_specie_names)) .* ", " )[1:end-2]
        variables_symbolic = eval(Meta.parse(_variables))

        # Note - below order counts (e.g having < first results in ~= incase what actually stands is <=)
        _condition = replace(condition, r"≤|≥|<=|>=|<|>|==" => "~")
        condition_symbolic = eval(Meta.parse(_condition))

        # Expression for the time at which the condition is triggered
        local expression_time
        try
            expression_time = string.(Symbolics.solve_for(condition_symbolic, variables_symbolic[1], simplify=true))
        catch
            throw(SBMLSupport("Not possible to solve for time event is activated"))
        end

        # Make compatible with the PEtab importer syntax
        for (i, specie_name) in pairs(model_specie_names)
            expression_time = SBMLImporter.replace_variable(expression_time, specie_name, "u["*string(i)*"]")
        end
        for (i, p_name) in pairs(p_ode_problem_names)
            expression_time = SBMLImporter.replace_variable(expression_time, p_name, "p["*string(i)*"]")
        end

        # dual_to_float is needed as tstops for the integrator cannot be of type Dual
        tstops_to_float[i] = "dual_to_float(" * expression_time * ")"
        tstops[i] = expression_time # Used when we convert timespan
        i += 1
    end

    if convert_tspan == true
        _tstops = "[" * prod([isempty(_t) ? "" : _t * ", " for _t in tstops])[1:end-2] * "]"
    else
        _tstops = " Float64[" * prod([isempty(_t) ? "" : _t * ", " for _t  in tstops_to_float])[1:end-2] * "]"
    end
    return _tstops, convert_tspan
end


function check_condition_has_states(condition::AbstractString, model_specie_names::Vector{String})::Bool
    for i in eachindex(model_specie_names)
        _condition = SBMLImporter.replace_variable(condition, model_specie_names[i], "")
        if _condition != condition
            return true
        end
    end
    return false
end


function check_has_parameter_to_estimate(condition::T,
                                         p_ode_problem_names::Vector{String},
                                         θ_indices::ParameterIndices)::Bool where T<:AbstractString

    # Parameters which are present for each experimental condition, and condition specific parameters
    i_ode_θ_all_conditions = θ_indices.map_ode_problem.i_ode_problem_θ_dynamic
    i_ode_problem_θ_dynamicCondition = reduce(vcat, [θ_indices.maps_conidition_id[i].i_ode_problem_θ_dynamic for i in keys(θ_indices.maps_conidition_id)])

    for i in eachindex(p_ode_problem_names)
        _condition = SBMLImporter.replace_variable(condition, p_ode_problem_names[i], "integrator.p["*string(i)*"]")
        if _condition != condition
            if i ∈ i_ode_θ_all_conditions || i ∈ i_ode_problem_θ_dynamicCondition
                return true
            end
        end
    end
    return false
end
