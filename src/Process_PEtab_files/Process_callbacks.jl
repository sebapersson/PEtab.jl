# Function generating callbacksets for time-depedent SBML piecewise expressions, as callbacks are more efficient than
# using ifelse (e.g better integration stabillity)
function create_callbacks_for_piecewise(system::ODESystem,
                                        parameter_map,
                                        state_map,
                                        SBML_dict::Dict,
                                        model_name::String,
                                        path_yaml::String,
                                        dir_julia::String;
                                        custom_parameter_values::Union{Nothing, Dict}=nothing,
                                        write_to_file::Bool=true)

    p_ode_problem_names = string.(parameters(system))
    model_state_names = replace.(string.(states(system)), "(t)" => "")

    # Compute indices tracking parameters (needed as down the line we need to know if a parameter should be estimated
    # or not, as if such a parameter triggers a callback we must let it be a continious callback)
    experimental_conditions, measurements_data, parameters_data, observables_data = read_petab_files(path_yaml)
    parameter_info = process_parameters(parameters_data, custom_parameter_values=custom_parameter_values)
    measurement_info = process_measurements(measurements_data, observables_data)
    θ_indices = compute_θ_indices(parameter_info, measurement_info, system, parameter_map, state_map, experimental_conditions)

    # In case of no-callbacks the function for getting callbacks will be empty, likewise for the function
    # which compute tstops (callback-times)
    model_name = replace(model_name, "-" => "_")
    write_callbacks_str = "function getCallbacks_" * model_name * "(foo)\n"
    write_tstops_str = "\nfunction computeTstops(u::AbstractVector, p::AbstractVector)\n"

    # In case we do not have any events
    if isempty(SBML_dict["boolVariables"]) && isempty(SBML_dict["events"])
        callback_names = ""
        check_activated_t0_names = ""
        write_tstops_str *= "\t return Float64[]\nend\n"
        convert_tspan = false
    else
        for key in keys(SBML_dict["boolVariables"])
            function_str, callback_str =  create_callback(key, SBML_dict, p_ode_problem_names, model_state_names)
            write_callbacks_str *= function_str * "\n"
            write_callbacks_str *= callback_str * "\n"
        end
        for key in keys(SBML_dict["events"])
            function_str, callback_str = create_callback_event(key, SBML_dict, p_ode_problem_names, model_state_names)
            write_callbacks_str *= function_str * "\n"
            write_callbacks_str *= callback_str * "\n"
        end

        _callback_names = vcat([key for key in keys(SBML_dict["boolVariables"])], [key for key in keys(SBML_dict["events"])])
        callback_names = prod(["cb_" * name * ", " for name in _callback_names])[1:end-2]
        # Only relevant for picewise expressions
        if !isempty(SBML_dict["boolVariables"])
            check_activated_t0_names = prod(["is_active_t0_" * key * "!, " for key in keys(SBML_dict["boolVariables"])])[1:end-2]
        else
            check_activated_t0_names = ""
        end

        _write_tstops_str, convert_tspan = create_tstops_function(SBML_dict, model_state_names, p_ode_problem_names, θ_indices)
        write_tstops_str *= "\treturn" * _write_tstops_str  * "\n" * "end"
    end

    write_callbacks_str *= "\treturn CallbackSet(" * callback_names * "), Function[" * check_activated_t0_names * "], " * string(convert_tspan)  * "\nend"
    path_save = joinpath(dir_julia, model_name * "_callbacks.jl")
    if isfile(path_save)
        rm(path_save)
    end
    if write_to_file == true
        io = open(path_save, "w")
        write(io, write_callbacks_str * "\n\n")
        write(io, write_tstops_str)
        close(io)
    end
    return write_callbacks_str, write_tstops_str
end


function create_callback(callback_name::String,
                         SBML_dict::Dict,
                         p_ode_problem_names::Vector{String},
                         model_state_names::Vector{String})

    # Check if the event trigger depend on parameters which are to be i) estimated, or ii) if it depend on models state.
    # For i) it must be a cont. event in order for us to be able to compute the gradient. For ii) we cannot compute
    # tstops (the event times) prior to starting to solve the ODE so it most be cont. callback
    _condition_formula = SBML_dict["boolVariables"][callback_name][1]
    has_model_states = check_condition_has_states(_condition_formula, model_state_names)
    discrete_event = has_model_states == true ? false : true

    # Replace any state or parameter with their corresponding index in the ODE system to be comaptible with event
    # syntax
    for i in eachindex(model_state_names)
        _condition_formula = replace_variable(_condition_formula, model_state_names[i], "u["*string(i)*"]")
    end
    for i in eachindex(p_ode_problem_names)
        _condition_formula = replace_variable(_condition_formula, p_ode_problem_names[i], "integrator.p["*string(i)*"]")
    end

    # Replace inequality with - (root finding cont. event) or with == in case of
    # discrete event
    replace_with = discrete_event == true ? "==" : "-"
    condition_formula = replace(_condition_formula, "<=" => replace_with)
    condition_formula = replace(condition_formula, ">=" => replace_with)
    condition_formula = replace(condition_formula, ">" => replace_with)
    condition_formula = replace(condition_formula, "<" => replace_with)

    # Build the condition statement used in the jl function
    condition_str = "\n\tfunction condition_" * callback_name * "(u, t, integrator)\n"
    condition_str *= "\t\t" * condition_formula * "\n\tend\n"

    # Build the affect function
    which_parameter = findfirst(x -> x == callback_name, p_ode_problem_names)
    affect_str = "\tfunction affect_" * callback_name * "!(integrator)\n"
    affect_str *= "\t\tintegrator.p[" * string(which_parameter) * "] = 1.0\n\tend\n"

    # Build the callback
    if discrete_event == false
        callback_str = "\tcb_" * callback_name * " = ContinuousCallback(" * "condition_" * callback_name * ", " * "affect_" * callback_name * "!, "
    else
        callback_str = "\tcb_" * callback_name * " = DiscreteCallback(" * "condition_" * callback_name * ", " * "affect_" * callback_name * "!, "
    end
    callback_str *= "save_positions=(false, false))\n" # So we do not get problems with saveat in the ODE solver

    # Building a function which check if a callback is activated at time zero (as this is not something Julia will
    # check for us)
    side_inequality = SBML_dict["boolVariables"][callback_name][2] == "right" ? "!" : "" # Check if true or false evaluates expression to true
    active_t0_str = "\tfunction is_active_t0_" * callback_name * "!(u, p)\n"
    active_t0_str *= "\t\tt = 0.0 # Used to check conditions activated at t0=0\n" * "\t\tp[" * string(which_parameter) * "] = 0.0 # Default to being off\n"
    condition_formula = replace(_condition_formula, "integrator." => "")
    condition_formula = replace(condition_formula, "<=" => "≤")
    condition_formula = replace(condition_formula, ">=" => "≥")
    active_t0_str *= "\t\tif " * side_inequality *"(" * condition_formula * ")\n" * "\t\t\tp[" * string(which_parameter) * "] = 1.0\n\t\tend\n\tend\n"

    function_str = condition_str * '\n' * affect_str * '\n' * active_t0_str * '\n'

    return function_str, callback_str
end


function create_callback_event(event_name::String,
                               SBML_dict::Dict,
                               p_ode_problem_names::Vector{String},
                               model_state_names::Vector{String})

    event = SBML_dict["events"][event_name]
    _condition_formula = event.trigger
    affects = event.formulas
    initial_value_cond = event.trigger_initial_value

    has_model_states = check_condition_has_states(_condition_formula, model_state_names)
    discrete_event = has_model_states == true ? false : true

    # If the event trigger does not contain a model state but fixed parameters it can at a maximum be triggered once.
    if discrete_event == false
        # If we have a trigger on the form a ≤ b then event should only be
        # activated when crossing the condition from left -> right. Reverse
        # holds for ≥
        affect_neg = occursin("≤", _condition_formula)
    else
        __condition_formula = _condition_formula
        _condition_formula = "\tcond = " * _condition_formula * " && from_neg[1] == true\n"
        _condition_formula *= "\t\tfrom_neg[1] = !(" * __condition_formula * ")\n\t\treturn cond"
    end

    # TODO : Refactor and merge functionality with above
    for i in eachindex(model_state_names)
        _condition_formula = replace_variable(_condition_formula, model_state_names[i], "u["*string(i)*"]")
    end
    for i in eachindex(p_ode_problem_names)
        _condition_formula = replace_variable(_condition_formula, p_ode_problem_names[i], "integrator.p["*string(i)*"]")
    end

    # Build the condition statement used in the jl function
    if discrete_event == false
        __condition_formula = replace(_condition_formula, "≤" => "-")
        __condition_formula = replace(__condition_formula, "≥" => "-")
        condition_str = "\n\tfunction condition_" * event_name * "(u, t, integrator)\n"
        condition_str *= "\t\t" * __condition_formula * "\n\tend\n"
    else
        condition_str = "\n\tfunction _condition_" * event_name * "(u, t, integrator, from_neg)\n"
        condition_str *= "\t" * _condition_formula * "\n\tend\n"
        condition_str *= "\n\tcondition_" * event_name * " = let from_neg=" * "[" * string(!initial_value_cond) * "]\n\t\t(u, t, integrator) -> _condition_" * event_name * "(u, t, integrator, from_neg)\n\tend\n"
    end

    # Building the affect function (which can act on states and/or parameters)
    affect_str = "\tfunction affect_" * event_name * "!(integrator)\n"
    affect_str *= "\t\tu_tmp = similar(integrator.u)\n"
    affect_str *= "\t\tu_tmp .= integrator.u\n"
    for i in eachindex(affects)
        affect_str1, affect_str2 = split(affects[i], "=")
        for j in eachindex(model_state_names)
            affect_str1 = replace_variable(affect_str1, model_state_names[j], "integrator.u["*string(j)*"]")
            affect_str2 = replace_variable(affect_str2, model_state_names[j], "u_tmp["*string(j)*"]")
        end
        affect_str *= "\t\t" * affect_str1 * " = " * affect_str2 * '\n'
    end
    affect_str *= "\tend"
    for i in eachindex(p_ode_problem_names)
        affect_str = replace_variable(affect_str, p_ode_problem_names[i], "integrator.p["*string(i)*"]")
    end

    # In case the event can be activated at time zero,
    if discrete_event == true && initial_value_cond == false
        initial_value_str = "\tfunction init_" * event_name * "(c,u,t,integrator)\n"
        initial_value_str *= "\t\tcond = condition_" * event_name * "(u, t, integrator)\n"
        initial_value_str *= "\t\tif cond == true\n"
        initial_value_str *= "\t\t\taffect_" * event_name * "!(integrator)\n\t\tend\n"
        initial_value_str *= "\tend"
    elseif discrete_event == false && initial_value_cond == false
        initial_value_str = "\tfunction init_" * event_name * "(c,u,t,integrator)\n"
        initial_value_str *= "\t\tcond = " * _condition_formula * "\n"
        initial_value_str *= "\t\tif cond == true\n"
        initial_value_str *= "\t\t\taffect_" * event_name * "!(integrator)\n\t\tend\n"
        initial_value_str *= "\tend"
    else
        initial_value_str = ""
    end

    # Build the callback
    if discrete_event == false
        if affect_neg == true
            callback_str = "\tcb_" * event_name * " = ContinuousCallback(" * "condition_" * event_name * ", nothing, " * "affect_" * event_name * "!,"
        else
            callback_str = "\tcb_" * event_name * " = ContinuousCallback(" * "condition_" * event_name * ", " * "affect_" * event_name * "!, nothing,"
        end
        if initial_value_cond == false
            callback_str *= " initialize=init_" * event_name * ", "
        end
    elseif discrete_event == true
        if initial_value_cond == false
            callback_str = "\tcb_" * event_name * " = DiscreteCallback(" * "condition_" * event_name * ", " * "affect_" * event_name * "!, initialize=init_" * event_name * ", "
        else
            callback_str = "\tcb_" * event_name * " = DiscreteCallback(" * "condition_" * event_name * ", " * "affect_" * event_name * "!, "
        end
    end
    callback_str *= "save_positions=(false, false))\n" # So we do not get problems with saveat in the ODE solver

    function_str = condition_str * '\n' * affect_str * '\n' * initial_value_str * '\n'

    return function_str, callback_str
end


# Function computing t-stops (time for events) for piecewise expressions using the symbolics package
# to symboically solve for where the condition is zero.
function create_tstops_function(SBML_dict::Dict,
                                model_state_names::Vector{String},
                                p_ode_problem_names::Vector{String},
                                θ_indices::Union{ParameterIndices, Nothing})

    condition_formulas = string.(vcat([SBML_dict["boolVariables"][key][1] for key in keys(SBML_dict["boolVariables"])], [e.trigger for e in values(SBML_dict["events"])]))

    return _create_tstops_function(condition_formulas, model_state_names, p_ode_problem_names, θ_indices)
end
function create_tstops_function(events::Vector{T},
                                system,
                                θ_indices::Union{ParameterIndices, Nothing}) where T<:PEtabEvent

    model_state_names = replace.(string.(states(system)), "(t)" => "")
    p_ode_problem_names = string.(parameters(system))
    condition_formulas = [string(event.condition) for event in events]
    for (i, condition) in pairs(condition_formulas)
        if PEtab.is_number(condition) || condition ∈ p_ode_problem_names
            condition_formulas[i] = "t == " * condition
        end
    end
    return _create_tstops_function(condition_formulas, model_state_names, p_ode_problem_names, θ_indices)
end


function _create_tstops_function(condition_formulas::Vector{String},
                                 model_state_names::Vector{String},
                                 p_ode_problem_names::Vector{String},
                                 θ_indices::Union{ParameterIndices, Nothing})

    convert_tspan = false
    tstops_str = Vector{String}(undef, length(condition_formulas))
    tstops_str_alt = Vector{String}(undef, length(condition_formulas))
    i = 1
    for condition_formula in condition_formulas
        # In case the activation formula contains a state we cannot precompute the t-stop time as it depends on
        # the actual ODE solution.
        if check_condition_has_states(condition_formula, model_state_names)
            tstops_str[i] = ""
            tstops_str_alt[i] = ""
            i += 1
            continue
        end
        if !isnothing(θ_indices) && check_has_parameter_to_estimate(condition_formula, p_ode_problem_names, θ_indices)
            convert_tspan = true
        end
        if isnothing(θ_indices)
            convert_tspan = true
        end

        # We need to make the parameters and states symbolic in order to solve the condition expression
        # using the Symbolics package.
        variables_str = "@variables t, "
        variables_str *= prod(string.(collect(p_ode_problem_names)) .* ", " )[1:end-2] * " "
        variables_str *= prod(string.(collect(model_state_names)) .* ", " )[1:end-2]
        variables_symbolic = eval(Meta.parse(variables_str))

        # Note - below order counts (e.g having < first results in ~= incase what actually stands is <=)
        condition_formula = replace(condition_formula, r"≤|≥|<=|>=|<|>|==" => "~")
        condition_symbolic = eval(Meta.parse(condition_formula))

        # Expression for the time at which the condition is triggered
        expression_time = string.(Symbolics.solve_for(condition_symbolic, variables_symbolic[1], simplify=true))

        # Make compatible with the PEtab importer syntax
        for i in eachindex(model_state_names)
            expression_time = replace_variable(expression_time, model_state_names[i], "u["*string(i)*"]")
        end
        for i in eachindex(p_ode_problem_names)
            expression_time = replace_variable(expression_time, p_ode_problem_names[i], "p["*string(i)*"]")
        end
        # dual_to_float is needed as tstops for the integrator cannot be of type Dual
        tstops_str[i] = "dual_to_float(" * expression_time * ")"
        tstops_str_alt[i] = expression_time # Used when we convert timespan
        i += 1
    end

    if convert_tspan == true
        tstops = "[" * prod([isempty(tstops_str_alt[i]) ? "" : tstops_str_alt[i] * ", " for i in eachindex(tstops_str_alt)])[1:end-2] * "]"
        return tstops, convert_tspan
    else
        tstops = " Float64[" * prod([isempty(tstops_str[i]) ? "" : tstops_str[i] * ", " for i in eachindex(tstops_str)])[1:end-2] * "]"
        return tstops, convert_tspan
    end
end


function check_condition_has_states(condition_formula::AbstractString, model_state_names::Vector{String})::Bool
    for i in eachindex(model_state_names)
        _condition_formula = replace_variable(condition_formula, model_state_names[i], "")
        if _condition_formula != condition_formula
            return true
        end
    end
    return false
end


function check_has_parameter_to_estimate(condition_formula::AbstractString,
                                         p_ode_problem_names::Vector{String},
                                         θ_indices::ParameterIndices)::Bool

    # Parameters which are present for each experimental condition, and condition specific parameters
    i_ode_θ_all_conditions = θ_indices.map_ode_problem.i_ode_problem_θ_dynamic
    i_ode_problem_θ_dynamicCondition = reduce(vcat, [θ_indices.maps_conidition_id[i].i_ode_problem_θ_dynamic for i in keys(θ_indices.maps_conidition_id)])

    for i in eachindex(p_ode_problem_names)
        _condition_formula = replace_variable(condition_formula, p_ode_problem_names[i], "integrator.p["*string(i)*"]")
        if _condition_formula != condition_formula
            if i ∈ i_ode_θ_all_conditions || i ∈ i_ode_problem_θ_dynamicCondition
                return true
            end
        end
    end
    return false
end
