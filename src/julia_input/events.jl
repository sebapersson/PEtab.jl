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
