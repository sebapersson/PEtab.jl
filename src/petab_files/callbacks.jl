function _parse_events(
        model_SBML::SBMLImporter.ModelSBML, petab_events::Vector{PEtabEvent},
        sys::ModelSystem, speciemap, parametermap, name, xindices::ParameterIndices,
        petab_tables::PEtabTables
    )::Tuple{Dict{Symbol, CallbackSet}, Bool}
    cbs = Dict{Symbol, CallbackSet}()
    conditions_df = _get_petab_tables(petab_tables, [:conditions])[1]

    float_tspan = _xdynamic_in_event_cond(model_SBML, xindices, petab_tables) |> !
    p_sys = string.(_get_ids_sys_order(sys, speciemap, parametermap))
    state_ids = _get_state_ids(sys)
    cbs_sbml = SBMLImporter.create_callbacks(
        sys, model_SBML, name; p_PEtab = p_sys, float_tspan = float_tspan,
        _specie_ids = state_ids
    )

    if isempty(petab_events)
        for condition_id in Symbol.(conditions_df.conditionId)
            cbs[condition_id] = deepcopy(cbs_sbml)
        end
        return cbs, float_tspan
    end

    # PEtab events have priority over SBML events. Hypothetically (which happens in the
    # PEtab v2 test-suite) a PEtab event can change model entities such that a
    # ContinuousCallback is activated. However, in CallbackSet this will not be triggered.
    # Thus, the affect function of the PEtab event must be modified to trigger the SBML
    # ContinuousCallback if required.
    for condition_id in Symbol.(conditions_df.conditionId)
        i_events = _get_petab_events_simulation_id(petab_events, condition_id)
        isempty(i_events) && continue

        _petab_events = parse_events(petab_events[i_events], sys)
        _model_SBML = SBMLImporter.ModelSBML(name; events = _petab_events)
        cbs_petab = SBMLImporter.create_callbacks(
            sys, _model_SBML, name; p_PEtab = p_sys, float_tspan = float_tspan
        )

        petab_callbacks = DiscreteCallback[]
        for cb in cbs_petab.discrete_callbacks
            affect_petab_event! = let _cbs_sbml = cbs_sbml, _affect_petab! = cb.affect!, _save_u = [false]
                (integrator) -> _affect_petab_v2_event!(
                    integrator, _affect_petab!, _cbs_sbml, _save_u
                )
            end
            cb_petab = DiscreteCallback(
                cb.condition, affect_petab_event!; initialize = cb.initialize,
                save_positions = (false, false)
            )
            push!(petab_callbacks, cb_petab)
        end
        cbs[condition_id] = CallbackSet(
            petab_callbacks..., cbs_sbml.discrete_callbacks...,
            cbs_sbml.continuous_callbacks...
        )
    end
    return cbs, float_tspan
end
function _parse_events(
        petab_events::Vector{PEtabEvent}, sys::ModelSystem, speciemap, parametermap, name,
        xindices::ParameterIndices, petab_tables::PEtabTables
    )::Tuple{Dict{Symbol, CallbackSet}, Bool}
    cbs = Dict{Symbol, CallbackSet}()
    conditions_df = _get_petab_tables(petab_tables, [:conditions])[1]

    p_sys = string.(_get_ids_sys_order(sys, speciemap, parametermap))
    state_ids = _get_state_ids(sys)

    # Whether t-span should be float or not depends on all events
    sbml_events = parse_events(petab_events, sys)
    model_SBML = SBMLImporter.ModelSBML(name; events = sbml_events)
    float_tspan = _xdynamic_in_event_cond(model_SBML, xindices, petab_tables) |> !

    for condition_id in Symbol.(conditions_df.conditionId)
        _events_condition = PEtabEvent[]

        for _event in petab_events
            if !(isempty(_event.condition_ids) || condition_id in _event.condition_ids)
                continue
            end
            push!(_events_condition, _event)
        end

        _sbml_events = parse_events(_events_condition, sys)
        _model_SBML = SBMLImporter.ModelSBML(name; events = _sbml_events)
        _cbs = SBMLImporter.create_callbacks(
            sys, _model_SBML, name; p_PEtab = p_sys, float_tspan = float_tspan,
            _specie_ids = state_ids
        )

        # DiscreteCallbacks must be assigned a _save_u flagging whether solution should
        # be saved after the event has fired. Following PEtab v2, this happens if the event
        # triggers at a measurement time-poitn
        discrete_callbacks = DiscreteCallback[]
        for cb in _cbs.discrete_callbacks
            affect_petab_event! = let _affect_petab! = cb.affect!, _save_u = [false]
                (integrator) -> _affect_petab_julia_event!(integrator, _affect_petab!, _save_u)
            end
            _cb = DiscreteCallback(
                cb.condition, affect_petab_event!; initialize = cb.initialize,
                save_positions = (false, false)
            )
            push!(discrete_callbacks, _cb)
        end

        cbs[condition_id] = CallbackSet(_cbs.continuous_callbacks..., discrete_callbacks...)
    end
    return cbs, float_tspan
end


function _affect_petab_v2_event!(
        integrator, _affect_petab_cb!::Function, cbs_sbml::CallbackSet,
        save_u::Vector{Bool}
    )::Nothing
    u_tmp1 = deepcopy(integrator.u)
    p_tmp1 = deepcopy(integrator.p)
    u_tmp2 = similar(u_tmp1)
    p_tmp2 = similar(p_tmp1)

    _affect_petab_cb!(integrator)

    # PEtab events can trigger SBML events by flipping the event trigger from false to
    # true. This is not something a Julia ContinuousCallback can capture, therefore
    # here it must be checked whether an event has been triggered.
    for cb in cbs_sbml.continuous_callbacks
        condition_after = cb.condition(integrator.u, integrator.t, integrator)

        u_tmp2 .= integrator.u
        p_tmp2 .= integrator.p
        integrator.u .= u_tmp1
        integrator.p .= p_tmp1
        condition_before = cb.condition(integrator.u, integrator.t, integrator)
        integrator.u .= u_tmp2
        integrator.p .= p_tmp2

        # Event can be triggered by passing the condition from pos-neg, and from neg-pos
        if !isnothing(cb.affect!) && !isnothing(cb.affect_neg!)
            if signbit(condition_before) != signbit(condition_after)
                cb.affect!(integrator)
            end
        end

        # Even only triggered when passing from neg-pos
        if !isnothing(cb.affect!) && isnothing(cb.affect_neg!)
            if condition_before < 0.0 && condition_after > 0.0
                cb.affect!(integrator)
            end
        end

        # Even only triggered when passing from pos-neg
        if isnothing(cb.affect!) && !isnothing(cb.affect_neg!)
            if condition_before > 0.0 && condition_after < 0.0
                cb.affect_neg!(integrator)
            end
        end
    end

    # For immediate discrete callbacks (SBML events triggered at the current time), ensure
    # the event does not re-trigger after this affect function runs. Otherwise,
    # `CallbackSet`'s priority handling will cause the event to fire again. Note, triggering
    # the SBML event via this `CallbackSet` would break the time-saving, as by default the
    # solution is not saved after SBML events.
    for cb in cbs_sbml.discrete_callbacks
        cb.condition(integrator.u, integrator.t, integrator) == false && continue

        cb.condition.from_neg[1] = false
        cb.affect!(integrator)
    end

    if save_u[1] == true
        SciMLBase.savevalues!(integrator, true)
    end
    return nothing
end

function _affect_petab_julia_event!(
        integrator, _affect_petab_cb!::Function, save_u::Vector{Bool}
    )::Nothing
    _affect_petab_cb!(integrator)
    if save_u[1] == true
        SciMLBase.savevalues!(integrator, true)
    end
    return nothing
end

function _get_petab_events_simulation_id(
        petab_events::Vector{PEtabEvent}, condition_id::Symbol
    )::Vector{Int64}
    isempty(petab_events) && return Int64[]
    petab_event_conditions_ids = getfield.(petab_events, :condition_ids)
    return findall(x -> condition_id in x || isempty(x), petab_event_conditions_ids)
end

function _set_trigger_time!(events::Vector{PEtabEvent})::Nothing
    for (i, event) in pairs(events)
        trigger_time = _get_trigger_time(event)
        if !isnan(trigger_time)
            events[i] = @set event.trigger_time = trigger_time
        end
    end

    n = length(events)
    if n ≤ 1
        return nothing
    end

    # Check for duplicate trigger times among events that apply to the same condition ids
    for i in 1:(n - 1)
        ti = events[i].trigger_time
        isnan(ti) && continue
        for j in (i + 1):n
            tj = events[j].trigger_time

            isnan(tj) && continue
            ti != tj && continue

            # Events overlap if either applies to all conditions (empty ids) or their
            # intersection is non-empty
            ids_i = isempty(events[i].condition_ids) ? [:all] : events[i].condition_ids
            ids_j = isempty(events[j].condition_ids) ? [:all] : events[j].condition_ids
            if ids_i == [:all] || ids_j == [:all] || !isempty(intersect(ids_i, ids_j))
                throw(PEtabFormatError("Events $i and $j have the same trigger time \
                    t = $(round(ti, sigdigits = 2)) and overlapping condition_ids $(ids_i) \
                    and $(ids_j). PEtab.jl does not support event priority, so two events \
                    with the same trigger time within a simulation condition are not \
                    allowed. As a workaround, merge events $i and $j into a single event."))
            end
        end
    end
    return nothing
end

function _get_trigger_time(event::PEtabEvent)::Float64
    condition = replace(string(event.condition), " " => "")
    if is_number(condition)
        return parse(Float64, condition)
    end

    if !(occursin("t==", condition) || occursin("==t", condition))
        return NaN
    end

    # If the event is a parameter to estimate (not a number), the likelihood that the event
    # will trigger at the exact same time-point as a measurement is close to zero, and
    # therefore trigger_time is not saved as the model output should not be saved following
    # the event.
    condition_formula_split = split(condition, "==")
    i_formula = findfirst(x -> x != "t", condition_formula_split)
    trigger_time = string(condition_formula_split[i_formula])
    if is_number(trigger_time)
        return parse(Float64, trigger_time)
    else
        return NaN
    end
end
