function _parse_events(model_SBML::SBMLImporter.ModelSBML, petab_events::Vector{PEtabEvent}, sys::ODESystem, speciemap, parametermap, name, xindices::ParameterIndices, petab_tables)::Tuple{Dict{Symbol, CallbackSet}, Bool}
    cbs = Dict{Symbol, CallbackSet}()
    condition_df = petab_tables[:conditions]

    float_tspan = _xdynamic_in_event_cond(model_SBML, xindices, petab_tables) |> !
    p_sys = string.(_get_sys_parameters(sys, speciemap, parametermap))
    cbs_sbml = SBMLImporter.create_callbacks(sys, model_SBML, name; p_PEtab = p_sys, float_tspan = float_tspan)
    if isempty(petab_events)
        for condition_id in Symbol.(condition_df.conditionId)
            cbs[condition_id] = deepcopy(cbs_sbml)
        end
        return cbs, float_tspan
    end

    # PEtab events have priority over SBML events. Hypothetically (which happens in the
    # PEtab v2 test-suite) a PEtab event can change model entities such that a
    # ContinuousCallback is activated. However, in CallbackSet this will not be triggered.
    # Thus, the affect function of the PEtab event must be modified to trigger the SBML
    # ContinuousCallback if required.
    for condition_id in Symbol.(condition_df.conditionId)
        i_events = _get_petab_events_simulation_id(petab_events, condition_id)
        isempty(i_events) && continue

        _petab_events = parse_events(petab_events[i_events], sys)
        _model_SBML = SBMLImporter.ModelSBML(name; events = _petab_events)
        cbs_petab = SBMLImporter.create_callbacks(sys, _model_SBML, name; p_PEtab = p_sys, float_tspan = float_tspan)

        petab_callbacks = DiscreteCallback[]
        for cb in cbs_petab.discrete_callbacks
            affect_petab_event! = let _cbs_sbml = cbs_sbml, _affect_petab! = cb.affect!, _save_u = [false]
                (integrator) -> _affect_petab_v2_event!(integrator, _affect_petab!, _cbs_sbml, _save_u)
            end
            cb_petab = DiscreteCallback(cb.condition, affect_petab_event!; initialize = cb.initialize, save_positions = (false, false))
            push!(petab_callbacks, cb_petab)
        end
        cbs[condition_id] = CallbackSet(petab_callbacks..., cbs_sbml.discrete_callbacks..., cbs_sbml.continuous_callbacks...)
    end
    return cbs, float_tspan
end

function _affect_petab_v2_event!(integrator, _affect_petab_cb!::Function, cbs_sbml::CallbackSet, save_u::Vector{Bool})::Nothing
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

function _get_petab_events_simulation_id(petab_events::Vector{PEtabEvent}, condition_id::Symbol)::Vector{Int64}
    isempty(petab_events) && return Int64[]
    petab_event_conditions_ids = getfield.(petab_events, :condition_ids)
    return findall(x -> condition_id in x, petab_event_conditions_ids)
end
