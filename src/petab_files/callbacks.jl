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
            affect_petab_event! = let _cbs_sbml = cbs_sbml, _affect_petab! = cb.affect!
                (integrator) -> _affect_petab_v2_event!(integrator, _affect_petab!, _cbs_sbml)
            end
            cb_petab = DiscreteCallback(cb.condition, affect_petab_event!)
            push!(petab_callbacks, cb_petab)
        end
        cbs[condition_id] = CallbackSet(petab_callbacks..., cbs_sbml.discrete_callbacks..., cbs_sbml.continuous_callbacks...)
    end
    return cbs, float_tspan
end

function _affect_petab_v2_event!(integrator, _affect_petab_cb!::Function, cbs_sbml::CallbackSet)::Nothing
    _affect_petab_cb!(integrator)
    for cb in cbs_sbml.continuous_callbacks
        if cb.condition(integrator.u, integrator.t, integrator) == 0.0
            cb.affect!(integrator)
        end
    end
    return nothing
end

function _get_petab_events_simulation_id(petab_events::Vector{PEtabEvent}, condition_id::Symbol)::Vector{Int64}
    isempty(petab_events) && return Int64[]
    petab_event_conditions_ids = getfield.(petab_events, :condition_ids)
    return findall(x -> condition_id in x, petab_event_conditions_ids)
end
