function parse_events(events::Vector{PEtabEvent}, sys::ModelSystem)::Dict{String, SBMLImporter.EventSBML}
    sbml_events = Dict{String, SBMLImporter.EventSBML}()
    isempty(events) && return sbml_events

    for (i, event) in pairs(events)
        name = "event" * string(i)
        sbml_events[name] = _parse_event(event, name, sys)
    end
    return sbml_events
end

function _parse_event(event::PEtabEvent, name::String,
                      sys::ModelSystem)::SBMLImporter.EventSBML
    @unpack condition, target_ids, target_values = event

    state_ids = _get_state_ids(sys) .|> string
    xids_sys = parameters(sys) .|> string

    # Sanity check the condition trigger
    condition = replace(condition, "(t)" => "")
    if is_number(condition) || condition in xids_sys
        condition = "t == " * condition

    elseif condition in state_ids
        throw(PEtabFormatError("The PEtabEvent even trigger ($condition) cannot be a \
                                model state. It must be a Boolean expression, a \
                                numeric value or a single parameter"))

    elseif !any(occursin.(["==", "!=", ">", "<", "≥", "≤"], condition))
        throw(PEtabFormatError("The PEtab event trigger ($condition) must be a Boolean \
                                expression (contain ==, !=, >, <, ≤, or ≥), or be a \
                                numeric value or a single parameter"))
    end

    target_ids = replace.(target_ids, "(t)" => "")
    for target_id in target_ids
        if target_id in state_ids || target_id in xids_sys
            continue
        end
        throw(PEtabFormatError("PEtabEvent target ($target_id) must be a model state or \
                                a model parameter"))
    end

    formulas = replace.(target_ids .* " = " .* target_values, "(t)" => "")
    return SBMLImporter.EventSBML(name, condition, formulas, true, false, false, false,
                                  false, false, false, false)
end
