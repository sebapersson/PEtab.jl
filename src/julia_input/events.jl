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
    state_ids = _get_state_ids(sys) .|> string
    xids_sys = parameters(sys) .|> string

    # Sanity check the condition trigger
    condition = replace(string(event.condition), "(t)" => "")
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

    # Input needs to be a Vector for downstream processing (both for target and effect)
    if typeof(event.target) <: AbstractVector
        targets = event.target
    else
        targets = [event.target]
    end
    if typeof(event.affect) <: AbstractVector
        affects = event.affect
    else
        affects = [event.affect]
    end

    # Sanity check, target
    if length(targets) != length(affects)
        throw(PEtabFormatError("The number of PEtabEvent targets ($(targets)) must \
                                equal the number of PEtabEvent affects ($(affects))"))
    end
    # Sanity check affect
    targets = replace.(string.(targets), "(t)" => "")
    for target in targets
        if target in state_ids || target in xids_sys
            continue
        end
        throw(PEtabFormatError("PEtabEvent target ($target) must be a model state or " *
                               "a model parameter"))
    end

    formulas = replace.(string.(targets) .* " = " .* string.(affects), "(t)" => "")
    return SBMLImporter.EventSBML(name, condition, formulas, false, false, false, false,
                                  false, false, false, false)
end
