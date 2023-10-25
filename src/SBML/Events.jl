function parse_SBML_events!(model_dict, model_SBML, non_constant_parameter_names::Vector{String})::Nothing

    event_index = 1
    for (event_name, event) in model_SBML.events

        # Parse the event trigger into proper Julia syntax
        formula_trigger = parse_SBML_math(event.trigger.math)
        formula_trigger = SBML_function_to_math(formula_trigger, model_dict["modelFunctions"])
        if isempty(formula_trigger)
            continue
        end
        formula_trigger = parse_event_trigger(formula_trigger, model_dict, model_SBML)

        # Parse assignments and assignment formulas 
        event_formulas::Vector{String} = Vector{String}(undef, length(event.event_assignments))
        event_assignments::Vector{String} = similar(event_formulas)
        not_skip_assignments::Vector{Bool} = fill(true, length(event_formulas))
        for (i, event_assignment) in pairs(event.event_assignments)

            # Check if event should be ignored as no math child was provided 
            if isempty(parse_SBML_math(event_assignment.math))
                not_skip_assignments[i] = false
                continue
            end

            event_assignments[i] = event_assignment.variable

            # Parse the event formula into the correct Julia syntax 
            event_formulas[i] = parse_SBML_math(event_assignment.math)
            event_formulas[i] = SBML_function_to_math(event_formulas[i], model_dict["modelFunctions"])
            event_formulas[i] = replace_variable(event_formulas[i], "t", "integrator.t")
            event_formulas[i] = replace_reactionid_with_math(event_formulas[i], model_SBML)
            
            # Formulas are given in concentration, but species can also be given in amounts. Thus, we must 
            # adjust for compartment for the latter
            if event_assignments[i] ∈ keys(model_SBML.species)
                if event_assignments[i] ∉ keys(model_SBML.species) 
                    continue
                end
                if model_dict["stateGivenInAmounts"][event_assignments[i]][1] == false 
                    continue
                end
                event_formulas[i] = model_SBML.species[event_assignments[i]].compartment *  " * (" * event_formulas[i] * ')'
                continue
            end
            if event_assignments[i] ∈ keys(model_SBML.parameters) || event_assignments[i] ∈ non_constant_parameter_names
                continue
            end

            # At this state the assign variable should be treated as a new specie in the model, which for the first time 
            # is defined here (allowed per SBML standard)
            model_dict["states"][event_assignments[i]] = 1.0
            model_dict["stateGivenInAmounts"][event_assignments[i]] = (true, collect(keys(model_SBML.compartments))[1])
            model_dict["hasOnlySubstanceUnits"][event_assignments[i]] =  false 
            model_dict["isBoundaryCondition"][event_assignments[i]] = false
            model_dict["derivatives"][event_assignments[i]] = "D(" * event_assignments[i] * ") ~ "   
        end

        #=
            Handle special case where an event assignment acts on a compartment, here specie conc. in said 
            compartment must be adjusted via;
            conc_new = conc_old * V_old / V_new
        =#
        adjust_event_compartment_change!!!(event_formulas, event_assignments, not_skip_assignments, model_dict, model_SBML)

        event_name = isempty(event_name) ? "event" * string(event_index) : event_name
        model_dict["events"][event_name] = [formula_trigger, event_assignments[not_skip_assignments] .* " = " .* event_formulas[not_skip_assignments], event.trigger.initial_value]
        event_index += 1
    end
end


# Note, first three arguments can change 
function adjust_event_compartment_change!!!(event_formulas::Vector{String},
                                            event_assignments::Vector{String}, 
                                            not_skip_assignments::Vector{Bool},
                                            model_dict::Dict, 
                                            model_SBML)::Nothing

    event_assignments_cp = deepcopy(event_assignments)
    if isempty(model_SBML.compartments)
        return nothing
    end
    if !any(occursin.(event_assignments_cp[not_skip_assignments], keys(model_SBML.compartments)))
        return nothing
    end

    for (i, assign_to) in pairs(event_assignments_cp[not_skip_assignments])

        # Only potentiallt adjust species if a compartment is assigned to in the 
        # event
        if assign_to ∉ keys(model_SBML.compartments)
            continue
        end

        # Check if a specie given in concentration is being assigned 
        for (specie_id, specie) in model_SBML.species
            if specie.compartment != assign_to
                continue
            end
            if model_dict["stateGivenInAmounts"][specie_id][1] == true
                continue
            end
            
            # Adjust the conc. of specie
            if specie_id ∈ event_assignments
                is = findfirst(x -> x == specie_id, event_assignments)
                event_formulas[is] = "(" * event_formulas[is] * ")" * "*" * assign_to * "/" * "(" * event_formulas[i] * ")"
                continue
            end

            # Must add new event new to adjust conc. for species that is assigned to during event, but whose 
            # conc. changes as its compart volume changes  
            _formula = specie_id * "*" * assign_to * "/" * "(" * event_formulas[i] * ")"
            event_assignments = push!(event_assignments, specie_id)
            event_formulas = push!(event_formulas, _formula)
            not_skip_assignments = push!(not_skip_assignments, true)
        end
    end
    return nothing
end


# Rewrites triggers in events to the correct Julia syntax
function parse_event_trigger(formula_trigger::T, model_dict, model_SBML)::T where T<:AbstractString

    # Stay consistent with t as time :)
    formula_trigger = replace_variable(formula_trigger, "time", "t")
    
    # SBML equations are given in conc, need to adapt scale state if said state is given in amounts, 
    # rateOf expressions are handled later
    for (species_id, specie) in model_SBML.species
        if species_id ∈ keys(model_SBML.species) && model_dict["stateGivenInAmounts"][species_id][1] == false
            continue
        end
        if occursin("rateOf", formula_trigger)
            continue
        end
        formula_trigger = replace_variable(formula_trigger, species_id, species_id * '/' * specie.compartment)
    end

    # For making downstream processing easier remove starting and ending paranthesis 
    if formula_trigger[1] == '(' && formula_trigger[end] == ')'
        formula_trigger = formula_trigger[2:end-1]
    end

    # Special case where the trigger formula already is given in Julia syntax (sometimes happen in SBML)
    if occursin(r"<|≤|>|≥", formula_trigger)
        formula_trigger = replace(formula_trigger, "<" => "≤")
        formula_trigger = replace(formula_trigger, ">" => "≥")
        return formula_trigger
    end

    # Trigger can be a number, then event is triggered when formula is not equal to said number
    if typeof(formula_trigger) <: Real || is_number(formula_trigger)
        return string(formula_trigger) * " != 0"
    end

    # Done handling special trigger formats, this is the standard processing
    if "geq" == formula_trigger[1:3]
        _formula_trigger = formula_trigger[5:end-1]
        separator = "≥"
    elseif "gt" == formula_trigger[1:2]
        _formula_trigger = formula_trigger[4:end-1]
        separator = "≥"
    elseif "leq" == formula_trigger[1:3]
        _formula_trigger = formula_trigger[5:end-1]
        separator = "≤"
    elseif "lt" == formula_trigger[1:2]
        _formula_trigger = formula_trigger[4:end-1]
        separator = "≤"
    end
    parts = split_between(_formula_trigger, ',')

    # Account for potential reaction-math kinetics making out the trigger 
    parts[1] = replace_reactionid_with_math(parts[1], model_SBML)
    parts[2] = replace_reactionid_with_math(parts[2], model_SBML)

    return parts[1] * " " * separator * " " * parts[2] 
end