function process_initial_assignment(model_SBML, model_dict::Dict)

    assigned_states = String[]
    for (assign_id, initial_assignment) in model_SBML.initial_assignments
        
        # Parse the assignment formula to Julia syntax
        formula = SBML_math_to_str(initial_assignment)
        formula = replace_reactionid_with_math(formula, model_SBML)
        formula = process_SBML_str_formula(formula, model_dict, model_SBML)
        formula = replace_whole_word(formula, "t", "0.0") # Initial time is zero 

        if assign_id ∈ keys(model_dict["states"])
            model_dict["states"][assign_id] = formula
            push!(assigned_states, assign_id)

        elseif assign_id ∈ keys(model_dict["parameters"])
            model_dict["parameters"][assign_id] = formula

        # At this point the assignment should be accepted as a state in the model, as it is neither a 
        # model parameter or state
        else
            model_dict["states"][assign_id] = formula
            model_dict["stateGivenInAmounts"][assign_id] = (true, collect(keys(model_SBML.compartments))[1])
            model_dict["hasOnlySubstanceUnits"][assign_id] =  false 
            model_dict["isBoundaryCondition"][assign_id] = false
            model_dict["derivatives"][assign_id] = "D(" * assign_id * ") ~ "
        end        
    end

    # If the initial assignment for a state is the value of another state unnest until 
    # initial values do not depend on other states
    for assign_id in assigned_states
        unnest_initial_assignment!(model_dict, assign_id)
    end

    # Initial assignment formula is given in conc, but if the state it acts on is given in amount 
    # we need to adjust by multiplaying with compartment compartment.
    for assign_id in assigned_states
        if assign_id ∉ keys(model_SBML.species)
            continue
        end
        if model_dict["stateGivenInAmounts"][assign_id][1] == false
            continue
        end
        model_dict["states"][assign_id] = '(' * model_dict["states"][assign_id] * ") * " * model_SBML.species[assign_id].compartment
    end
end


function unnest_initial_assignment!(model_dict::Dict, assign_id::String)::Nothing
    
    # rateOf expression are handled later in the importer once all the rules, functions 
    # etc... have been processed in order to get a proper expression for the rate of, 
    # for example, a model specie
    local formula = model_dict["states"][assign_id]
    if occursin("rateOf", formula)
        return nothing
    end

    while true
        formula_start = deepcopy(formula)
        for (state_id, state_at_t0) in model_dict["states"]
            if state_id == assign_id
                continue
            end
            formula = replace_whole_word(formula, state_id, state_at_t0)
        end

        if formula == formula_start
            break
        end
    end

    model_dict["states"][assign_id] = formula
    return nothing
end

