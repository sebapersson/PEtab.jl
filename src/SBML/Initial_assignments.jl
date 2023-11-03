function parse_initial_assignments!(model_dict::Dict, model_SBML::SBML.Model)::Nothing

    assigned_states = String[]
    for (assign_id, initial_assignment) in model_SBML.initial_assignments

        # Parse the assignment formula to Julia syntax
        formula = parse_SBML_math(initial_assignment)
        formula = replace_reactionid_formula(formula, model_SBML)
        formula = process_SBML_str_formula(formula, model_dict, model_SBML)
        formula = replace_variable(formula, "t", "0.0") # Initial time is zero

        if assign_id ∈ keys(model_dict["species"])
            model_dict["species"][assign_id].initial_value = formula
            push!(assigned_states, assign_id)

        elseif assign_id ∈ keys(model_dict["parameters"])
            if assign_id ∈ keys(model_SBML.initial_assignments) && assign_id ∈ model_dict["rate_rule_variables"]
                model_dict["parameters"][assign_id].initial_value = formula
            else
                model_dict["parameters"][assign_id].formula = formula
            end

        elseif assign_id ∈ keys(model_dict["compartments"])
            if assign_id ∈ keys(model_SBML.initial_assignments) && assign_id ∈ model_dict["rate_rule_variables"]
                model_dict["compartments"][assign_id].initial_value = formula
            else
                model_dict["compartments"][assign_id].formula = formula
                model_dict["compartments"][assign_id].initial_value = formula
            end            

        # At this point the assignment should be accepted as a state in the model, as it is neither a
        # model parameter or state
        else
            model_dict["species"][assign_id] = SpecieSBML(assign_id, false, false, formula,
                                                          "",
                                                          collect(keys(model_SBML.compartments))[1], :Amount,
                                                          false, false, false, false)
        end
    end

    # If the initial assignment for a state is the value of another state unnest until
    # initial values do not depend on other states
    for assign_id in assigned_states
        unnest_initial_assignment!(model_dict, assign_id)
    end

    # Initial assignment formula is given in conc, but if the state it acts on is given in amount
    # we need to adjust by multiplaying with compartment.
    for assign_id in assigned_states
        if assign_id ∉ keys(model_dict["species"])
            continue
        end
        if model_dict["species"][assign_id].unit == :Concentration
            continue
        end
        model_dict["species"][assign_id].initial_value = '(' * model_dict["species"][assign_id].initial_value * ") * " * model_dict["species"][assign_id].compartment
    end
end


function unnest_initial_assignment!(model_dict::Dict, assign_id::String)::Nothing

    # rateOf expression are handled later in the importer once all the rules, functions
    # etc... have been processed in order to get a proper expression for the rate of,
    # for example, a model specie
    local formula = model_dict["species"][assign_id].initial_value
    if occursin("rateOf", formula)
        return nothing
    end

    while true
        formula_start = deepcopy(formula)
        for (specie_id, specie) in model_dict["species"]
            initial_value = specie.initial_value
            if specie_id == assign_id
                continue
            end
            formula = replace_variable(formula, specie_id, string(initial_value))
        end

        if formula == formula_start
            break
        end
    end

    model_dict["species"][assign_id].initial_value = formula
    return nothing
end

