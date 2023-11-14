function parse_SBML_initial_assignments!(model_SBML::ModelSBML, libsbml_model::SBML.Model)::Nothing

    assigned_states = String[]
    for (assign_id, initial_assignment) in libsbml_model.initial_assignments

        # Parse the assignment formula to Julia syntax
        formula = parse_SBML_math(initial_assignment)
        formula = replace_reactionid_formula(formula, libsbml_model)
        formula = process_SBML_str_formula(formula, model_SBML, libsbml_model)
        formula = replace_variable(formula, "t", "0.0") # Initial time is zero

        if assign_id ∈ keys(model_SBML.species)
            model_SBML.species[assign_id].initial_value = formula
            push!(assigned_states, assign_id)

        elseif assign_id ∈ keys(model_SBML.parameters)
            if assign_id ∈ keys(libsbml_model.initial_assignments) && assign_id ∈ model_SBML.rate_rule_variables
                model_SBML.parameters[assign_id].initial_value = formula
            else
                model_SBML.parameters[assign_id].formula = formula
            end

        elseif assign_id ∈ keys(model_SBML.compartments)
            if assign_id ∈ keys(libsbml_model.initial_assignments) && assign_id ∈ model_SBML.rate_rule_variables
                model_SBML.compartments[assign_id].initial_value = formula
            else
                model_SBML.compartments[assign_id].formula = formula
                model_SBML.compartments[assign_id].initial_value = formula
            end            

        # At this point the assignment should be accepted as a state in the model, as it is neither a
        # model parameter or state
        else
            model_SBML.species[assign_id] = SpecieSBML(assign_id, false, false, formula,
                                                          "",
                                                          "1.0", "", :Amount,
                                                          false, false, false, false)
        end
    end

    # If the initial assignment for a state is the value of another state unnest until
    # initial values do not depend on other states
    for assign_id in assigned_states
        unnest_initial_assignment!(model_SBML, assign_id)
    end

    # Initial assignment formula is given in conc, but if the state it acts on is given in amount
    # we need to adjust by multiplaying with compartment.
    for assign_id in assigned_states
        if assign_id ∉ keys(model_SBML.species)
            continue
        end
        if model_SBML.species[assign_id].unit == :Concentration
            continue
        end
        model_SBML.species[assign_id].initial_value = '(' * model_SBML.species[assign_id].initial_value * ") * " * model_SBML.species[assign_id].compartment
    end

    # Adjusting for unit, sometimes species might have their initial value multiplied by a 
    # compartment, that might be given by an assignment rule. This insert the rule-formula 
    # to ensure correct mapping for simulations
    for (specie_id, specie) in model_SBML.species
        for (compartment_id, compartment) in model_SBML.compartments
            if compartment.assignment_rule == false
                continue
            end
            specie.initial_value = replace_variable(specie.initial_value, compartment_id, compartment.formula)
        end
        for (parameter_id, parameter) in model_SBML.parameters
            if parameter.assignment_rule == false
                continue
            end
            specie.initial_value = replace_variable(specie.initial_value, parameter_id, parameter.formula)
        end
    end
end


function unnest_initial_assignment!(model_SBML::ModelSBML, assign_id::String)::Nothing

    # rateOf expression are handled later in the importer once all the rules, functions
    # etc... have been processed in order to get a proper expression for the rate of,
    # for example, a model specie
    local formula = model_SBML.species[assign_id].initial_value
    if occursin("rateOf", formula)
        return nothing
    end

    while true
        formula_start = deepcopy(formula)
        for (specie_id, specie) in model_SBML.species
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

    model_SBML.species[assign_id].initial_value = formula
    return nothing
end

