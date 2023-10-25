# TODO: Refactor code and document functions. Check SBMLToolkit if can be used.



"""
    SBML_to_ModellingToolkit(pathXml::String, model_name::String, dir_model::String)

Convert a SBML file in pathXml to a Julia ModelingToolkit file and store
the resulting file in dir_model with name model_name.jl.
"""
function SBML_to_ModellingToolkit(pathXml::String, path_jl_file::String, model_name::AbstractString; only_extract_model_dict::Bool=false, 
                                  ifelse_to_event::Bool=true, write_to_file::Bool=true)

    model_SBML = readSBML(pathXml)
    model_dict = build_model_dict(model_SBML, ifelse_to_event)

    if only_extract_model_dict == false
        model_str = create_ode_model(model_dict, path_jl_file, model_name, write_to_file)
        return model_dict, model_str
    end

    return model_dict, ""
end


function build_model_dict(model_SBML, ifelse_to_event::Bool)

    # Nested dictionaries to store relevant model data:
    # i) Model parameters (constant during for a simulation)
    # ii) Model parameters that are nonConstant (e.g due to events) during a simulation
    # iii) Model states
    # iv) Model function (functions in the SBML file we rewrite to Julia syntax)
    # v) Model rules (rules defined in the SBML model we rewrite to Julia syntax)
    # vi) Model derivatives (derivatives defined by the SBML model)
    model_dict = Dict()
    model_dict["states"] = Dict()
    model_dict["hasOnlySubstanceUnits"] = Dict()
    model_dict["stateGivenInAmounts"] = Dict()
    model_dict["isBoundaryCondition"] = Dict()
    model_dict["parameters"] = Dict()
    model_dict["modelFunctions"] = Dict()
    model_dict["derivatives"] = Dict()
    model_dict["boolVariables"] = Dict()
    model_dict["events"] = Dict()
    model_dict["reactions"] = Dict()
    model_dict["algebraicRules"] = Dict()
    model_dict["assignmentRulesStates"] = Dict()
    model_dict["compartment_formula"] = Dict()
    model_dict["inputFunctions"] = Dict()

    for (state_id, state) in model_SBML.species
        # If initial amount is zero or nothing (default) should use initial-concentration if non-empty 
        if isnothing(state.initial_amount) && isnothing(state.initial_concentration)
            model_dict["states"][state_id] = "0.0"
            model_dict["stateGivenInAmounts"][state_id] = (false, state.compartment)
        elseif !isnothing(state.initial_concentration)
            model_dict["states"][state_id] = string(state.initial_concentration)
            model_dict["stateGivenInAmounts"][state_id] = (false, state.compartment)
        else 
            model_dict["states"][state_id] = string(state.initial_amount)
            model_dict["stateGivenInAmounts"][state_id] = (true, state.compartment)
        end

        # Setup for downstream processing 
        model_dict["hasOnlySubstanceUnits"][state_id] = isnothing(state.only_substance_units) ? false : state.only_substance_units
        model_dict["isBoundaryCondition"][state_id] = state.boundary_condition 

        # In case equation is given in conc., but state is given in amounts 
        model_dict["derivatives"][state_id] = "D(" * state_id * ") ~ "

        # In case being a boundary condition the state can only be changed by the user 
        if model_dict["isBoundaryCondition"][state_id] == true
           model_dict["derivatives"][state_id] *= "0.0"
        end

        # In case the conc. is given in initial conc, but the state should be in amounts this 
        # must be acounted for with initial values
        if model_dict["stateGivenInAmounts"][state_id][1] == false && model_dict["hasOnlySubstanceUnits"][state_id] == true
            model_dict["stateGivenInAmounts"][state_id] = (true, state.compartment)
            model_dict["states"][state_id] = string(state.initial_concentration) * " * " * state.compartment
        end
    end

    # Extract model parameters and their default values. In case a parameter is non-constant 
    # it is treated as a state. Compartments are treated simular to states (allowing them to 
    # be dynamic)
    non_constant_parameter_names::Vector{String} = String[]
    for (parameter_id, parameter) in model_SBML.parameters
        if parameter.constant == true
            model_dict["parameters"][parameter_id] = string(parameter.value)
            continue
        end

        model_dict["hasOnlySubstanceUnits"][parameter_id] = false
        model_dict["stateGivenInAmounts"][parameter_id] = (false, "")
        model_dict["isBoundaryCondition"][parameter_id] = false
        model_dict["states"][parameter_id] = isnothing(parameter.value) ? "0.0" : string(parameter.value)
        model_dict["derivatives"][parameter_id] = parameter_id * " ~ "
        non_constant_parameter_names = push!(non_constant_parameter_names, parameter_id)
    end
    for (compartment_id, compartment) in model_SBML.compartments
        # Allowed in SBML ≥ 2.0 with nothing, should then be interpreted as 
        # having no compartment (equal to a value of 1.0 for compartment)
        if compartment.constant == true
            size = isnothing(compartment.size) ? 1.0 : compartment.size
            model_dict["parameters"][compartment_id] = string(size)
            continue
        end
        
        model_dict["hasOnlySubstanceUnits"][compartment_id] = false
        model_dict["stateGivenInAmounts"][compartment_id] = (false, "")
        model_dict["isBoundaryCondition"][compartment_id] = false
        model_dict["states"][compartment_id] = isnothing(compartment.size) ? 1.0 : compartment.size
        model_dict["derivatives"][compartment_id] = compartment_id * " ~ "
        non_constant_parameter_names = push!(non_constant_parameter_names, compartment_id)
    end

    # Rewrite SBML functions into Julia syntax functions and store in dictionary to allow them to
    # be inserted into equation formulas downstream
    for (function_name, SBML_function) in model_SBML.function_definitions
        if isnothing(SBML_function.body)
            continue
        end
        args = get_SBML_function_args(SBML_function)
        function_formula = parse_SBML_math(SBML_function.body.body, true)
        model_dict["modelFunctions"][function_name] = [args, function_formula]
    end

    parse_SBML_events!(model_dict, model_SBML, non_constant_parameter_names)

    assignment_rules_names = []
    rate_rules_names = []
    for rule in model_SBML.rules
        if rule isa SBML.AssignmentRule
            rule_formula = extract_rule_formula(rule)
            assignment_rules_names = push!(assignment_rules_names, rule.variable)
            process_assignment_rule!(model_dict, rule_formula, rule.variable, model_SBML)
        end

        if rule isa SBML.RateRule
            rule_formula = extract_rule_formula(rule)
            rate_rules_names = push!(rate_rules_names, rule.variable)
            process_rate_rule!(model_dict, rule_formula, rule.variable, model_SBML)
        end

        if rule isa SBML.AlgebraicRule
            _rule_formula = extract_rule_formula(rule)
            rule_formula = SBML_function_to_math(_rule_formula, model_dict["modelFunctions"])
            rule_name = isempty(model_dict["algebraicRules"]) ? "1" : maximum(keys(model_dict["algebraicRules"])) * "1" # Need placeholder key 
            model_dict["algebraicRules"][rule_name] = "0 ~ " * rule_formula
        end
    end

    # In case we have that the compartment is given by an assignment rule, then we need to account for this 
    for (compartment_id, compartment_formula) in model_dict["compartment_formula"]
        for (eventId, event) in model_dict["events"]
            trigger_formula = event[1]
            event_assignments = event[2]
            trigger_formula = replace_variable(trigger_formula, compartment_id, compartment_formula)
            for i in eachindex(event_assignments)
                event_assignments[i] = replace_variable(event_assignments[i], compartment_id, compartment_formula)
            end
            model_dict["events"][eventId] = [trigger_formula, event_assignments, event[3]]
        end
    end

    # Positioned after rules since some assignments may include functions
    process_initial_assignment(model_SBML, model_dict)

    # Process chemical reactions 
    for (id, reaction) in model_SBML.reactions
        # Process kinetic math into Julia syntax 
        _formula = parse_SBML_math(reaction.kinetic_math)
               
        # Add values for potential kinetic parameters (where-statements)
        for (parameter_id, parameter) in reaction.kinetic_parameters
            _formula = replace_variable(_formula, parameter_id, string(parameter.value))
        end

        formula = process_SBML_str_formula(_formula, model_dict, model_SBML, check_scaling=true)
        model_dict["reactions"][reaction.name] = formula
        
        for reactant in reaction.reactants
            model_dict["isBoundaryCondition"][reactant.species] == true && continue # Constant state  
            compartment = model_SBML.species[reactant.species].compartment
            stoichiometry = isnothing(reactant.stoichiometry) ? "1" : string(reactant.stoichiometry)
            stoichiometry = stoichiometry[1] == '-' ? "(" * stoichiometry * ")" : stoichiometry
            compartment_scaling = model_dict["hasOnlySubstanceUnits"][reactant.species] == true ? "*" : "/" * compartment * "*"
            model_dict["derivatives"][reactant.species] *= " - " * stoichiometry * compartment_scaling * "(" * formula * ")"
        end
        for product in reaction.products
            model_dict["isBoundaryCondition"][product.species] == true && continue # Constant state  
            compartment = model_SBML.species[product.species].compartment
            if isnothing(product.id)
                stoichiometry = isnothing(product.stoichiometry) ? "1" : string(product.stoichiometry)
            else
                stoichiometry = product.id
            end
            stoichiometry = stoichiometry[1] == '-' ? "(" * stoichiometry * ")" : stoichiometry
            compartment_scaling = model_dict["hasOnlySubstanceUnits"][product.species] == true ? "*" : "/" * compartment * "*"
            model_dict["derivatives"][product.species] *= " + " * stoichiometry * compartment_scaling * "(" * formula * ")"
        end
    end
    # For states given in amount but model equations are in conc., multiply with compartment, also handle potential 
    # reaction identifayers in the derivative
    for (state_id, derivative) in model_dict["derivatives"]

        model_dict["derivatives"][state_id] = replace_reactionid_with_math(model_dict["derivatives"][state_id], model_SBML)

        if model_dict["stateGivenInAmounts"][state_id][1] == false
            continue
        end
        # Here equations should be given in amounts 
        if model_dict["hasOnlySubstanceUnits"][state_id] == true
            continue
        end
        # Algebraic rule (see below)
        if replace(derivative, " " => "")[end] == '~' || replace(derivative, " " => "")[end] == '0'
            continue
        end
        derivative = replace(derivative, "~" => "~ (") 
        model_dict["derivatives"][state_id] = derivative * ") * " * model_SBML.species[state_id].compartment
    end

    # For states given by assignment rules 
    for (state, formula) in model_dict["assignmentRulesStates"]
        # Must track if species is given in amounts or conc.
        if !occursin("rateOf", formula)
            _formula = process_SBML_str_formula(formula, model_dict, model_SBML; check_scaling=true)
            if state ∈ keys(model_SBML.species) && model_SBML.species[state].only_substance_units == false
                cmult = model_dict["stateGivenInAmounts"][state][1] == true ? " * " * model_SBML.species[state].compartment : ""
                _formula = "(" * _formula * ")" * cmult
            end
        else
            _formula = formula
        end
        model_dict["derivatives"][state] = state * " ~ " * _formula
        if state ∈ non_constant_parameter_names
            delete!(model_dict["states"], state)
            delete!(model_dict["parameters"], state)
            non_constant_parameter_names = filter(x -> x != state, non_constant_parameter_names)
        end
    end
    
    # Check which parameters are a part derivatives or input function. If a parameter is not a part, e.g is an initial
    # assignment parameters, add to dummy variable to keep it from being simplified away.
    is_in_ode = falses(length(model_dict["parameters"]))
    for du in values(model_dict["derivatives"])
        for (i, pars) in enumerate(keys(model_dict["parameters"]))
            if replace_variable(du, pars, "") !== du
                is_in_ode[i] = true
            end
        end
    end
    for input_function in values(model_dict["inputFunctions"])
        for (i, pars) in enumerate(keys(model_dict["parameters"]))
            if replace_variable(input_function, pars, "") !== input_function
                is_in_ode[i] = true
            end
        end
    end

    # Rewrite any time-dependent ifelse to boolean statements such that we can express these as events.
    # This is recomended, as it often increases the stabillity when solving the ODE, and decreases run-time
    if ifelse_to_event == true
        time_dependent_ifelse_to_bool!(model_dict)
    end

    # In case the model has algebraic rules some of the derivatives (up to this point) are zero. To figure out 
    # which variable for which the derivative should be eliminated as the state conc. is given by the algebraic
    # rule cycle through rules to see which state has not been given as assignment by another rule. Moreover, return 
    # flag that model is a DAE so it can be properly processed when creating PEtabODEProblem. 
    if !isempty(model_dict["algebraicRules"])
        for (species, reaction) in model_dict["derivatives"]
            should_continue = true
            # In case we have zero derivative for a state (e.g S ~ 0 or S ~)
            if species ∈ rate_rules_names || species ∈ assignment_rules_names
                continue
            end
            if replace(reaction, " " => "")[end] != '~' && replace(reaction, " " => "")[end] != '0'
                continue
            end
            if species ∈ keys(model_SBML.species) && model_SBML.species[species].constant == true
                continue
            end
            if model_dict["isBoundaryCondition"][species] == true && model_dict["stateGivenInAmounts"][species][1] == true && model_SBML.species[species].constant == true
                continue
            end
            if species ∈ keys(model_SBML.species) && model_dict["stateGivenInAmounts"][species][1] == false && model_dict["isBoundaryCondition"][species] == true 
                continue
            end

            # Check if state occurs in any of the algebraic rules 
            for (rule_id, rule) in model_dict["algebraicRules"]
                if replace_variable(rule, species, "") != rule 
                    should_continue = false
                end
            end
            should_continue == true && continue

            # If we reach this point the state eqution is zero without any form 
            # of assignment -> state must be solved for via the algebraic rule 
            delete!(model_dict["derivatives"], species)
        end
    end
    for non_constant_parameter in non_constant_parameter_names
        if non_constant_parameter ∉ keys(model_dict["derivatives"])
            continue
        end
        if replace(model_dict["derivatives"][non_constant_parameter], " " => "")[end] == '~'
            model_dict["derivatives"][non_constant_parameter] *= string(model_dict["states"][non_constant_parameter])
        end
    end

    # Up to this point technically some states can have a zero derivative, but their value can change because 
    # their compartment changes. To sidestep this, turn the state into an equation 
    for (specie, reaction) in model_dict["derivatives"]
        if specie ∉ keys(model_SBML.species)
            continue
        end
        if model_dict["isBoundaryCondition"][specie] == true && model_SBML.species[specie].constant == false
            continue
        end
        if replace(reaction, " " => "")[end] != '~' && replace(reaction, " " => "")[end] != '0'
            continue
        end
        divide_with_compartment = model_dict["stateGivenInAmounts"][specie][1] == false
        c = model_SBML.species[specie].compartment
        if divide_with_compartment == false
            continue
        end
        if model_dict["stateGivenInAmounts"][specie][1] == true
            model_dict["derivatives"][specie] = specie * " ~ (" * model_dict["states"][specie] * ") / " * c
        else
            # Must account for the fact that the compartment can change in size, and thus need to 
            # account for its initial value 
            if c ∈ keys(model_dict["parameters"])
                model_dict["derivatives"][specie] = specie * " ~ (" * model_dict["states"][specie] * ")"
            else
                model_dict["derivatives"][specie] = specie * " ~ (" * model_dict["states"][specie] * ")" * " * " * string(model_dict["states"][c]) * " / " * c 
            end
        end
    end

    # In case the model has a conversion factor 
    for (specie, reaction) in model_dict["derivatives"]
        if specie ∈ assignment_rules_names
            continue
        end
        if !isnothing(model_SBML.conversion_factor)
            model_dict["derivatives"][specie] *= " * " * model_SBML.conversion_factor
        end
    end


    # Sometimes parameter can be non-constant, but still have a constant rhs and they primarly change value 
    # because of event assignments. This must be captured, so the SBML importer will look at the RHS of non-constant 
    # parameters, and if it is constant the parameter will be moved to the parameter regime again in order to avoid 
    # simplifaying the parameter away.
    for id in non_constant_parameter_names
        # Algebraic rule 
        if id ∉ keys(model_dict["derivatives"])
            continue
        end
        lhs, rhs = replace.(split(model_dict["derivatives"][id], '~'), " " => "")
        if lhs[1] == 'D'
            continue
        end
        if !is_number(rhs)
            continue
        end
        model_dict["derivatives"][id] = "D(" * id * ") ~ 0" 
        model_dict["states"][id] = rhs
        non_constant_parameter_names = filter(x -> x != id, non_constant_parameter_names)
    end

    #=
        Sometimes the volume might change over time but the amount should stay constant, as we have a boundary condition
        In this case it follows that amount n (amount), V (compartment) and conc. are related 
        via the chain rule by - I need to change my ODE:s and add state
        dn/dt = d(n/V)/dt*V + n*dV/dt 
    =#
    for specie in keys(model_SBML.species)

        if !(model_dict["stateGivenInAmounts"][specie][1] == true && 
             model_SBML.species[specie].compartment ∈  rate_rules_names && 
             model_dict["isBoundaryCondition"][specie] == true &&
             specie ∈ rate_rules_names && 
             model_dict["hasOnlySubstanceUnits"][specie] == false)
            continue
        end

        # Derivative and inital values for concentratin species
        compartment = model_SBML.species[specie].compartment
        specie_conc = "__" * specie * "__conc__"
        model_dict["states"][specie_conc] = model_dict["states"][specie] * " / " * compartment
        i_start = findfirst(x -> x == '~', model_dict["derivatives"][specie]) + 1
        i_end = findlast(x -> x == ')', model_dict["derivatives"][specie])
        model_dict["derivatives"][specie_conc] = "D(" * specie_conc  * ") ~ " * model_dict["derivatives"][specie][i_start:i_end]

        # Rebuild derivative for amount specie
        itmp1, itmp2 = findfirst(x -> x == '~', model_dict["derivatives"][specie_conc]) + 1, findfirst(x -> x == '~', model_dict["derivatives"][compartment]) + 1
        model_dict["derivatives"][specie] = "D(" * specie * ") ~ " * model_dict["derivatives"][specie_conc][itmp1:end] * "*" *  compartment * " + " * specie * "*" * model_dict["derivatives"][compartment][itmp2:end] * " / " * compartment
    end

    model_dict["numOfParameters"] = string(length(keys(model_dict["parameters"])))
    model_dict["numOfSpecies"] = string(length(keys(model_dict["states"])))
    model_dict["non_constant_parameter_names"] = non_constant_parameter_names
    model_dict["rate_rules_names"] = rate_rules_names


    # Replace potential rateOf expression with corresponding rate
    for (parameter_id, parameter) in model_dict["parameters"]
        model_dict["parameters"][parameter_id] = replace_rateOf(parameter, model_dict)
    end
    for (state_id, state) in model_dict["states"]
        model_dict["states"][state_id] = replace_rateOf(state, model_dict)
    end
    for (state_id, derivative) in model_dict["derivatives"]
        model_dict["derivatives"][state_id] = replace_rateOf(derivative, model_dict)
    end
    for (id, rule_formula) in model_dict["assignmentRulesStates"]
        model_dict["assignmentRulesStates"][id] = replace_rateOf(rule_formula, model_dict)
    end
    for (event_id, event) in model_dict["events"]
        for (i, assignment) in pairs(event[2])
            event[2][i] = replace_rateOf(event[2][i], model_dict)
        end
        # Trigger
        event[1] = replace_rateOf(event[1], model_dict)
    end
    for (rule_id, rule) in model_dict["algebraicRules"]
        model_dict["algebraicRules"][rule_id] = replace_rateOf(rule, model_dict)
    end

    return model_dict
end


"""
    create_ode_model(model_dict, path_jl_file, model_name, juliaFile, write_to_file::Bool)

Takes a model_dict as defined by build_model_dict
and creates a Julia ModelingToolkit file and stores
the resulting file in dir_model with name model_name.jl.
"""
function create_ode_model(model_dict, path_jl_file, model_name, write_to_file::Bool)

    dict_model_str = Dict()
    dict_model_str["variables"] = Dict()
    dict_model_str["stateArray"] = Dict()
    dict_model_str["variableParameters"] = Dict()
    dict_model_str["algebraicVariables"] = Dict()
    dict_model_str["parameters"] = Dict()
    dict_model_str["parameterArray"] = Dict()
    dict_model_str["derivatives"] = Dict()
    dict_model_str["ODESystem"] = Dict()
    dict_model_str["initialSpeciesValues"] = Dict()
    dict_model_str["trueParameterValues"] = Dict()

    dict_model_str["variables"] = "    ModelingToolkit.@variables t "
    dict_model_str["stateArray"] = "    stateArray = ["
    dict_model_str["variableParameters"] = ""
    dict_model_str["algebraicVariables"] = ""
    dict_model_str["parameters"] = "    ModelingToolkit.@parameters "
    dict_model_str["parameterArray"] = "    parameterArray = ["
    dict_model_str["derivatives"] = "    eqs = [\n"
    dict_model_str["ODESystem"] = "    @named sys = ODESystem(eqs, t, stateArray, parameterArray)"
    dict_model_str["initialSpeciesValues"] = "    initialSpeciesValues = [\n"
    dict_model_str["trueParameterValues"] = "    trueParameterValues = [\n"

    # Add dummy to create system if empty 
    if isempty(model_dict["states"])
        model_dict["states"]["fooo"] = "0.0"
        model_dict["derivatives"]["fooo"] = "D(fooo) ~ 0.0"
    end            

    for key in keys(model_dict["states"])
        dict_model_str["variables"] *= key * "(t) "
    end
    for (key, value) in model_dict["assignmentRulesStates"]
        dict_model_str["variables"] *= key * "(t) "
    end

    for (key, value) in model_dict["states"]
        dict_model_str["stateArray"] *= key * ", "
    end
    for (key, value) in model_dict["assignmentRulesStates"]
        dict_model_str["stateArray"] *= key * ", "
    end
    dict_model_str["stateArray"] = dict_model_str["stateArray"][1:end-2] * "]"
    
    if length(model_dict["inputFunctions"]) > 0
        dict_model_str["algebraicVariables"] = "    ModelingToolkit.@variables"
        for key in keys(model_dict["inputFunctions"])
            dict_model_str["algebraicVariables"] *= " " * key * "(t)"
        end
    end
            
    for key in keys(model_dict["parameters"])
        dict_model_str["parameters"] *= key * " "
    end

    for (index, key) in enumerate(keys(model_dict["parameters"]))
        if index < length(model_dict["parameters"])
            dict_model_str["parameterArray"] *= key * ", "
        else
            dict_model_str["parameterArray"] *= key * "]"
        end
    end
    if isempty(model_dict["parameters"])
        dict_model_str["parameters"] = ""
        dict_model_str["parameterArray"] *= "]"
    end


    s_index = 1
    for key in keys(model_dict["states"])
        # If the state is not part of any reaction we set its value to zero, 
        # unless is has been removed from derivative dict as it is given by 
        # an algebraic rule 
        if key ∉ keys(model_dict["derivatives"]) # Algebraic rule given 
            continue
        end
        if occursin(Regex("~\\s*\$"),model_dict["derivatives"][key])
            model_dict["derivatives"][key] *= "0.0"
        end
        if s_index == 1
            dict_model_str["derivatives"] *= "    " * model_dict["derivatives"][key]
        else
            dict_model_str["derivatives"] *= ",\n    " * model_dict["derivatives"][key]
        end
        s_index += 1
    end
    for key in keys(model_dict["inputFunctions"])
        if s_index != 1
            dict_model_str["derivatives"] *= ",\n    " * model_dict["inputFunctions"][key]
        else
            dict_model_str["derivatives"] *= "    " * model_dict["inputFunctions"][key]
            s_index += 1
        end
    end
    for key in keys(model_dict["algebraicRules"])
        if s_index != 1
            dict_model_str["derivatives"] *= ",\n    " * model_dict["algebraicRules"][key]
        else
            dict_model_str["derivatives"] *= "    " * model_dict["algebraicRules"][key]
            s_index += 1
        end
    end
    for (key, formula) in model_dict["assignmentRulesStates"]
        what_write = key ∈ keys(model_dict["derivatives"]) ? model_dict["derivatives"][key] : key * " ~ " * model_dict["assignmentRulesStates"][key]
        if s_index != 1
            dict_model_str["derivatives"] *= ",\n    " * what_write
        else
            dict_model_str["derivatives"] *= "    " * what_write
            s_index += 1
        end
    end
    dict_model_str["derivatives"] *= "\n"
    dict_model_str["derivatives"] *= "    ]"

    index = 1
    for (key, value) in model_dict["states"]

        # These should not be mapped into the u0Map as they are just dynamic 
        # parameters expression which are going to be simplifed away (and are 
        # not in a sense states since they are not give by a rate-rule)
        if key ∈ model_dict["non_constant_parameter_names"] && key ∉ model_dict["rate_rules_names"]
            continue
        end
        if typeof(value) <: Real
            value = string(value)
        elseif tryparse(Float64, value) !== nothing
            value = string(parse(Float64, value))
        end
        if index == 1
            assign_str = "    " * key * " => " * value
        else
            assign_str = ",\n    " * key * " => " * value
        end
        dict_model_str["initialSpeciesValues"] *= assign_str
        index += 1
    end
    for (key, value) in model_dict["assignmentRulesStates"]
        if index != 1
            assign_str = ",\n    " * key * " => " * value
        else
            assign_str = "    " * key * " => " * value
            index += 1
        end
        dict_model_str["initialSpeciesValues"] *= assign_str
    end
    dict_model_str["initialSpeciesValues"] *= "\n"
    dict_model_str["initialSpeciesValues"] *= "    ]"
        
    for (index, (key, value)) in enumerate(model_dict["parameters"])
        if tryparse(Float64,value) !== nothing
            value = string(parse(Float64,value))
        end
        if index == 1
            assign_str = "    " * key * " => " * value
        else
            assign_str = ",\n    " * key * " => " * value
        end
        dict_model_str["trueParameterValues"] *= assign_str
    end
    dict_model_str["trueParameterValues"] *= "\n"
    dict_model_str["trueParameterValues"] *= "    ]"

    ### Writing to file
    model_name = replace(model_name, "-" => "_")
    io = IOBuffer()
    println(io, "function getODEModel_" * model_name * "(foo)")
    println(io, "\t# Model name: " * model_name)
    println(io, "\t# Number of parameters: " * model_dict["numOfParameters"])
    println(io, "\t# Number of species: " * model_dict["numOfSpecies"])
    println(io, "")

    println(io, "    ### Define independent and dependent variables")
    println(io, dict_model_str["variables"])
    println(io, "")
    println(io, "    ### Store dependent variables in array for ODESystem command")
    println(io, dict_model_str["stateArray"])
    println(io, "")
    println(io, "    ### Define variable parameters")
    println(io, dict_model_str["variableParameters"])
    println(io, "    ### Define potential algebraic variables")
    println(io, dict_model_str["algebraicVariables"])
    println(io, "    ### Define parameters")
    println(io, dict_model_str["parameters"])
    println(io, "")
    println(io, "    ### Store parameters in array for ODESystem command")
    println(io, dict_model_str["parameterArray"])
    println(io, "")
    println(io, "    ### Define an operator for the differentiation w.r.t. time")
    println(io, "    D = Differential(t)")
    println(io, "")
    println(io, "    ### Derivatives ###")
    println(io, dict_model_str["derivatives"])
    println(io, "")
    println(io, dict_model_str["ODESystem"])
    println(io, "")
    println(io, "    ### Initial species concentrations ###")
    println(io, dict_model_str["initialSpeciesValues"])
    println(io, "")
    println(io, "    ### SBML file parameter values ###")
    println(io, dict_model_str["trueParameterValues"])
    println(io, "")
    println(io, "    return sys, initialSpeciesValues, trueParameterValues")
    println(io, "")
    println(io, "end")
    model_str = String(take!(io))
    close(io)
    
    # In case user request file to be written 
    if write_to_file == true
        open(path_jl_file, "w") do f
            write(f, model_str)
        end
    end
    return model_str
end


function replace_rateOf(_formula::T, model_dict::Dict) where T<:Union{<:AbstractString, <:Real}

    formula = string(_formula)
    if !occursin("rateOf", formula)
        return formula
    end

    # Invalid character problems
    formula = replace(formula, "≤" => "<=")
    formula = replace(formula, "≥" => ">=")

    # Find rateof expressions 
    start_rateof = findall(i -> formula[i:(i+6)] == "rateOf(", 1:(length(formula)-6))
    end_rateof = [findfirst(x -> x == ')', formula[start:end])+start-1 for start in start_rateof]
    args = [formula[start_rateof[i]+7:end_rateof[i]-1] for i in eachindex(start_rateof)]
        
    replace_with = Vector{String}(undef, length(args))
    for (i, arg) in pairs(args)
        # A constant parameter does not have a rate 
        if arg ∈ keys(model_dict["parameters"])
            replace_with[i] = "0.0"
        end
        if is_number(arg)
            replace_with[i] = "0.0"
        end
        if arg ∈ keys(model_dict["states"]) || arg ∈ model_dict["rate_rules_names"]
            rate_change = model_dict["derivatives"][arg]
            replace_with[i] = rate_change[(findfirst(x -> x == '~', rate_change)+1):end]
        end
        if (arg ∈ keys(model_dict["states"]) && 
            model_dict["stateGivenInAmounts"][arg][1] == true && 
            !isempty(model_dict["stateGivenInAmounts"][arg][2] == true) &&
            model_dict["hasOnlySubstanceUnits"][arg] == false)

            replace_with[i] = "(" * replace_with[i] * ") / " * model_dict["stateGivenInAmounts"][arg][2]
        end
    end

    formula_cp = deepcopy(formula)
    for i in eachindex(replace_with)
        formula = replace(formula, formula_cp[start_rateof[i]:end_rateof[i]] => replace_with[i])
    end

    formula = replace(formula, "<=" => "≤")
    formula = replace(formula, ">=" => "≥")
    
    return formula
end


function replace_reactionid_with_math(formula::T, model_SBML)::T where T<:AbstractString
    for (reaction_id, reaction) in model_SBML.reactions
        reaction_math = parse_SBML_math(reaction.kinetic_math)
        formula = replace_variable(formula, reaction_id, reaction_math)
    end
    return formula
end
