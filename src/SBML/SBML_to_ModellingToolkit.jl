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


# Rewrites triggers in events to propper form for ModelingToolkit
function as_trigger(trigger_formula, model_dict, model_SBML)

    if trigger_formula[1] == '(' && trigger_formula[end] == ')'
        trigger_formula = trigger_formula[2:end-1]
    end

    if occursin(r"<|≤|>|≥", trigger_formula)
        trigger_formula = replace_whole_word(trigger_formula, "time", "t")
        trigger_formula = replace(trigger_formula, "<" => "≤")
        trigger_formula = replace(trigger_formula, ">" => "≥")
        for (species_id, specie) in model_SBML.species
            if species_id ∈ keys(model_SBML.species) && model_dict["stateGivenInAmounts"][species_id][1] == false
                continue
            end
            if occursin("rateOf", trigger_formula)
                continue
            end
            trigger_formula = replace_whole_word(trigger_formula, species_id, species_id * '/' * specie.compartment)
        end
        return trigger_formula
    end

    if typeof(trigger_formula) <: Real || is_number(trigger_formula)
        return string(trigger_formula) * " != 0"
    end

    if "geq" == trigger_formula[1:3]
        stripped_formula = trigger_formula[5:end-1]
        separator_use = "≥"
    elseif "gt" == trigger_formula[1:2]
        stripped_formula = trigger_formula[4:end-1]
        separator_use = "≥"
    elseif "leq" == trigger_formula[1:3]
        stripped_formula = trigger_formula[5:end-1]
        separator_use = "≤"
    elseif "lt" == trigger_formula[1:2]
        stripped_formula = trigger_formula[4:end-1]
        separator_use = "≤"
    end
    parts = split_between(stripped_formula, ',')
    if occursin("time", parts[1])
        parts[1] = replace_whole_word(parts[1], "time", "t")
    end

    # Account for potential reaction-math kinetics making out the trigger 
    parts[1] = replace_reactionid_with_math(parts[1], model_SBML)
    parts[2] = replace_reactionid_with_math(parts[2], model_SBML)

    # States in ODE-system are typically in substance units, but formulas in 
    # concentratio. Thus, each state is divided with its corresponding 
    # compartment 
    for (species_id, specie) in model_SBML.species
        if species_id ∈ keys(model_SBML.species) && model_dict["stateGivenInAmounts"][species_id][1] == false
            continue
        end
        if occursin("rateOf", parts[1]) || occursin("rateOf", parts[2])
            continue
        end
        parts[1] = replace_whole_word(parts[1], species_id, species_id * '/' * specie.compartment)
        parts[2] = replace_whole_word(parts[2], species_id, species_id * '/' * specie.compartment)
    end

    return parts[1] * " " * separator_use * " " * parts[2] 
end


# Rewrites derivatives for a model by replacing functions, any lagging piecewise, and power functions.
function rewrite_derivatives(derivative_str, model_dict, base_functions, model_SBML; check_scaling=false)
    
    new_derivative_str = replace_function_with_formula(derivative_str, model_dict["modelFunctions"])
    new_derivative_str = replace_function_with_formula(new_derivative_str, model_dict["modelRuleFunctions"])
    
    if occursin("pow(", new_derivative_str)
        new_derivative_str = remove_power_functions(new_derivative_str)
    end
    if occursin("piecewise(", new_derivative_str)
        new_derivative_str = rewrite_piecewise_to_ifelse(new_derivative_str, "foo", model_dict, base_functions, model_SBML, ret_formula=true)
    end

    new_derivative_str = replace_whole_word_dict(new_derivative_str, model_dict["modelFunctions"])
    new_derivative_str = replace_whole_word_dict(new_derivative_str, model_dict["modelRuleFunctions"])

    if check_scaling == false
        return new_derivative_str
    end

    # Handle case when specie is given in amount, but the equations are given in concentration 
    for (state_id, state) in model_SBML.species
        if model_dict["stateGivenInAmounts"][state_id][1] == true && model_dict["hasOnlySubstanceUnits"][state_id] == false
            compartment = state.compartment
            new_derivative_str = replace_whole_word(new_derivative_str, state_id, "(" * state_id * "/" * compartment * ")")
        end
    end

    # Handle that in SBML models sometimes t is decoded as time
    new_derivative_str = replace_whole_word(new_derivative_str, "time", "t")

    return new_derivative_str
end


function process_initial_assignment(model_SBML, model_dict::Dict, base_functions::Array{String, 1})

    initally_assigned_variable = Dict{String, String}()
    initally_assigned_parameter = Dict{String, String}()
    for (assignId, initialAssignment) in model_SBML.initial_assignments
        
        # The formula might refer to reaciton, in this case insert reaction dynamics 
        _formula = SBML_math_to_str(initialAssignment)
        _formula = replace_reactionid_with_math(_formula, model_SBML)
        
        formula = rewrite_derivatives(_formula, model_dict, base_functions, model_SBML)
        # Initial time i zero 
        formula = replace_whole_word(formula, "t", "0.0")

        # Figure out wheter parameters or state is affected by the initial assignment
        if assignId ∈ keys(model_dict["states"])
            model_dict["states"][assignId] = formula
            initally_assigned_variable[assignId] = "states"

        elseif assignId ∈ keys(model_dict["nonConstantParameters"])
            model_dict["nonConstantParameters"][assignId] = formula
            initally_assigned_variable[assignId] = "nonConstantParameters"

        elseif assignId ∈ keys(model_dict["parameters"])
            model_dict["parameters"][assignId] = formula
            initally_assigned_variable[assignId] = "parameters"

        # At this point the assignment should be accepted as a state in the model, 
        # and thus has to be added             
        else
            model_dict["states"][assignId] = formula
            model_dict["stateGivenInAmounts"][assignId] = (true, collect(keys(model_SBML.compartments))[1])
            model_dict["hasOnlySubstanceUnits"][assignId] =  false 
            model_dict["isBoundaryCondition"][assignId] = false
            model_dict["derivatives"][assignId] = "D(" * assignId * ") ~ "
        end        
    end

    # If the initial assignment for a state is the value of another state apply recursion until continue looping
    # until we have the initial assignment expressed as non state variables
    while true
        # Skip if rateOf occurs as this is processed longer down when derivatives are done 
        nested_variables = false
        for (variable, dictName) in initally_assigned_variable
            if dictName == "states"
                variable_value = model_dict["states"][variable]
                if occursin("rateOf", variable_value)
                    continue
                end
                args = split(get_arguments(variable_value, base_functions))
                for arg in args
                    if arg in keys(model_dict["states"])
                        nested_variables = true
                        variable_value = replace_whole_word(variable_value, arg, model_dict["states"][arg])
                    end
                end
                model_dict["states"][variable] = variable_value
            end
        end
        nested_variables || break
    end

    # If the initial assignment for a parameters is the value of another parameters apply recursion
    # until we have the initial assignment expressed as non parameters
    while true
        nested_parameter = false
        for (parameter, dictName) in initally_assigned_parameter
            parameter_value = model_dict["parameters"][parameter]
            args = split(get_arguments(parameter_value, base_functions))
            for arg in args
                if arg in keys(model_dict["parameters"])
                    nested_parameter = true
                    parameter_value = replace_whole_word(parameter_value, arg, model_dict["parameters"][arg])
                end
            end
            model_dict["parameters"][parameter] = parameter_value
        end
        nested_parameter || break
    end

    # Lastly, if initial assignment refers to a state we need to scale with compartment 
    for id in keys(initally_assigned_variable)
        if id ∉ keys(model_SBML.species)
            continue
        end
        if isnothing(model_SBML.species[id].substance_units)
            continue
        end
        # We end up here 
        if !(any([val[1] for val in  values(model_dict["stateGivenInAmounts"])]) == true)
            continue
        end
        if model_SBML.species[id].substance_units == "substance"
            model_dict["stateGivenInAmounts"][id] = (true, model_SBML.species[id].compartment)
        end
        model_dict["states"][id] = '(' * model_dict["states"][id] * ") * " * model_SBML.species[id].compartment
    end
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
    model_dict["nonConstantParameters"] = Dict()
    model_dict["modelFunctions"] = Dict()
    model_dict["modelRuleFunctions"] = Dict()
    model_dict["modelRules"] = Dict()
    model_dict["derivatives"] = Dict()
    model_dict["eventDict"] = Dict()
    model_dict["discreteEventDict"] = Dict()
    model_dict["inputFunctions"] = Dict()
    model_dict["stringOfEvents"] = Dict()
    model_dict["numOfParameters"] = Dict()
    model_dict["numOfSpecies"] = Dict()
    model_dict["boolVariables"] = Dict()
    model_dict["events"] = Dict()
    model_dict["reactions"] = Dict()
    model_dict["algebraicRules"] = Dict()
    model_dict["assignmentRulesStates"] = Dict()
    model_dict["compartment_formula"] = Dict()
    # Mathemathical base functions (can be expanded if needed)
    base_functions = ["exp", "log", "log2", "log10", "sin", "cos", "tan", "pi"]

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
    non_constant_parameter_names = []
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
        function_formula = SBML_math_to_str(SBML_function.body.body, true)
        model_dict["modelFunctions"][function_name] = [args, function_formula]
    end

    # Later by the process callback function these events are rewritten to 
    # DiscreteCallback:s if possible 
    e_index = 1
    for (event_name, event) in model_SBML.events
        _trigger_formula = replace_function_with_formula(SBML_math_to_str(event.trigger.math), model_dict["modelFunctions"])
        if isnothing(_trigger_formula)
            continue
        end
        trigger_formula = as_trigger(_trigger_formula, model_dict, model_SBML)
        event_formulas = Vector{String}(undef, length(event.event_assignments))
        event_assign_to = similar(event_formulas)
        not_skip_assignments = fill(true, length(event.event_assignments))
        for (i, event_assignment) in pairs(event.event_assignments)

            # Check if event should be ignored as no math child was provided 
            if isnothing(SBML_math_to_str(event_assignment.math))
                not_skip_assignments[i] = false
                continue
            end

            event_assign_to[i] = event_assignment.variable
            event_formulas[i] = replace_function_with_formula(SBML_math_to_str(event_assignment.math), model_dict["modelFunctions"])
            event_formulas[i] = replace_whole_word(event_formulas[i], "t", "integrator.t")
            event_formulas[i] = replace_reactionid_with_math(event_formulas[i], model_SBML)

            # Species typically given in substance units, but formulas in conc. Thus we must account for assignment 
            # formula being in conc., but we are changing something by amount 
            if event_assign_to[i] ∈ keys(model_SBML.species)
                if event_assign_to[i] ∈ keys(model_SBML.species) && model_dict["stateGivenInAmounts"][event_assign_to[i]][1] == false
                    continue
                end
                event_formulas[i] = model_SBML.species[event_assign_to[i]].compartment *  " * (" * event_formulas[i] * ')'
                continue
            end
            if event_assign_to[i] ∈ keys(model_SBML.parameters)
                continue
            end
            if event_assign_to[i] ∈ non_constant_parameter_names
                continue
            end
            # At this state the assign variable should be treated as state that for the first time is introduced 
            # here, and takes the value 1 prior to being event assigned 
            model_dict["states"][event_assign_to[i]] = 1.0
            model_dict["stateGivenInAmounts"][event_assign_to[i]] = (true, collect(keys(model_SBML.compartments))[1])
            model_dict["hasOnlySubstanceUnits"][event_assign_to[i]] =  false 
            model_dict["isBoundaryCondition"][event_assign_to[i]] = false
            model_dict["derivatives"][event_assign_to[i]] = "D(" * event_assign_to[i] * ") ~ "   
        end

        # Handle special case where an event assignment acts on a compartment, and a species with said 
        # compartment 
        __event_assign_to = deepcopy(event_assign_to)
        for (i, assign_to) in pairs(__event_assign_to[not_skip_assignments])
            if assign_to ∉ keys(model_SBML.compartments)
                continue
            end
            for (specie_id, specie) in model_SBML.species
                if specie.compartment != assign_to
                    continue
                end
                if model_dict["stateGivenInAmounts"][specie_id][1] == true
                    continue
                end
                if specie_id ∈ event_assign_to
                    is = findfirst(x -> x == specie_id, event_assign_to)
                    event_formulas[is] = "(" * event_formulas[is] * ")" * "*" * assign_to * "/" * "(" * event_formulas[i] * ")"
                    continue
                end
                _assign_to = specie_id
                # assign_to = compartment
                _formula = specie_id * "*" * assign_to * "/" * "(" * event_formulas[i] * ")"
                event_assign_to = vcat(event_assign_to, _assign_to)
                event_formulas = vcat(event_formulas, _formula)
                not_skip_assignments = vcat(not_skip_assignments, true)
            end
        end

        event_name = isempty(event_name) ? "event" * string(e_index) : event_name
        model_dict["events"][event_name] = [trigger_formula, event_assign_to[not_skip_assignments] .* " = " .* event_formulas[not_skip_assignments], event.trigger.initial_value]
        e_index += 1
    end

    assignment_rules_names = []
    rate_rules_names = []
    for rule in model_SBML.rules
        if rule isa SBML.AssignmentRule
            rule_formula = extract_rule_formula(rule)
            assignment_rules_names = push!(assignment_rules_names, rule.variable)
            process_assignment_rule!(model_dict, rule_formula, rule.variable, base_functions, model_SBML)
        end

        if rule isa SBML.RateRule
            rule_formula = extract_rule_formula(rule)
            rate_rules_names = push!(rate_rules_names, rule.variable)
            process_rate_rule!(model_dict, rule_formula, rule.variable, model_SBML, base_functions)
        end

        if rule isa SBML.AlgebraicRule
            _rule_formula = extract_rule_formula(rule)
            rule_formula = replace_function_with_formula(_rule_formula, model_dict["modelFunctions"])
            rule_name = isempty(model_dict["algebraicRules"]) ? "1" : maximum(keys(model_dict["algebraicRules"])) * "1" # Need placeholder key 
            model_dict["algebraicRules"][rule_name] = "0 ~ " * rule_formula
        end
    end

    # In case we have that the compartment is given by an assignment rule, then we need to account for this 
    for (compartment_id, compartment_formula) in model_dict["compartment_formula"]
        for (eventId, event) in model_dict["events"]
            trigger_formula = event[1]
            event_assignments = event[2]
            trigger_formula = replace_whole_word(trigger_formula, compartment_id, compartment_formula)
            for i in eachindex(event_assignments)
                event_assignments[i] = replace_whole_word(event_assignments[i], compartment_id, compartment_formula)
            end
            model_dict["events"][eventId] = [trigger_formula, event_assignments, event[3]]
        end
    end

    # Positioned after rules since some assignments may include functions
    process_initial_assignment(model_SBML, model_dict, base_functions)

    # Process chemical reactions 
    for (id, reaction) in model_SBML.reactions
        # Process kinetic math into Julia syntax 
        _formula = SBML_math_to_str(reaction.kinetic_math)
               
        # Add values for potential kinetic parameters (where-statements)
        for (parameter_id, parameter) in reaction.kinetic_parameters
            _formula = replace_whole_word(_formula, parameter_id, parameter.value)
        end

        formula = rewrite_derivatives(_formula, model_dict, base_functions, model_SBML, check_scaling=true)
        model_dict["reactions"][reaction.name] = formula
        
        for reactant in reaction.reactants
            model_dict["isBoundaryCondition"][reactant.species] == true && continue # Constant state  
            compartment = model_SBML.species[reactant.species].compartment
            stoichiometry = isnothing(reactant.stoichiometry) ? "1" : string(reactant.stoichiometry)
            stoichiometry = stoichiometry[1] == '-' ? "(" * stoichiometry * ")" : stoichiometry
            compartment_scaling = model_dict["hasOnlySubstanceUnits"][reactant.species] == true ? " * " : " * ( 1 /" * compartment * " ) * "
            model_dict["derivatives"][reactant.species] *= "-" * stoichiometry * compartment_scaling * "(" * formula * ")"
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
            compartment_scaling = model_dict["hasOnlySubstanceUnits"][product.species] == true ? " * " : " * ( 1 /" * compartment * " ) * "
            model_dict["derivatives"][product.species] *= "+" * stoichiometry * compartment_scaling * "(" * formula * ")"
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
            _formula = rewrite_derivatives(formula, model_dict, base_functions, model_SBML; check_scaling=true)
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
            if replace_whole_word(du, pars, "") !== du
                is_in_ode[i] = true
            end
        end
    end
    for input_function in values(model_dict["inputFunctions"])
        for (i, pars) in enumerate(keys(model_dict["parameters"]))
            if replace_whole_word(input_function, pars, "") !== input_function
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
            if model_dict["isBoundaryCondition"][species] == true && model_SBML.species[species].constant == true
                continue
            end

            # Check if state occurs in any of the algebraic rules 
            for (rule_id, rule) in model_dict["algebraicRules"]
                if replace_whole_word(rule, species, "") != rule 
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

    if length(model_dict["nonConstantParameters"]) > 0
        dict_model_str["variableParameters"] = "    ModelingToolkit.@variables"
        for key in keys(model_dict["nonConstantParameters"])
            dict_model_str["variableParameters"] *= " " * key * "(t)"
        end
    end
        
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
    for key in keys(model_dict["nonConstantParameters"])
        if s_index != 1
            dict_model_str["derivatives"] *= ",\n    D(" * key * ") ~ 0"
        else
            dict_model_str["derivatives"] *= ",    D(" * key * ") ~ 0"
            s_index += 1
        end
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
    for (key, value) in model_dict["nonConstantParameters"]
        if index != 1
            assign_str = ",\n    " * key * " => " * value
        else
            assign_str = "    " * key * " => " * value
            index += 1
        end
        dict_model_str["initialSpeciesValues"] *= assign_str
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


function SBML_math_to_str(math)
    math_str, _ = _SBML_math_to_str(math)
    return math_str
end
function SBML_math_to_str(math::SBML.MathApply, inequality_to_julia::Bool)
    math_str, _ = _SBML_math_to_str(math; inequality_to_julia=inequality_to_julia)
    return math_str
end
function SBML_math_to_str(math, inequality_to_julia::Bool)
    math_str, _ = _SBML_math_to_str(math)
    return math_str
end
function SBML_math_to_str(math::Nothing)
    return nothing
end


function _SBML_math_to_str(math::SBML.MathApply; inequality_to_julia::Bool=false)

    if math.fn ∈ ["*", "/", "+", "-", "power"] && length(math.args) == 2
        fn = math.fn == "power" ? "^" : math.fn
        _part1, add_parenthesis1 = _SBML_math_to_str(math.args[1])
        _part2, add_parenthesis2 = _SBML_math_to_str(math.args[2])
        # In case we hit the bottom in the recursion we do not need to add paranthesis 
        # around the math-expression making the equations easier to read
        part1 = add_parenthesis1 ?  '(' * _part1 * ')' : _part1
        part2 = add_parenthesis2 ?  '(' * _part2 * ')' : _part2
        return part1 * fn * part2, true
    end

    if math.fn == "log" && length(math.args) == 2
        base, add_parenthesis1 = _SBML_math_to_str(math.args[1])
        arg, add_parenthesis2 = _SBML_math_to_str(math.args[2])
        part1 = add_parenthesis1 ?  '(' * base * ')' : base
        part2 = add_parenthesis2 ?  '(' * arg * ')' : arg
        return "log(" * part1 * ", " * part2 * ")", true
    end


    if math.fn == "root" && length(math.args) == 2
        base, add_parenthesis1 = _SBML_math_to_str(math.args[1])
        arg, add_parenthesis2 = _SBML_math_to_str(math.args[2])
        part1 = add_parenthesis1 ?  '(' * base * ')' : base
        part2 = add_parenthesis2 ?  '(' * arg * ')' : arg
        return  part2 * "^(1 / " * part1 * ")", true
    end

    if math.fn ∈ ["+", "-"] && length(math.args) == 1
        _formula, add_parenthesis = _SBML_math_to_str(math.args[1])
        formula = add_parenthesis ? '(' * _formula * ')' : _formula
        return math.fn * formula, true
    end

    if math.fn == "quotient" 
        arg1, _ = _SBML_math_to_str(math.args[1])
        arg2, _ = _SBML_math_to_str(math.args[2])
        return "div(" * arg1 * ", " * arg2 * ")", false
    end

    # Piecewise can have arbibrary number of arguments 
    if math.fn == "piecewise"
        formula = "piecewise("
        for arg in math.args
            _formula, _ = _SBML_math_to_str(arg) 
            formula *= _formula * ", "
        end
        return formula[1:end-2] * ')', false
    end

    if math.fn ∈ ["lt", "gt", "leq", "geq", "eq"] && inequality_to_julia == false
        @assert length(math.args) == 2
        part1, _ = _SBML_math_to_str(math.args[1]) 
        part2, _ = _SBML_math_to_str(math.args[2])
        return math.fn * "(" * part1 * ", " * part2 * ')', false
    end

    if math.fn ∈ ["lt", "gt", "leq", "geq", "eq"] && inequality_to_julia == true
        @assert length(math.args) == 2
        part1, _ = _SBML_math_to_str(math.args[1]) 
        part2, _ = _SBML_math_to_str(math.args[2])
        if math.fn == "lt"
            operator = "<"
        elseif math.fn == "gt"
            operator = ">"
        elseif math.fn == "geq"
            operator = "≥"
        elseif math.fn == "leq"
            operator = "≤"
        elseif math.fn == "eq"
            operator = "=="
        end

        return "(" * part1 * operator * part2 * ')', false
    end

    if math.fn ∈ ["exp", "log", "log2", "log10", "sin", "cos", "tan"]
        @assert length(math.args) == 1
        formula, _ = _SBML_math_to_str(math.args[1])
        return math.fn * '(' * formula * ')', false
    end

    if math.fn ∈ ["arctan", "arcsin", "arccos", "arcsec", "arctanh", "arcsinh", "arccosh", 
                  "arccsc", "arcsech", "arccoth", "arccot", "arccot", "arccsch"]
        @assert length(math.args) == 1
        formula, _ = _SBML_math_to_str(math.args[1])
        return "a" * math.fn[4:end] * '(' * formula * ')', false
    end

    if math.fn ∈ ["exp", "log", "log2", "log10", "sin", "cos", "tan", "csc", "ln"]
        fn = math.fn == "ln" ? "log" : math.fn
        @assert length(math.args) == 1
        formula, _ = _SBML_math_to_str(math.args[1])
        return fn * '(' * formula * ')', false
    end

    # Special function which must be rewritten to Julia syntax 
    if math.fn == "ceiling"
        formula, _ = _SBML_math_to_str(math.args[1])
        return "ceil" * '(' * formula * ')', false
    end

    # Factorials are, naturally, very challenging for ODE solvers. In case against the odds they 
    # are provided we compute the factorial via the gamma-function (to handle Num type). 
    if math.fn == "factorial"
        @warn "Factorial in the ODE model. PEtab.jl can handle factorials, but, solving the ODEs with factorial is 
            numerically challenging, and thus if possible should be avioded"
        formula, _ = _SBML_math_to_str(math.args[1])
        return "SpecialFunctions.gamma" * '(' * formula * " + 1.0)", false
    end

    # At this point the only feasible option left is a SBML_function
    formula = math.fn * '('
    if length(math.args) == 0
        return formula * ')', false
    end
    for arg in math.args
        _formula, _ = _SBML_math_to_str(arg) 
        formula *= _formula * ", "
    end
    return formula[1:end-2] * ')', false
end
function _SBML_math_to_str(math::SBML.MathVal)
    return string(math.val), false
end
function _SBML_math_to_str(math::SBML.MathIdent)
    return string(math.id), false
end
function _SBML_math_to_str(math::SBML.MathTime)
    # Time unit is consistently in models refered to as time 
    return "t", false
end
function _SBML_math_to_str(math::SBML.MathAvogadro)
    # Time unit is consistently in models refered to as time 
    return "6.02214179e23", false
end
function _SBML_math_to_str(math::SBML.MathConst)
    if math.id == "exponentiale"
        return "2.718281828459045", false
    elseif math.id == "pi"
        return "3.1415926535897", false
    else
        return math.id, false
    end
end


function replace_reactionid_with_math(formula::T, model_SBML)::T where T<:AbstractString
    for (reaction_id, reaction) in model_SBML.reactions
        reaction_math = SBML_math_to_str(reaction.kinetic_math)
        formula = replace_whole_word(formula, reaction_id, reaction_math)
    end
    return formula
end
