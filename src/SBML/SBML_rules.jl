function parse_SBML_rules!(model_dict::Dict, model_SBML::SBML.Model)

    for rule in model_SBML.rules

        if rule isa SBML.AssignmentRule
            push!(model_dict["assignment_rule_variables"], rule.variable)
            parse_assignment_rule!(model_dict, rule, model_SBML)
        end

        if rule isa SBML.RateRule
            push!(model_dict["rate_rule_variables"], rule.variable)
            parse_rate_rule!(model_dict, rule, model_SBML)
        end

        if rule isa SBML.AlgebraicRule
            rule_formula = parse_SBML_math(rule.math)
            rule_formula = replace_variable(rule_formula, "time", "t")
            rule_formula = SBML_function_to_math(rule_formula, model_dict["SBML_functions"])
            rule_name = isempty(model_dict["algebraic_rules"]) ? "1" : maximum(keys(model_dict["algebraic_rules"])) * "1" # Need placeholder key 
            model_dict["algebraic_rules"][rule_name] = "0 ~ " * rule_formula
        end
    end    
end


function parse_assignment_rule!(model_dict::Dict, rule::SBML.AssignmentRule, model_SBML::SBML.Model)::Nothing

    rule_variable = rule.variable
    rule_formula = parse_SBML_math(rule.math)
    rule_formula = replace_variable(rule_formula, "time", "t")

    if isempty(rule_formula)
        rule_formula = "0.0"
    end

    # If piecewise occurs in the rule we need to unnest, rewrite to ifelse, special 
    # case handled separately 
    if occursin("piecewise(", rule_formula)
        rewrite_piecewise_to_ifelse(rule_formula, rule_variable, model_dict, model_SBML)
        if rule_variable ∈ keys(model_dict["derivatives"])
            delete!(model_dict["derivatives"], rule_variable)
        end
        return nothing
    end

    # Handle reaction ids, and unnest potential SBML functions 
    rule_formula = replace_reactionid_formula(rule_formula, model_SBML)
    rule_formula = process_SBML_str_formula(rule_formula, model_dict, model_SBML; check_scaling=true, rate_rule=false)

    # Check if variable affects a specie, parameter or compartment. In case of specie the assignment rule takes 
    # priority over any potential reactions later.
    if rule_variable ∈ keys(model_dict["species"])
        model_dict["species"][rule_variable].assignment_rule = true
        # If specie is given in amount account for the fact the that equation is given 
        # in conc. per SBML standard 
        _specie = model_dict["species"][rule_variable]
        if _specie.unit == :Amount && _specie.only_substance_units == false
            rule_formula = "(" * rule_formula * ")*" * _specie.compartment
        end
        _specie.formula = rule_formula
        if isempty(model_dict["species"][rule_variable].initial_value)
            _specie.initial_value = rule_formula
        end
        return nothing
    end
    if rule_variable ∈ keys(model_dict["parameters"])
        model_dict["parameters"][rule_variable].assignment_rule = true
        model_dict["parameters"][rule_variable].formula = rule_formula
        model_dict["parameters"][rule_variable].initial_value = rule_formula
        return nothing
    end
    if rule_variable ∈ keys(model_dict["compartments"])
        model_dict["compartments"][rule_variable].assignment_rule = true
        model_dict["compartments"][rule_variable].formula = rule_formula
        return nothing
    end

    # At this point the assignment rule create a new state or it is a speciesreference, 
    # to be used in stochiometry. If it has the name generatedId... it is related to 
    # StoichometryMath.
    if length(rule_variable) ≥ 11 && rule_variable[1:11] == "generatedId"
        model_dict["generated_ids"][rule_variable] = rule_formula
        return nothing
    end
    # Case with new specie 
    model_dict["species"][rule_variable] = SpecieSBML(rule_variable, false, false, rule_formula, 
                                                      rule_formula, 
                                                      collect(keys(model_SBML.compartments))[1], :Amount, 
                                                      false, true, false, false)

    return nothing
end


function parse_rate_rule!(model_dict::Dict, rule::SBML.RateRule, model_SBML::SBML.Model)::Nothing

    rule_variable = rule.variable
    rule_formula = parse_SBML_math(rule.math)
    rule_formula = replace_variable(rule_formula, "time", "t")    

    # Rewrite rule to function if there are not any piecewise, eles rewrite to formula with ifelse
    if occursin("piecewise(", rule_formula)
        rule_formula = rewrite_piecewise_to_ifelse(rule_formula, rule_variable, model_dict, model_SBML, ret_formula=true)
    else
        rule_formula = SBML_function_to_math(rule_formula, model_dict["SBML_functions"])
    end

    rule_formula = replace_reactionid_formula(rule_formula, model_SBML)
    rule_formula = process_SBML_str_formula(rule_formula, model_dict, model_SBML; check_scaling=false, rate_rule=true)
    # Adjust formula to be in conc. (if any specie is given in amount must divide with compartment)
    for (state_id, state) in model_dict["species"]
        if state.unit == :Concentration && state.only_substance_units == false
            continue
        end
        compartment = state.compartment
        rule_formula = replace_variable(rule_formula, state_id, "(" * state_id * "/" * compartment * ")")
    end

    if rule_variable ∈ keys(model_dict["species"])
        model_dict["species"][rule_variable].rate_rule = true
        # Must multiply with compartment if specie is given in amount as 
        # SBML formulas are given in conc. 
        _specie = model_dict["species"][rule_variable]
        if _specie.unit == :Amount && _specie.only_substance_units == false
            _specie.formula = "(" * rule_formula * ") * " * _specie.compartment
        else
            _specie.formula = rule_formula
        end

        return nothing
    end
    if rule_variable ∈ keys(model_dict["parameters"])
        model_dict["parameters"][rule_variable].rate_rule = true
        model_dict["parameters"][rule_variable].initial_value = model_dict["parameters"][rule_variable].formula
        model_dict["parameters"][rule_variable].formula = rule_formula        
        return nothing
    end
    if rule_variable ∈ keys(model_dict["compartments"])
        model_dict["compartments"][rule_variable].rate_rule = true
        model_dict["compartments"][rule_variable].initial_value = model_dict["compartments"][rule_variable].formula
        model_dict["compartments"][rule_variable].formula = rule_formula        
        return nothing
    end

    # At this state we introduce a new specie for said rule, this for example happens when we a 
    # special stoichometry        
    model_dict["species"][rule_variable] = SpecieSBML(rule_variable, false, false, "1.0", 
                                                      rule_formula, 
                                                      collect(keys(model_SBML.compartments))[1], :Amount, 
                                                      false, false, true, false)
    return nothing
end


function identify_algebraic_rule_variables!(model_dict::Dict)::Nothing

    # In case the model has algebraic rules some of the formulas (up to this point) are zero. To figure out 
    # which variable check which might be applicable
    if isempty(model_dict["algebraic_rules"])
        return nothing
    end

    candidates = String[]
    for (specie_id, specie) in model_dict["species"]
        
        if specie.rate_rule == true || specie.assignment_rule == true || specie.constant == true
            continue
        end
        if specie_id ∈ model_dict["appear_in_reactions"] && specie.boundary_condition == false
            continue
        end
        if !(specie.formula == "0.0" || isempty(specie.formula))
            continue
        end

        # Check if specie-id actually occurs in any algebraic rule
        should_continue::Bool = false
        for (rule_id, rule) in model_dict["algebraic_rules"]
            if replace_variable(rule, specie_id, "") != rule 
                should_continue = false
            end
        end
        should_continue == true && continue

        push!(candidates, specie_id)
    end
    # To be set as algebraic rule variable a soecue must not appear in reactions, 
    # or be a boundary condition. Sometimes several species can fulfill this for 
    # a single rule, in this case we choose the first valid 
    if !isempty(candidates)
        model_dict["species"][candidates[1]].algebraic_rule = true
    end


    for (parameter_id, parameter) in model_dict["parameters"]
        if parameter.rate_rule == true || parameter.assignment_rule == true || parameter.constant == true
            continue
        end

        # Check if specie-id actually occurs in any algebraic rule
        should_continue::Bool = false
        for (rule_id, rule) in model_dict["algebraic_rules"]
            if replace_variable(rule, parameter_id, "") != rule 
                should_continue = false
            end
        end
        should_continue == true && continue

        parameter.algebraic_rule = true
    end

    for (compartment_id, compartment) in model_dict["compartments"]
        if compartment.rate_rule == true || compartment.assignment_rule == true || compartment.constant == true
            continue
        end

        # Check if specie-id actually occurs in any algebraic rule
        should_continue::Bool = false
        for (rule_id, rule) in model_dict["algebraic_rules"]
            if replace_variable(rule, compartment_id, "") != rule 
                should_continue = false
            end
        end
        should_continue == true && continue

        compartment.algebraic_rule = true
    end

    return nothing
end


# Rewrites time-dependent ifElse-statements to depend on a boolean variable. This makes it possible to treat piecewise
# as events, allowing us to properly handle discontinious. Does not rewrite ifElse if the activation criteria depends
# on a state.
function time_dependent_ifelse_to_bool!(model_dict::Dict)

    # Rewrite piecewise using Boolean variables. Due to the abillity of piecewiese statements to be nested
    # recursion is needed.
    for key in keys(model_dict["inputFunctions"])
        formula_with_ifelse = model_dict["inputFunctions"][key]
        model_dict["inputFunctions"][key] = _time_dependent_ifelse_to_bool(string(formula_with_ifelse), model_dict, key)
    end
end


function _time_dependent_ifelse_to_bool(formula_with_ifelse::String, model_dict::Dict, key::String)::String

    formula_replaced = formula_with_ifelse

    index_ifelse = get_index_piecewise(formula_with_ifelse)
    if isempty(index_ifelse)
        return formula_replaced
    end

    for i in eachindex(index_ifelse)

        ifelse_formula = formula_with_ifelse[index_ifelse[i]][8:end-1]
        activationRule, left_side, right_side = split_ifelse(ifelse_formula)

        # Find inequality
        iLt = findfirst(x -> x == '<', activationRule)
        iGt = findfirst(x -> x == '>', activationRule)
        if isnothing(iGt) && !isnothing(iLt)
            sign_used = "lt"
            if activationRule[iLt:(iLt+1)] == "<="
                splitBy = "<="
            else
                splitBy = "<"
            end
        elseif !isnothing(iGt) && isnothing(iLt)
            sign_used = "gt"
            if activationRule[iGt:(iGt+1)] == ">="
                splitBy = ">="
            else
                splitBy = ">"
            end
        elseif occursin("!=", activationRule) || occursin("==", activationRule)
            rewrite_ifelse = false
            continue
        else
            println("Error : Did not find criteria to split ifelse on")
        end
        lhsRule, rhsRule = split(activationRule, string(splitBy))

        # Identify which side of ifelse expression is activated with time
        time_right = check_for_time(string(rhsRule))
        time_left = check_for_time(string(lhsRule))
        rewrite_ifelse = true
        if time_left == false && time_left == false
            @info "Have ifelse statements which does not contain time. Hence we do not rewrite as event, but rather keep it as an ifelse." maxlog=1
            rewrite_ifelse = false
            continue
        elseif time_left == true
            sign_time = check_sign_time(string(lhsRule))
            if (sign_time == 1 && sign_used == "lt") || (sign_time == -1 && sign_used == "gt")
                side_activated_with_time = "right"
            elseif (sign_time == 1 && sign_used == "gt") || (sign_time == -1 && sign_used == "lt")
                side_activated_with_time = "left"
            end
        elseif time_right == true
            sign_time = check_sign_time(string(rhsRule))
            if (sign_time == 1 && sign_used == "lt") || (sign_time == -1 && sign_used == "gt")
                side_activated_with_time = "left"
            elseif (sign_time == 1 && sign_used == "gt") || (sign_time == -1 && sign_used == "lt")
                side_activated_with_time = "right"
            end
        end

        # In case of nested ifelse rewrite left-hand and right-hand side
        left_side = _time_dependent_ifelse_to_bool(string(left_side), model_dict, key)
        right_side = _time_dependent_ifelse_to_bool(string(right_side), model_dict, key)

        if rewrite_ifelse == true
            j = 1
            local variable_name = ""
            while true
                variable_name = string(key) * "_bool" * string(j)
                if variable_name ∉ keys(model_dict["boolVariables"])
                    break
                end
                j += 1
            end
            activated_with_time = side_activated_with_time == "left" ? left_side : right_side
            deactivated_with_time = side_activated_with_time == "left" ? right_side : left_side
            formula_in_model = "((1 - " * variable_name * ")*" * "(" * deactivated_with_time *") + " * variable_name * "*(" * activated_with_time * "))"
            model_dict["parameters"][variable_name].formula = "0.0"
            formula_replaced = replace(formula_replaced, formula_with_ifelse[index_ifelse[i]] => formula_in_model)
            model_dict["boolVariables"][variable_name] = [activationRule, side_activated_with_time]
        end
    end

    return formula_replaced
end


# Here we assume we receive the arguments to ifelse(a ≤ 1 , b, c) on the form
# a ≤ 1, b, c and our goal is to return tuple(a ≤ 1, b, c)
function split_ifelse(str::String)
    paranthesis_level = 0
    split, i = 1, 1
    first_set, second_set, thirdSet = 1, 1, 1
    while i < length(str)

        if str[i] == '('
            paranthesis_level += 1
        elseif str[i] == ')'
            paranthesis_level -= 1
        end

        if str[i] == ',' && paranthesis_level == 0
            if split == 1
                first_set = 1:(i-1)
                split += 1
            elseif split == 2
                second_set = (first_set[end]+2):(i-1)
                thirdSet = (second_set[end]+2):length(str)
                break
            end
        end
        i += 1
    end
    return str[first_set], str[second_set], str[thirdSet]
end


function get_index_piecewise(str::String)

    ret = Array{Any, 1}(undef, 0)

    i_start, i_end = 0, 0
    i = 1
    while i < length(str)

        if !(length(str) > i+6)
            break
        end

        if str[i:(i+5)] == "ifelse"
            i_start = i
            paranthesis_level = 1
            for j in (i+7):length(str)
                if str[j] == '('
                    paranthesis_level += 1
                elseif str[j] == ')'
                    paranthesis_level -= 1
                end
                if paranthesis_level == 0
                    i_end = j
                    break
                end
            end
            ret = push!(ret, collect(i_start:i_end))
            i = i_end + 1
            continue
        end
        i += 1
    end

    return ret
end
