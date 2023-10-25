function extract_rule_formula(rule)
    _rule_formula = parse_SBML_math(rule.math)
    rule_formula = replace_variable(_rule_formula, "time", "t")
    return rule_formula
end


function process_assignment_rule!(model_dict::Dict, rule_formula::String, rule_variable::String, model_SBML)

    # If piecewise occurs in the rule we are looking at a time-based event which is encoded as an
    # event into the model to ensure proper evaluation of the gradient.
    if occursin("piecewise(", rule_formula)
        rewrite_piecewise_to_ifelse(rule_formula, rule_variable, model_dict, model_SBML)
        if rule_variable ∈ keys(model_dict["derivatives"])
            delete!(model_dict["derivatives"], rule_variable)
        end
        return 
    end

    rule_formula = replace_reactionid_with_math(rule_formula, model_SBML)

    #=
        If the rule does not involve a piecewise expression simply encode it as a function which downsteram
        is integrated into the equations when inserting "functions" into the reactions 
    =#
    # Extract the parameters and states which make out the rule, if the rule nests another function 
    # function is written to math 
    rule_formula = SBML_function_to_math(rule_formula, model_dict["modelFunctions"])

    if rule_variable in keys(model_dict["states"])
        model_dict["assignmentRulesStates"][rule_variable] = rule_formula
        # Delete from state dictionary (as we no longer should assign an initial value to the state)
        delete!(model_dict["states"], rule_variable)
    end
    if rule_variable ∈ keys(model_SBML.compartments)
        # In case compartment is assigned we need to track assignment formula 
        model_dict["compartment_formula"][rule_variable] = rule_formula
    end

    # At this point the assignment rule create a new state via a speciesreference, 
    # should thus be added to species list
    if !(rule_variable ∈ keys(model_dict["states"]) || 
         rule_variable ∈ keys(model_SBML.compartments) || 
         rule_variable ∈ keys(model_SBML.parameters) ||
         rule_variable ∈ keys(model_dict["assignmentRulesStates"]))

        model_dict["states"][rule_variable] = rule_formula
        model_dict["stateGivenInAmounts"][rule_variable] = (true, collect(keys(model_SBML.compartments))[1])
        model_dict["hasOnlySubstanceUnits"][rule_variable] =  false 
        model_dict["isBoundaryCondition"][rule_variable] = false
        model_dict["derivatives"][rule_variable] = "D(" * rule_variable * ") ~ "        
    end

    return 
end


function process_rate_rule!(model_dict::Dict, rule_formula::String, rule_variable::String, model_SBML)

    # Rewrite rule to function if there are not any piecewise, eles rewrite to formula with ifelse
    if occursin("piecewise(", rule_formula)
        rule_formula = rewrite_piecewise_to_ifelse(rule_formula, rule_variable, model_dict, model_SBML, ret_formula=true)
    else
        rule_formula = SBML_function_to_math(rule_formula, model_dict["modelFunctions"])
    end

    rule_formula = replace_reactionid_with_math(rule_formula, model_SBML)

    # Add rate rule as part of model derivatives and remove from parameters dict if rate rule variable
    # is a parameter
    if rule_variable ∈ keys(model_dict["parameters"])
        model_dict["derivatives"][rule_variable] = "D(" * rule_variable * ") ~ " * rule_formula
        model_dict["states"][rule_variable] = model_dict["parameters"][rule_variable]
        model_dict["stateGivenInAmounts"][rule_variable] = (false, "")
        model_dict["hasOnlySubstanceUnits"][rule_variable] = false
        delete!(model_dict["parameters"], rule_variable)
        model_dict["derivatives"][rule_variable] = "D(" * rule_variable * ") ~ " * rule_formula
    elseif rule_variable ∈ keys(model_dict["states"])
        # Paranthesis needed to downstream add compartment to scale conc. properly to correct unit for the 
        # state given by a rate-rule.
        for (state_id, state) in model_SBML.species
            if !(model_dict["stateGivenInAmounts"][state_id][1] == true && model_dict["hasOnlySubstanceUnits"][state_id] == false)
                continue
            end
            compartment = state.compartment
            rule_formula = replace_variable(rule_formula, state_id, "(" * state_id * "/" * compartment * ")")
        end
        model_dict["derivatives"][rule_variable] = "D(" * rule_variable * ") ~ " * "(" * rule_formula * ")"
    else
        @error "Warning : Cannot find rate rule variable in either model states or parameters"
    end
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
            model_dict["parameters"][variable_name] = "0.0"
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
