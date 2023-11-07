#=
    Functionality for correctly handling SBML piecewise statements and if 
    possible, rewrite to discrete callbacks 
=#


# Handles piecewise functions that are to be redefined with ifelse speciements in the model
# equations to allow MKT symbolic calculations.
function piecewise_to_ifelse(rule_formula, model_dict, model_SBML)

    piecewise_eqs = extract_x(rule_formula, "piecewise")
    ifelse_eqs = Vector{String}(undef, length(piecewise_eqs))

    for (i, piecewise_eq) in pairs(piecewise_eqs)

        # Extract components
        _piecewise_eq = piecewise_eq[11:end-1] # Everything inside paranthesis 
        _args = split_between(_piecewise_eq, ',')
        values = _args[1:2:end]
        conditions = _args[2:2:end]

        if length(conditions) > 1
            @warn "Potentially breaking example with multiple conditions"
        end

        # Process inactive and active value, apply recursion if nested 
        if occursin("piecewise(", values[1])
            value_active = piecewise_to_ifelse(values[1], model_dict, model_SBML)
            value_active = process_SBML_str_formula(value_active, model_dict, model_SBML)
        else
            value_active = process_SBML_str_formula(values[1], model_dict, model_SBML)
        end
        if occursin("piecewise(", values[end])
            value_inactive = piecewise_to_ifelse(values[end], model_dict, model_SBML)
            value_inactive = process_SBML_str_formula(value_inactive, model_dict, model_SBML)
        else
            value_inactive = process_SBML_str_formula(values[end], model_dict, model_SBML)
        end

        # How to formula the ifelse depends on the condition. The condition can be direct (gt, lt...), a gate 
        # (and, if, or...) or a nested piecewise. Each case is handled separately
        condition = conditions[1]
        if condition[1:2] ∈ ["lt", "gt", "eq"] || condition[1:3] ∈ ["neq", "geq", "leq"] 
            ifelse_eqs[i] = parse_piecewise_bool_condition(condition, value_active, value_inactive)

        elseif condition[1:2] ∈ ["if", "or"] || condition[1:3] ∈ ["and", "xor", "not"]
            ifelse_eqs[i] = parse_piecewise_gate_condition(condition, value_active, value_inactive)

        # Condition can be nested, in this case apply recursion         
        elseif length(condition) ≥ 9 && condition[1:9] == "piecewise"
            _condition = piecewise_to_ifelse(condition, model_dict, model_SBML)
            _condition = process_SBML_str_formula(_condition, model_dict, model_SBML)
            ifelse_eqs[i] = parse_piecewise_bool_condition(_condition, value_active, value_inactive)
        
        else
            @error "We cannot process the piecewise expression, condition = $condition"
        end
    end

    # Finally replace piecewise with ifelese
    formula_ret = deepcopy(rule_formula)
    for i in eachindex(piecewise_eqs)
        formula_ret = replace(formula_ret, piecewise_eqs[i] => ifelse_eqs[i])
    end
    return formula_ret
end


function extract_x(formula::String, x::String; retindex::Bool=false)::Union{Vector{String}, Vector{UnitRange}}

    # Find number of piecewise expressions, and where they start 
    index_piecewise_start = findall(x * "(", formula)
    n_x = length(index_piecewise_start)
    xs::Vector{String} = String[]
    ixs::Vector{UnitRange} = UnitRange[]

    # Extract entire piecewise expression while handling inner paranthesis, e.g
    # when we have "2*piecewise(0, lt(t - insulin_time_1, 0), 1)" it extracts
    # the full expression piecewise(0, lt(t - insulin_time_1, 0), 1) while coutning 
    # the inner (but ignoring it as it is handled via recursion)
    i, k = 1, 1
    while i ≤ n_x
        n_inner_paranthesis = 0
        i_end = index_piecewise_start[i][end]
        while true
            i_end += 1
            if n_inner_paranthesis == 0 && formula[i_end] == ')'
                break
            end

            n_inner_paranthesis = formula[i_end] == '(' ? n_inner_paranthesis+1 : n_inner_paranthesis
            n_inner_paranthesis = formula[i_end] == ')' ? n_inner_paranthesis-1 : n_inner_paranthesis
        end

        push!(xs, formula[index_piecewise_start[i][1]:i_end])
        push!(ixs, index_piecewise_start[i][1]:i_end)

        # Check how many picewise have been processed
        i += length(findall(x * "(", xs[k]))
        k += 1
    end

    if retindex == false
        return xs
    else
        return ixs
    end
end


function parse_piecewise_bool_condition(condition::String, value_active::String, value_inactive::String)::String

    nested_condition::Bool = false
    if "leq" == condition[1:3]
        condition_components = condition[5:end-1]
        comparison_operator = " <= "
    elseif "lt" == condition[1:2]
        condition_components = condition[4:end-1]
        comparison_operator = " < "
    elseif "geq" == condition[1:3]
        condition_components = condition[5:end-1]
        comparison_operator = " >= "
    elseif "gt" == condition[1:2]
        condition_components = condition[4:end-1]
        comparison_operator = " > "
    elseif "eq" == condition[1:2]
        condition_components = condition[4:end-1]
        comparison_operator = " == "        
    elseif "neq" == condition[1:3]
        condition_components = condition[5:end-1]
        comparison_operator = " != "
    # Edge case, but condition is allowed to simply be true (albeit stupied)          
    elseif condition[1:4] == "true" 
        return "true"
    # As above
    elseif condition[1:5] == "false"
        return "false"
    # Can happen in more complex piecewise
    elseif "ifelse" == condition[1:6]
        nested_condition = true
    else
        @error "Cannot recognize inequality condition = $condition in piecewise"
    end

    if nested_condition == true
        return "ifelse(" * condition * ", " * value_active * ", " * value_inactive * ")"
    end
    
    parts = split_between(condition_components, ',')
    return "ifelse(" * parts[1] * comparison_operator * parts[2] * ", " * value_active * ", " * value_inactive * ")"
end


function parse_piecewise_gate_condition(condition, value_active, value_inactive)
    
    event_str = _parse_piecewise_gate_condition(condition)

    return event_str * " * (" * value_active * ") + (1 - " * event_str *") * (" * value_inactive * ")"
end


function _parse_piecewise_gate_condition(condition::String)::String

    
    if condition[1:2] ∈ ["if", "or"] || condition[1:3] ∈ ["and", "xor"]
        
        if condition[1:2] ∈ ["if", "or"]
            condition_parts = condition[4:end-1]
        elseif condition[1:3] ∈ ["and", "xor"]
            condition_parts = condition[5:end-1]
        end

        left_part, rigth_part = split_between(condition_parts, ',')
        left_ifelse = _parse_piecewise_gate_condition(left_part)
        right_ifelse = _parse_piecewise_gate_condition(rigth_part)

        # Here cont. representations of the different gates are used, this in order 
        # to allow modellingtoolkit to work (as it will not appreciate something like 
        # && or || in its equations)
        # To emulate an or condition for the two Bool tanh is used 
        if condition[1:2] ∈ ["if", "or"] 
            return "tanh(10 * (" * left_ifelse * "+" *  right_ifelse * "))"
        # To emulate an and gate multiplication between the two conditions is used 
        elseif condition[1:3] == "and"    
            return "(" * left_ifelse * ") * (" * right_ifelse * ")"
        # xor for two boolean variables can be replicated with a second degree polynominal 
        # which is zero at 0 and 2, but 1 at x=1
        elseif condition[1:3] == "xor"
            return "(-(" * left_ifelse * "+" *  right_ifelse * ")^2 + 2*(" * left_ifelse * "+" * right_ifelse * "))"
        end
    end
        
    # Simply invert condition with 1 - condition
    if condition[1:3] == "not"
        condition_parts = condition[5:end-1]
        condition = _parse_piecewise_gate_condition(condition_parts)
        return "(1 - " * condition * ")"        
    end

    # End of recursion arriving at a simple bool condition 
    return parse_piecewise_bool_condition(condition, "1.0", "0.0")
end


# Splits strings by a given delimiter, but only if the delimiter is not inside a function / parenthesis.
function split_between(formula::String, delimiter::Char)::Vector{String}
    
    parts::Vector{String} = Vector{String}(undef, 0)

    in_parenthesis, istart, iend = 0, 1, 1
    for (i, char) in pairs(formula)
        if char == '('
            in_parenthesis += 1
        elseif char == ')'
            in_parenthesis -= 1
        end

        if char == delimiter && in_parenthesis == 0
            iend = i-1
            push!(parts, string(strip(formula[istart:iend])))
            istart = i+1
        end
    end
    # Add ending part of str 
    push!(parts, string(strip(formula[istart:end])))

    return parts
end


#= 
    Functionality for rewriting piecewise equations to Boolean event representation 
=#


function time_dependent_ifelse_to_bool!(model_dict::Dict)

    # Rewrite piecewise using Boolean variables. Handles nested piecewise (or rather in this case ifelse) 
    # via recursion
    for variable in model_dict["has_piecewise"]

        if variable in keys(model_dict["species"])
            _variable = model_dict["species"][variable]
        elseif variable in keys(model_dict["parameters"])
            _variable = model_dict["parameters"][variable]
        end

        _variable.formula = _time_dependent_ifelse_to_bool(_variable.formula, model_dict, _variable.name)
    end
end


function _time_dependent_ifelse_to_bool(formula::String, model_dict::Dict, key::String)::String

    if !occursin("ifelse", formula)
        return formula
    end
    formula_ret = deepcopy(formula)

    indices_ifelse = extract_x(formula, "ifelse"; retindex=true)
    for i in eachindex(indices_ifelse)

        rewrite_ifelse = true

        _args = formula[indices_ifelse[i]][8:end-1]
        condition, left_side, right_side = split_between(_args, ',')

        # Find direction of piecewise inequality to figure out whether left 
        # or right side is activated with increasing time
        i_lt = findfirst(x -> x == '<', condition)
        i_gt = findfirst(x -> x == '>', condition)
        if isnothing(i_gt) && !isnothing(i_lt)
            sign_used = "lt"
            split_by = condition[i_lt:(i_lt+1)] == "<=" ? "<=" : "<"

        elseif !isnothing(i_gt) && isnothing(i_lt)
            sign_used = "gt"
            split_by = condition[i_gt:(i_gt+1)] == ">=" ? ">=" : ">"

        elseif occursin("!=", condition) || occursin("==", condition)
            rewrite_ifelse = false
            continue
        else
            @error "Error : Did not find criteria to split ifelse on"
        end
        # Figure out which side of piecewise is activated when time increases -
        # do we go from false -> true or from true -> false
        lhs_condition, rhs_condition = string.(split(condition, split_by))
        time_right = time_in_formula(rhs_condition)
        time_left = time_in_formula(lhs_condition)
        if time_left == false && time_left == false
            @info "Have ifelse statements which does not contain time. Hence we do not rewrite as event, but rather keep it as an ifelse." maxlog=1
            rewrite_ifelse = false
            continue

        elseif time_left == true
            sign_time = check_sign_time(lhs_condition)
            if (sign_time == 1 && sign_used == "lt") || (sign_time == -1 && sign_used == "gt")
                side_activated_with_time = "right"
            elseif (sign_time == 1 && sign_used == "gt") || (sign_time == -1 && sign_used == "lt")
                side_activated_with_time = "left"
            end

        elseif time_right == true
            sign_time = check_sign_time(rhs_condition)
            if (sign_time == 1 && sign_used == "lt") || (sign_time == -1 && sign_used == "gt")
                side_activated_with_time = "left"
            elseif (sign_time == 1 && sign_used == "gt") || (sign_time == -1 && sign_used == "lt")
                side_activated_with_time = "right"
            end
        end

        # In case of nested ifelse rewrite left-hand and right-hand side of ifelse
        left_side = _time_dependent_ifelse_to_bool(left_side, model_dict, key)
        right_side = _time_dependent_ifelse_to_bool(right_side, model_dict, key)

        if rewrite_ifelse == false
            continue
        end
        
        # Set the name of the piecewise variable
        local j = 1
        while true
            parameter_name = "__parameter_ifelse" * string(j)
            if parameter_name ∉ keys(model_dict["ifelse_parameters"])
                break
            end
            j += 1
        end
        parameter_name = "__parameter_ifelse" * string(j)

        # Rewrite the ifelse to contain Bool variables
        activated_with_time = side_activated_with_time == "left" ? left_side : right_side
        deactivated_with_time = side_activated_with_time == "left" ? right_side : left_side
        _formula = "((1 - " * parameter_name * ")*" * "(" * deactivated_with_time *") + " * parameter_name * "*(" * activated_with_time * "))"
        model_dict["parameters"][parameter_name] = ParameterSBML(parameter_name, true, "0.0", "", false, false, false)
        formula_ret = replace(formula_ret, formula[indices_ifelse[i]] => _formula)

        # Store ifelse parameters to later handle them correctly in model Callbacks 
        model_dict["ifelse_parameters"][parameter_name] = [condition, side_activated_with_time]
    end

    return formula_ret
end


# In piecewise condition cond > cond_limit, check whether cond is increasing or decreasing
# with time. Important to later rewrite piecewise to an event 
function check_sign_time(formula::String)

    formula = replace(formula, " " => "")
    _formula = find_term_with_t(formula)

    if _formula == ""
        str_write = "In $formula for condition in piecewise cannot identify which term time appears in."
        throw(PEtabFileError(str_write))
    end
    
    _formula = replace(_formula, "(" => "")
    _formula = replace(_formula, ")" => "")
    if _formula == "t"
        return 1
    elseif length(_formula) ≥ 2 && _formula[1:2] == "-t"
        return -1
    elseif length(_formula) ≥ 3 && _formula[1:3] == "--t"
        return 1
    elseif length(_formula) ≥ 3 && (_formula[1:3] == "+-t" || _formula[1:3] == "-+t")
        return -1
    end

    # If '-' appears anywhere after the cases above we might be able to infer direction, 
    # but infering in this situation is hard! - so throw an error as the user should 
    # be able to write condition in a more easy manner (avoid several sign changing minus signs)
    if !occursin('-', _formula)
        return 1
    else
        str_write = "For piecewise with time in condition we cannot infer direction for $formula, that is if the condition value increases or decreases with time. This happens if the formula contains a minus sign in the term where t appears."
        throw(PEtabFileError(str_write))
    end
end


# Often we have condition t - parameter, here identify the term which t occurs 
# in order to satisfy the check_sign_time function
function find_term_with_t(formula::String)::String

    # Simple edge case
    if length(formula) == 1 && formula == "t"
        return formula
    end

    istart, parenthesis_level = 1, 0
    local term = ""
    for (i, char) in pairs(formula)

        if i == 1 && char ∈ ['+', '-']
            continue
        end

        parenthesis_level = char == '(' ? parenthesis_level+1 : parenthesis_level
        parenthesis_level = char == ')' ? parenthesis_level-1 : parenthesis_level

        if parenthesis_level == 0 && (char ∈ ['+', '-'] || i == length(formula))
            if formula[i-1] ∈ ['+', '-'] && i != length(formula)
                continue
            end

            term = formula[istart:i]
            if time_in_formula(term) == true || (i == length(formula) && char == 't')
                return term[end] ∈ ['+', '-'] ? term[1:end-1] : term
            end
            istart = i
        end
    end
    return ""
end