#=
    Functionality for correctly handling SBML piecewise statements and if 
    possible, rewrite to discrete callbacks 
=#


# Handles piecewise functions that are to be redefined with ifelse speciements in the model
# equations to allow MKT symbolic calculations.
function piecewise_to_ifelse(rule_formula, model_dict, model_SBML)

    piecewise_eqs = extract_piecewises(rule_formula)
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


function extract_piecewises(formula::String)::Vector{String}

    # Find number of piecewise expressions, and where they start 
    index_piecewise_start = findall("piecewise(", formula)
    n_piecewise = length(index_piecewise_start)
    piecewises::Vector{String} = String[]

    # Extract entire piecewise expression while handling inner paranthesis, e.g
    # when we have "2*piecewise(0, lt(t - insulin_time_1, 0), 1)" it extracts
    # the full expression piecewise(0, lt(t - insulin_time_1, 0), 1) while coutning 
    # the inner (but ignoring it as it is handled via recursion)
    i, k = 1, 1
    while i ≤ n_piecewise
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

        push!(piecewises, formula[index_piecewise_start[i][1]:i_end])

        # Check how many picewise have been processed
        i += length(findall("piecewise(", piecewises[k]))
        k += 1
    end

    return piecewises
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
function split_between(formula::String, delimiter::Union{Char, String})::Vector{String}
    
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
