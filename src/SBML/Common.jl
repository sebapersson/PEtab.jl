"""
    replace_variable(formula::T, to_replace::String, replace_with::String)::T where T<:AbstractString

In a formula replaces to_replace with replace_with. Note exact match is required, so if to_replace=time1
and replace_with=1 while formula = time * 2 nothing is replaced as time != time1
"""
function replace_variable(formula::T, to_replace::String, replace_with::String)::T where T<:AbstractString

    _to_replace = Regex("(\\b" * to_replace * "\\b)")
    return replace(formula, _to_replace => replace_with)
end


"""
    process_SBML_str_formula(formula::T, model_dict, model_SBML; check_scaling=false)::T where T<:AbstractString

Processes a string formula by inserting SBML functions, rewriting piecewise to ifelse, and scaling species
"""
function process_SBML_str_formula(formula::T, model_dict::Dict, model_SBML::SBML.Model; check_scaling=false, rate_rule::Bool=false)::T where T<:AbstractString
    
    _formula = SBML_function_to_math(formula, model_dict["SBML_functions"])
    if occursin("piecewise(", _formula)
        _formula = piecewise_to_ifelse(_formula, model_dict, model_SBML)
    end
    _formula = replace_variable(_formula, "time", "t") # Sometimes t is decoded as time

    # SBML equations are given in concentration, in case an amount specie appears in the equation scale with the 
    # compartment in the formula every time the species appear
    for (specie_id, specie) in model_dict["species"]
        if check_scaling == false
            continue
        end
        if specie.unit == :Concentration || specie.only_substance_units == true
            continue
        end

        compartment = specie.compartment
        _formula = replace_variable(_formula, specie_id, "(" * specie_id * "/" * compartment * ")")
    end

    # Replace potential expressions given in initial assignment and that appear in stoichemetric experssions
    # of reactions (these are not species, only math expressions that should be replaced)
    for id in keys(model_SBML.initial_assignments)
        # In case ID does not occur in stoichemetric expressions
        if isempty(model_SBML.reactions)
            continue
        end
        if id ∉ reduce(vcat, vcat([[_r.id for _r in r.products] for r in values(model_SBML.reactions)], [[_r.id for _r in r.reactants] for r in values(model_SBML.reactions)]))
            continue
        end
        if id ∉ keys(model_dict["species"]) && rate_rule == false
            continue
        end
        if isnothing(id)
            continue
        end
        # Do not rewrite is stoichemetric is controlled via event
        if !isempty(model_dict["events"]) && any(occursin.(id, reduce(vcat, [e.formulas for e in values(model_dict["events"])])))
            continue
        end
        if rate_rule == false
            _formula = replace_variable(_formula, id, "(" * model_dict["species"][id].initial_value * ")")
        else
            replace_with = parse_SBML_math(model_SBML.initial_assignments[id])
            _formula = replace_variable(_formula, id, "(" * replace_with * ")")
        end
    end

    # Sometimes we have a stoichemetric expression appearing in for example rule expressions, etc... but it does not 
    # have any initial assignment, or rule assignment. In this case the reference should be replaced with its corresponding 
    # stoichemetry
    for (_, reaction) in model_SBML.reactions
        specie_references = vcat([reactant for reactant in reaction.reactants], [product for product in reaction.products])
        for specie_reference in specie_references
            if isnothing(specie_reference.id)
                continue
            end
            if specie_reference.id ∈ keys(model_SBML.initial_assignments)
                continue
            end
            if specie_reference.id ∈ [rule isa SBML.AlgebraicRule ? "" : rule.variable for rule in model_SBML.rules]
                continue
            end
            if specie_reference.id ∈ keys(model_SBML.species)
                continue
            end
            _formula = replace_variable(_formula, specie_reference.id, string(specie_reference.stoichiometry))
        end
    end

    return _formula
end


function time_in_formula(formula::String)::Bool
    _formula = replace_variable(formula, "t", "")
    return formula != _formula
end


function replace_reactionid_formula(formula::T, model_SBML::SBML.Model)::T where T<:AbstractString
    for (reaction_id, reaction) in model_SBML.reactions
        reaction_math = parse_SBML_math(reaction.kinetic_math)
        formula = replace_variable(formula, reaction_id, reaction_math)
    end
    return formula
end


function replace_rateOf!(model_dict::Dict)::Nothing

    for (parameter_id, parameter) in model_dict["parameters"]
        parameter.formula = replace_rateOf(parameter.formula, model_dict)
        parameter.initial_value = replace_rateOf(parameter.initial_value, model_dict)
    end
    for (specie_id, specie) in model_dict["species"]
        specie.formula = replace_rateOf(specie.formula, model_dict)
        specie.initial_value = replace_rateOf(specie.initial_value, model_dict)
    end
    for (event_id, event) in model_dict["events"]
        for (i, formula) in pairs(event.formulas)
            event.formulas[i] = replace_rateOf(formula, model_dict)
        end
        event.trigger = replace_rateOf(event.trigger, model_dict)
    end
    for (rule_id, rule) in model_dict["algebraic_rules"]
        model_dict["algebraic_rules"][rule_id] = replace_rateOf(rule, model_dict)
    end

    return nothing
end


function replace_rateOf(_formula::T, model_dict::Dict)::String where T<:Union{<:AbstractString, <:Real}

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
    # Compenstate for nested paranthesis 
    for i in eachindex(end_rateof) 
        if any(occursin.(['*', '/'], formula[start_rateof[i]:end_rateof[i]]))
            end_rateof[i] += 1
        end
    end
    args = [formula[start_rateof[i]+7:end_rateof[i]-1] for i in eachindex(start_rateof)]

    replace_with = Vector{String}(undef, length(args))
    for (i, arg) in pairs(args)

        # A constant parameter does not have a rate
        if arg ∈ keys(model_dict["parameters"]) && model_dict["parameters"][arg].constant == true
            replace_with[i] = "0.0"
            continue
        end
        # A parameter via a rate-rule has a rate
        if arg ∈ keys(model_dict["parameters"]) && model_dict["parameters"][arg].rate_rule == true
            replace_with[i] = model_dict["parameters"][arg].formula
            continue
        end

        # A number does not have a rate
        if is_number(arg)
            replace_with[i] = "0.0"
            continue
        end

        # If specie is via a rate-rule we do not scale the state in the expression
        if arg ∈ keys(model_dict["species"]) && model_dict["species"][arg].rate_rule == true
            replace_with[i] = model_dict["species"][arg].formula
            continue
        end

        # Default case, use formula for given specie, and if specie is given in amount
        # Here it might happen that arg is scaled with compartment, e.g. S / C thus 
        # first the specie is extracted 
        arg = filter(x -> x ∉ ['(', ')'], arg)
        arg = occursin('/', arg) ? arg[1:findfirst(x -> x == '/', arg)-1] : arg
        arg = occursin('*', arg) ? arg[1:findfirst(x -> x == '*', arg)-1] : arg
        specie = model_dict["species"][arg]
        scale_with_compartment = specie.unit == :Amount && specie.only_substance_units == false
        if scale_with_compartment == true
            replace_with[i] = "(" * specie.formula * ") / " * specie.compartment
        else
            replace_with[i] = specie.formula
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


function replace_reactionid!(model_dict::Dict)::Nothing

    for (specie_id, specie) in model_dict["species"]
        for (reaction_id, reaction) in model_dict["reactions"]
            specie.formula = replace_variable(specie.formula, reaction_id, reaction.kinetic_math)
        end
    end

    return nothing
end
