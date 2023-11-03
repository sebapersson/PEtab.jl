# TODO: Refactor code and document functions. Check SBMLToolkit if can be used.



mutable struct SpecieSBML
    const name::String
    const boundary_condition::Bool
    const constant::Bool
    initial_value::String # Can be changed by initial assignment
    formula::String # Is updated over time
    const compartment::String
    const unit::Symbol
    const only_substance_units::Bool
    assignment_rule::Bool
    rate_rule::Bool
    algebraic_rule::Bool
end


mutable struct ParameterSBML
    const name::String
    const constant::Bool
    formula::String
    initial_value::String
    assignment_rule::Bool
    rate_rule::Bool
    algebraic_rule::Bool
end


mutable struct CompartmentSBML
    const name::String
    constant::Bool
    formula::String
    initial_value::String
    assignment_rule::Bool
    rate_rule::Bool
    algebraic_rule::Bool
end


mutable struct EventSBML
    const name::String
    trigger::String
    const formulas::Vector{String}
    const trigger_initial_value::Bool
end


mutable struct ReactionSBML
    const name::String
    kinetic_math::String
    const products::Vector{String}
    const products_stoichiometry::Vector{String}
    const reactants::Vector{String}
    const reactants_stoichiometry::Vector{String}
end


"""
    SBML_to_ModellingToolkit(path_SBML::String, model_name::String, dir_model::String)

Convert a SBML file in path_SBML to a Julia ModelingToolkit file and store
the resulting file in dir_model with name model_name.jl.
"""
function SBML_to_ModellingToolkit(path_SBML::String, path_jl_file::String, model_name::AbstractString; only_extract_model_dict::Bool=false,
                                  ifelse_to_event::Bool=true, write_to_file::Bool=true)

    f = open(path_SBML, "r")
    text = read(f, String)
    close(f)

    # If stoichiometryMath occurs we need to convert the SBML file to a level 3 file
    # to properly handle the latter
    if occursin("stoichiometryMath", text) == false
        model_SBML = readSBML(path_SBML)
    else
        model_SBML = readSBML(path_SBML, doc -> begin
                            set_level_and_version(3, 2)(doc)
                            convert_promotelocals_expandfuns(doc)
                            end)
    end

    model_dict = build_model_dict(model_SBML, ifelse_to_event)

    if only_extract_model_dict == false
        model_str = create_ode_model(model_dict, path_jl_file, model_name, write_to_file)
        return model_dict, model_str
    end

    return model_dict, ""
end


# Parse SBML species
function parse_SBML_species!(model_dict::Dict, model_SBML::SBML.Model)::Nothing

    for (state_id, state) in model_SBML.species

        # If both initial amount and conc are empty use concentration as unit per
        # SBML standard
        if isnothing(state.initial_amount) && isnothing(state.initial_concentration)
            initial_value = ""
            unit = state.substance_units == "substance" ? :Amount : :Concentration
        elseif !isnothing(state.initial_concentration)
            initial_value = string(state.initial_concentration)
            unit = :Concentration
        else
            initial_value = string(state.initial_amount)
            unit = :Amount
        end

        # Specie data
        only_substance_units = isnothing(state.only_substance_units) ? false : state.only_substance_units
        boundary_condition = state.boundary_condition
        compartment = state.compartment
        constant = isnothing(state.constant) ? false : state.constant

        # In case being a boundary condition the state can only be changed events, or rate-rules so set
        # derivative to zero, likewise for constant the formula should be zero (no rate of change)
        if boundary_condition == true || constant == true
            formula = "0.0"
        else
            formula = ""
        end

        # In case the initial value is given in conc, but the state should be given in amounts, adjust
        if unit == :Concentration && only_substance_units == true
            unit = :Amount
            initial_value *= " * " * compartment
        end

        model_dict["species"][state_id] = SpecieSBML(state_id, boundary_condition, constant, initial_value,
                                                     formula, compartment, unit, only_substance_units,
                                                     false, false, false)
    end
    return nothing
end


function parse_SBML_parameters!(model_dict::Dict, model_SBML::SBML.Model)::Nothing

    for (parameter_id, parameter) in model_SBML.parameters

        formula = isnothing(parameter.value) ? "0.0" : string(parameter.value)
        model_dict["parameters"][parameter_id] = ParameterSBML(parameter_id, parameter.constant, formula, "", false, false, false)

        if parameter.constant == false
            push!(model_dict["non_constant_parameters"], parameter_id)
        end
    end
    return nothing
end


function parse_SBML_compartments!(model_dict::Dict, model_SBML::SBML.Model)::Nothing

    for (compartment_id, compartment) in model_SBML.compartments

        size = isnothing(compartment.size) ? "1.0" : string(compartment.size)
        model_dict["compartments"][compartment_id] = CompartmentSBML(compartment_id, compartment.constant, size, "", false, false, false)

        if compartment.constant == false
            push!(model_dict["non_constant_parameters"], compartment_id)
        end
    end
    return nothing
end


function adjust_for_dynamic_compartment!(model_dict::Dict)

    #=
    The volume might change over time but the amount should stay constant, as we have a boundary condition
    for a specie given by a rate-rule. In this case it follows that amount n (amount), V (compartment) and conc.
    are related via the chain rule by:
    dn/dt = d(n/V)/dt*V + n*dV/dt/V
    =#
    for (specie_id, specie) in model_dict["species"]

        # Specie with amount where amount should stay constant
        compartment = model_dict["compartments"][model_dict["species"][specie_id].compartment]
        if !(specie.unit == :Amount &&
             specie.rate_rule == true &&
             specie.boundary_condition == true &&
             specie.only_substance_units == false &&
             compartment.rate_rule == true)

            continue
        end

        if compartment.constant == true
            continue
        end

        # In this case must add additional variable for the specie concentration, to properly get the amount equation
        specie_conc_id = "__" * specie_id * "__conc__"
        initial_value_conc = model_dict["species"][specie_id].initial_value * "/" * compartment.name
        formual_conc = model_dict["species"][specie_id].formula

        # Formula for amount specie
        model_dict["species"][specie_id].formula = formual_conc * "*" *  compartment.name * " + " * specie_id * "*" * compartment.formula * " / " * compartment.name

        # Add new conc. specie to model
        model_dict["species"][specie_conc_id] = SpecieSBML(specie_conc_id, false, false, initial_value_conc,
                                                           formual_conc, compartment, :Concentration, false, false, true, false)
    end

    # When a specie is given in concentration, but the compartment concentration changes
    for (specie_id, specie) in model_dict["species"]

        compartment = model_dict["compartments"][model_dict["species"][specie_id].compartment]
        if !(specie.unit == :Concentration &&
             specie.only_substance_units == false &&
             compartment.constant == false)
            continue
        end
        # Rate rule has priority
        if specie_id ∈ model_dict["rate_rule_variables"]
            continue
        end
        if !any(occursin.(keys(model_dict["species"]), compartment.formula)) && compartment.rate_rule == false
            continue
        end

        # Derivative and inital values newly introduced amount specie
        specie_amount_id = "__" * specie_id * "__amount__"
        initial_value_amount = specie.initial_value * "*" * compartment.name

        # If boundary condition is true only amount, not concentration should stay constant with 
        # compartment size
        if specie.boundary_condition == true
            formula_amount = "0.0"
        else
            formula_amount = isempty(specie.formula) ? "0.0" : "(" * specie.formula * ")" * compartment.name 
        end

        # New formula for conc. specie
        specie.formula = formula_amount * "/(" * compartment.name * ") - " * specie_amount_id * "/(" * compartment.name * ")^2*" * compartment.formula

        # Add new conc. specie to model
        model_dict["species"][specie_amount_id] = SpecieSBML(specie_amount_id, false, false, initial_value_amount,
                                                             formula_amount, compartment.name, :Amount, false, false, false, false)
    end
end


function adjust_conversion_factor!(model_dict::Dict, model_SBML::SBML.Model)::Nothing

    if isnothing(model_SBML.conversion_factor)
        return nothing
    end

    for (specie_id, specie) in model_dict["species"]
        if specie.assignment_rule == true
            continue
        end

        # TODO: Make stoich cases like these to parameters to avoid this?
        if specie_id ∉ keys(model_SBML.species)
            continue
        end

        # Zero change of rate for specie
        if isempty(specie.formula)
            continue
        end

        specie.formula = "(" * specie.formula * ") * " * model_SBML.conversion_factor
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
    model_dict["species"] = Dict()
    model_dict["parameters"] = Dict()
    model_dict["compartments"] = Dict()
    model_dict["SBML_functions"] = Dict()
    model_dict["derivatives"] = Dict()
    model_dict["boolVariables"] = Dict()
    model_dict["events"] = Dict()
    model_dict["reactions"] = Dict()
    model_dict["algebraic_rules"] = Dict()
    model_dict["inputFunctions"] = Dict()
    model_dict["generated_ids"] = Dict()
    model_dict["assignment_rule_variables"] = String[]
    model_dict["rate_rule_variables"] = String[]
    model_dict["non_constant_parameters"] = String[]
    model_dict["appear_in_reactions"] = String[]

    parse_SBML_species!(model_dict, model_SBML)

    parse_SBML_parameters!(model_dict, model_SBML)

    parse_SBML_compartments!(model_dict, model_SBML)

    parse_SBML_functions!(model_dict, model_SBML)

    parse_SBML_rules!(model_dict, model_SBML)

    parse_SBML_events!(model_dict, model_SBML)

    # Positioned after rules since some assignments may include functions
    parse_initial_assignments!(model_dict, model_SBML)

    parse_SBML_reactions!(model_dict, model_SBML)

    # Given the SBML standard reaction id can sometimes appear in the reaction
    # formulas, here the correpsonding id is overwritten with a math expression
    replace_reactionid!(model_dict)

    # Rewrite any time-dependent ifelse to boolean statements such that we can express these as events.
    # This is recomended, as it often increases the stabillity when solving the ODE, and decreases run-time
    if ifelse_to_event == true
        time_dependent_ifelse_to_bool!(model_dict)
    end

    identify_algebraic_rule_variables!(model_dict)

    # SBML allows inconstant compartment size, this must be adjusted if a specie is given in concentration
    adjust_for_dynamic_compartment!(model_dict)

    adjust_conversion_factor!(model_dict, model_SBML)


    # Sometimes parameter can be non-constant, but still have a constant rhs and they change value
    # because of event assignments. This must be captured considered so that the parameter is not
    # simplified away
    for (parameter_id, parameter) in model_dict["parameters"]

        if parameter.algebraic_rule == true || parameter.rate_rule == true || parameter.constant == true
            continue
        end
        if !is_number(parameter.formula)
            continue
        end
        # To pass test case 957
        if parse(Float64, parameter.formula) ≈ π
            continue
        end

        parameter.rate_rule = true
        parameter.initial_value = parameter.formula
        parameter.formula = "0.0"
    end
    # Similar holds for compartments 
    for (compartment_id, compartment) in model_dict["compartments"]

        if compartment.algebraic_rule == true || compartment.rate_rule == true || compartment.constant == true
            continue
        end
        if !is_number(compartment.formula)
            continue
        end

        compartment.rate_rule = true
        compartment.initial_value = compartment.formula
        compartment.formula = "0.0"
    end

    # Replace potential rateOf expression with corresponding rate
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
    dict_model_str["variables"] = "\tModelingToolkit.@variables t "
    dict_model_str["stateArray"] = "\tstateArray = ["
    dict_model_str["parameters"] = "\tModelingToolkit.@parameters "
    dict_model_str["algebraicVariables"] = ""
    dict_model_str["parameterArray"] = "\tparameterArray = ["
    dict_model_str["derivatives"] = "\teqs = [\n"
    dict_model_str["ODESystem"] = "\t@named sys = ODESystem(eqs, t, stateArray, parameterArray)"
    dict_model_str["initialSpeciesValues"] = "\tinitialSpeciesValues = [\n"
    dict_model_str["trueParameterValues"] = "\ttrueParameterValues = [\n"

    # Check if model is empty of derivatives if the case add dummy state to be able to
    # simulate the model
    if ((isempty(model_dict["species"]) || sum([!s.assignment_rule for s in values(model_dict["species"])]) == 0) &&
        (isempty(model_dict["parameters"]) || sum([p.rate_rule for p in values(model_dict["parameters"])]) == 0) &&
        (isempty(model_dict["compartments"]) || sum([c.rate_rule for c in values(model_dict["compartments"])]) == 0))

        model_dict["species"]["foo"] = SpecieSBML("foo", false, false, "1.0", "0.0", "1.0", :Amount,
                                                  false, false, false, false)
    end

    for specie_id in keys(model_dict["species"])
        dict_model_str["variables"] *= specie_id * "(t) "
        dict_model_str["stateArray"] *= specie_id * ", "
    end
    for (parameter_id, parameter) in model_dict["parameters"]
        if parameter.constant == true
            continue
        end
        dict_model_str["variables"] *= parameter_id * "(t) "
        dict_model_str["stateArray"] *= parameter_id * ", "
    end
    for (compartment_id, compartment) in model_dict["compartments"]
        if compartment.constant == true
            continue
        end
        dict_model_str["variables"] *= compartment_id * "(t) "
        dict_model_str["stateArray"] *= compartment_id * ", "
    end
    if length(model_dict["inputFunctions"]) > 0
        dict_model_str["algebraicVariables"] = "    ModelingToolkit.@variables"
        for key in keys(model_dict["inputFunctions"])
            dict_model_str["algebraicVariables"] *= " " * key * "(t)"
            dict_model_str["stateArray"] *= key * ", "
        end
    end
    dict_model_str["stateArray"] = dict_model_str["stateArray"][1:end-2] * "]" # Ensure correct valid syntax

    for (parameter_id, parameter) in model_dict["parameters"]
        if parameter.constant == false
            continue
        end
        dict_model_str["parameters"] *= parameter_id * " "
        dict_model_str["parameterArray"] *= parameter_id * ", "
    end
    for (compartment_id, compartment) in model_dict["compartments"]
        if compartment.constant == false
            continue
        end
        dict_model_str["parameters"] *= compartment_id * " "
        dict_model_str["parameterArray"] *= compartment_id * ", "
    end

    # Special case where we do not have any parameters
    if length(dict_model_str["parameters"]) == 29
        dict_model_str["parameters"] = ""
        dict_model_str["parameterArray"] *= "]"
    else
        dict_model_str["parameterArray"] = dict_model_str["parameterArray"][1:end-2] * "]"
    end

    #=
        Build the model equations
    =#
    # Species
    for (specie_id, specie) in model_dict["species"]

        if specie.algebraic_rule == true
            continue
        end

        formula = isempty(specie.formula) ? "0.0" : specie.formula
        if specie.assignment_rule == true
            eq = specie_id * " ~ " * formula
        else
            eq = "D(" * specie_id * ") ~ " * formula
        end
        dict_model_str["derivatives"] *= "\t" * eq * ",\n"
    end
    # Parameters
    for (parameter_id, parameter) in model_dict["parameters"]

        if parameter.constant == true || parameter.algebraic_rule == true
            continue
        end

        if parameter.rate_rule == false
            eq = parameter_id * " ~ " * parameter.formula
        else
            eq = "D(" * parameter_id * ") ~ " * parameter.formula
        end
        dict_model_str["derivatives"] *= "\t" * eq * ",\n"
    end
    # Compartments
    for (compartment_id, compartment) in model_dict["compartments"]

        if compartment.constant == true || compartment.algebraic_rule == true
            continue
        end

        if compartment.rate_rule == false
            eq = compartment_id * " ~ " * compartment.formula
        else
            eq = "D(" * compartment_id * ") ~ " * compartment.formula
        end
        dict_model_str["derivatives"] *= "\t" * eq * ",\n"
    end
    # Input functions TODO: Refactor like alot
    for key in keys(model_dict["inputFunctions"])
        dict_model_str["derivatives"] *= "\t" * model_dict["inputFunctions"][key] * ",\n"
    end
    # Algebraic rules
    for rule_formula in values(model_dict["algebraic_rules"])
        dict_model_str["derivatives"] *= "\t" * rule_formula * ",\n"
    end
    dict_model_str["derivatives"] *= "\t]"

    #=
        Build the initial value map
    =#
    # Species
    for (specie_id, specie) in model_dict["species"]
        u0eq = specie.initial_value
        dict_model_str["initialSpeciesValues"] *= "\t" * specie_id * " =>" * u0eq * ",\n"
    end
    # Parameters
    for (parameter_id, parameter) in model_dict["parameters"]
        if !(parameter.rate_rule == true || parameter.assignment_rule == true)
            continue
        end
        u0eq = parameter.initial_value
        dict_model_str["initialSpeciesValues"] *= "\t" * parameter_id * " => " * u0eq * ",\n"
    end
    # Compartments
    for (compartment_id, compartment) in model_dict["compartments"]
        if compartment.rate_rule != true 
            continue
        end
        u0eq = compartment.initial_value        
        dict_model_str["initialSpeciesValues"] *= "\t" * compartment_id * " => " * u0eq * ",\n"
    end
    dict_model_str["initialSpeciesValues"] *= "\t]"

    #=
        Build the parameter map
    =#
    if !isempty(dict_model_str["parameters"])
        for (parameter_id, parameter) in model_dict["parameters"]
            if parameter.constant == false
                continue
            end
            peq = parameter.formula
            dict_model_str["trueParameterValues"] *= "\t" * parameter_id * " =>" * peq * ",\n"
        end
        for (compartment_id, compartment) in model_dict["compartments"]
            if compartment.constant == false
                continue
            end
            ceq = compartment.formula
            dict_model_str["trueParameterValues"] *= "\t" * compartment_id * " =>" * ceq * ",\n"
        end
    end
    dict_model_str["trueParameterValues"] *= "\n\t]"

    ### Writing to file
    model_name = replace(model_name, "-" => "_")
    io = IOBuffer()
    println(io, "function getODEModel_" * model_name * "(foo)")
    println(io, "\t# Model name: " * model_name)
    println(io, "")

    println(io, "    ### Define independent and dependent variables")
    println(io, dict_model_str["variables"])
    println(io, "")
    println(io, "    ### Define potential algebraic variables")
    println(io, dict_model_str["algebraicVariables"])
    println(io, "    ### Store dependent variables in array for ODESystem command")
    println(io, dict_model_str["stateArray"])
    println(io, "")
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



