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


function build_model_dict(model_SBML, ifelse_to_event::Bool)

    # Nested dictionaries to act as intermedidate struct to store relevent 
    # model data
    model_dict = Dict("species" => Dict{String, SpecieSBML}(), 
                      "parameters" => Dict{String, ParameterSBML}(), 
                      "compartments" => Dict{String, CompartmentSBML}(), 
                      "SBML_functions" => Dict{String, Vector{String}}(), 
                      "ifelse_parameters" => Dict(), 
                      "events" => Dict{String, EventSBML}(), 
                      "reactions" => Dict{String, ReactionSBML}(), 
                      "algebraic_rules" => Dict{String, String}(), 
                      "generated_ids" => Dict{String, String}(), 
                      "piecewise_expressions" => Dict{String, String}(),
                      "ifelse_bool_expressions" => Dict{String, String}(),
                      "assignment_rule_variables" => String[], 
                      "rate_rule_variables" => String[], 
                      "appear_in_reactions" => String[], 
                      "has_piecewise" => String[])

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

    # Ensure that event participating parameters and compartments are not simplfied away when calling 
    # structurally_simplify
    include_event_parameters_in_model!(model_dict)

    # Per level3 rateOf can appear in any formula, and should be replaced with corresponding rate
    replace_rateOf!(model_dict)

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
    dict_model_str["species"] = "\tspecies = ["
    dict_model_str["parameters_sym"] = "\tModelingToolkit.@parameters "
    dict_model_str["algebraicVariables"] = ""
    dict_model_str["parameters"] = "\tparameters = ["
    dict_model_str["eqs"] = "\teqs = [\n"
    dict_model_str["ODESystem"] = "\t@named sys = ODESystem(eqs, t, species, parameters)"
    dict_model_str["specie_map"] = "\tspecie_map = [\n"
    dict_model_str["parameter_map"] = "\tparameter_map = [\n"

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
        dict_model_str["species"] *= specie_id * ", "
    end
    for (parameter_id, parameter) in model_dict["parameters"]
        if parameter.constant == true
            continue
        end
        dict_model_str["variables"] *= parameter_id * "(t) "
        dict_model_str["species"] *= parameter_id * ", "
    end
    for (compartment_id, compartment) in model_dict["compartments"]
        if compartment.constant == true
            continue
        end
        dict_model_str["variables"] *= compartment_id * "(t) "
        dict_model_str["species"] *= compartment_id * ", "
    end
    dict_model_str["species"] = dict_model_str["species"][1:end-2] * "]" # Ensure correct valid syntax

    for (parameter_id, parameter) in model_dict["parameters"]
        if parameter.constant == false
            continue
        end
        dict_model_str["parameters_sym"] *= parameter_id * " "
        dict_model_str["parameters"] *= parameter_id * ", "
    end
    for (compartment_id, compartment) in model_dict["compartments"]
        if compartment.constant == false
            continue
        end
        dict_model_str["parameters_sym"] *= compartment_id * " "
        dict_model_str["parameters"] *= compartment_id * ", "
    end

    # Special case where we do not have any parameters
    if length(dict_model_str["parameters_sym"]) == 29
        dict_model_str["parameters_sym"] = ""
        dict_model_str["parameters"] *= "]"
    else
        dict_model_str["parameters"] = dict_model_str["parameters"][1:end-2] * "]"
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
        dict_model_str["eqs"] *= "\t" * eq * ",\n"
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
        dict_model_str["eqs"] *= "\t" * eq * ",\n"
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
        dict_model_str["eqs"] *= "\t" * eq * ",\n"
    end
    # Algebraic rules
    for rule_formula in values(model_dict["algebraic_rules"])
        dict_model_str["eqs"] *= "\t" * rule_formula * ",\n"
    end
    dict_model_str["eqs"] *= "\t]"

    #=
        Build the initial value map
    =#
    # Species
    for (specie_id, specie) in model_dict["species"]
        u0eq = specie.initial_value
        dict_model_str["specie_map"] *= "\t" * specie_id * " =>" * u0eq * ",\n"
    end
    # Parameters
    for (parameter_id, parameter) in model_dict["parameters"]
        if !(parameter.rate_rule == true || parameter.assignment_rule == true)
            continue
        end
        u0eq = parameter.initial_value
        dict_model_str["specie_map"] *= "\t" * parameter_id * " => " * u0eq * ",\n"
    end
    # Compartments
    for (compartment_id, compartment) in model_dict["compartments"]
        if compartment.rate_rule != true 
            continue
        end
        u0eq = compartment.initial_value        
        dict_model_str["specie_map"] *= "\t" * compartment_id * " => " * u0eq * ",\n"
    end
    dict_model_str["specie_map"] *= "\t]"

    #=
        Build the parameter map
    =#
    for (parameter_id, parameter) in model_dict["parameters"]
        if parameter.constant == false
            continue
        end
        peq = parameter.formula
        dict_model_str["parameter_map"] *= "\t" * parameter_id * " =>" * peq * ",\n"
    end
    for (compartment_id, compartment) in model_dict["compartments"]
        if compartment.constant == false
            continue
        end
        ceq = compartment.formula
        dict_model_str["parameter_map"] *= "\t" * compartment_id * " =>" * ceq * ",\n"
    end
    dict_model_str["parameter_map"] *= "\t]"

    ### Writing to file
    model_name = replace(model_name, "-" => "_")
    io = IOBuffer()
    println(io, "function get_ODESystem_" * model_name * "(foo)")
    println(io, "\t# Model name: " * model_name)
    println(io, "")

    println(io, dict_model_str["variables"])
    println(io, dict_model_str["algebraicVariables"])
    println(io, dict_model_str["species"])
    println(io, "")
    println(io, dict_model_str["parameters_sym"])
    println(io, dict_model_str["parameters"])
    println(io, "")
    println(io, "    D = Differential(t)")
    println(io, "")
    println(io, dict_model_str["eqs"])
    println(io, "")
    println(io, dict_model_str["ODESystem"])
    println(io, "")
    println(io, dict_model_str["specie_map"])
    println(io, "")
    println(io, "\t# SBML file parameter values")
    println(io, dict_model_str["parameter_map"])
    println(io, "")
    println(io, "    return sys, specie_map, parameter_map")
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
