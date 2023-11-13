#=
    Functionality for parsing, and handling SBML species (e.g conversion factor etc...)
=#


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
        conversion_factor = isnothing(state.conversion_factor) ? "" : state.conversion_factor
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
                                                     formula, compartment, conversion_factor, unit, 
                                                     only_substance_units, false, false, false)
    end
    return nothing
end


# Adjust specie equation if compartment is dynamic
function adjust_for_dynamic_compartment!(model_dict::Dict)::Nothing

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
                                                           formual_conc, compartment, specie.conversion_factor,  
                                                           :Concentration, false, false, true, false)
    end

    # When a specie is given in concentration, but the compartment concentration changes
    for (specie_id, specie) in model_dict["species"]

        compartment = model_dict["compartments"][model_dict["species"][specie_id].compartment]
        compartment_name = compartment.name
        if compartment.assignment_rule == true && compartment.formula ∈ model_dict["rate_rule_variables"]
            compartment = model_dict["parameters"][compartment.formula]
        end

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
            formula_amount = isempty(specie.formula) ? "0.0" : "(" * specie.formula * ")" * compartment_name 
        end

        # New formula for conc. specie
        specie.formula = formula_amount * "/(" * compartment_name * ") - " * specie_amount_id * "/(" * compartment_name * ")^2*" * compartment.formula

        # Add new conc. specie to model
        model_dict["species"][specie_amount_id] = SpecieSBML(specie_amount_id, false, false, initial_value_amount,
                                                             formula_amount, compartment_name, specie.conversion_factor, 
                                                             :Amount, false, false, false, false)
    end
    return nothing
end


# Adjust specie via conversion factor 
function adjust_conversion_factor!(model_dict::Dict, model_SBML::SBML.Model)::Nothing

    for (specie_id, specie) in model_dict["species"]
        if specie.assignment_rule == true
            continue
        end

        if specie_id ∉ keys(model_SBML.species)
            continue
        end

        # Conversion factors only affect species whose values are changed via reactions, 
        # but not rate-rules
        if specie.rate_rule == true
            continue
        end

        # Boundary value species are not affected 
        if specie.boundary_condition == true
            continue
        end

        # Zero change of rate for specie
        if isempty(specie.formula)
            continue
        end

        if !isempty(specie.conversion_factor)
            conversion_factor = specie.conversion_factor
        elseif !isnothing(model_SBML.conversion_factor)
            conversion_factor = model_SBML.conversion_factor
        else
            return nothing
        end

        specie.formula = "(" * specie.formula * ") * " * conversion_factor
    end
    
    return nothing
end