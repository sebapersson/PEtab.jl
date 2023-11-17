mutable struct SpecieSBML
    const name::String
    const boundary_condition::Bool
    const constant::Bool
    initial_value::String # Can be changed by initial assignment
    formula::String # Is updated over time
    const compartment::String
    const conversion_factor::String
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


struct ModelSBML
    species::Dict{String, SpecieSBML}
    parameters::Dict{String, ParameterSBML}
    compartments::Dict{String, CompartmentSBML}
    events::Dict{String, EventSBML}
    reactions::Dict{String, ReactionSBML}
    functions::Dict{String, Vector{String}}
    algebraic_rules::Dict{String, String}
    generated_ids::Dict{String, String}
    piecewise_expressions::Dict{String, String}
    ifelse_bool_expressions::Dict{String, String}
    ifelse_parameters::Dict{String, Vector{String}}
    rate_rule_variables::Vector{String}
    species_in_reactions::Vector{String}
    variables_with_piecewise::Vector{String}
end
function ModelSBML()::ModelSBML
    model_SBML = ModelSBML(Dict{String, SpecieSBML}(),
                           Dict{String, ParameterSBML}(),
                           Dict{String, CompartmentSBML}(),
                           Dict{String, EventSBML}(),
                           Dict{String, ReactionSBML}(),
                           Dict{String, Vector{String}}(), # SBML reactions
                           Dict{String, String}(), # Algebraic rules
                           Dict{String, String}(), # Generated id:s
                           Dict{String, String}(), # Piecewise to ifelse_expressions
                           Dict{String, String}(), # Ifelse to bool expression
                           Dict{String, Vector{String}}(), # Ifelse parameters
                           Vector{String}(undef, 0), # Rate rule variables
                           Vector{String}(undef, 0), # Species_appearing in reactions
                           Vector{String}(undef, 0)) # Variables with piecewise
    return model_SBML                           
end