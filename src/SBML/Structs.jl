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
    algebraic_rules::Dict{String, String}
    generated_ids::Dict{String, String}
    piecewise_expressions::Dict{String, String}
    ifelse_bool_expressions::Dict{String, String}
    assignment_rule_variables::Vector{String}
    ifelse_parameters::Vector{String}
    rate_rule_variables::Vector{String}
    appear_in_reactions::Vector{String}
    has_piecewise::Vector{String}
end