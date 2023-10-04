function PEtab.PEtabModel(system::ReactionSystem,
                          simulation_conditions::Dict{String, T},
                          observables::Dict{String, PEtab.PEtabObservable},
                          measurements::DataFrame,
                          petab_parameters::Vector{PEtab.PEtabParameter};
                          state_map::Union{Nothing, Vector{Pair{T1, Float64}}}=nothing,
                          parameter_map::Union{Nothing, Vector{Pair{T2, Float64}}}=nothing,
                          events::Union{T3, Vector{T3}, Nothing}=nothing,
                          verbose::Bool=false)::PEtab.PEtabModel where {T1<:Union{Symbol, Any}, T2<:Union{Symbol, Any}, T<:Dict, T3<:PEtabEvent}

    model_name = "ReactionSystemModel"                          
    return PEtab._PEtabModel(system, model_name, simulation_conditions, observables, measurements, 
                             petab_parameters, state_map, parameter_map, events, verbose)
end
function PEtab.PEtabModel(system::ReactionSystem,
                          observables::Dict{String, PEtab.PEtabObservable},
                          measurements::DataFrame,
                          petab_parameters::Vector{PEtab.PEtabParameter};
                          state_map::Union{Nothing, Vector{Pair{T1, Float64}}}=nothing,
                          parameter_map::Union{Nothing, Vector{Pair{T2, Float64}}}=nothing,
                          events::Union{T3, Vector{T3}, Nothing}=nothing,
                          verbose::Bool=false)::PEtab.PEtabModel where {T1<:Union{Symbol, Any}, T2<:Union{Symbol, Any}, T3<:PEtabEvent}

    simulation_conditions = Dict("__c0__" => Dict())                        
    model_name = "ReactionSystemModel"                          
    return PEtab._PEtabModel(system, model_name, simulation_conditions, observables, measurements, 
                             petab_parameters, state_map, parameter_map, events, verbose)
end


function PEtab.get_default_values(system::ReactionSystem)
    return Catalyst.get_defaults(system)
end


function PEtab.add_parameter_inital_values!(system::ReactionSystem, state_map)
    return nothing
end


function PEtab.add_model_parameter!(system::ReactionSystem, new_parameter)
    eval(Meta.parse("@parameters " * new_parameter))
    Catalyst.addparam!(system, eval(Meta.parse(new_parameter)))
end
