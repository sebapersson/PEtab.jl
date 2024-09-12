function PEtabModel(sys::ModelSystem, simulation_conditions::Dict,
                    observables::Dict{String, <:PEtabObservable}, measurements::DataFrame,
                    parameters::Vector{PEtabParameter};
                    statemap::Union{Nothing, AbstractVector} = nothing,
                    parametermap::Union{Nothing, AbstractVector} = nothing,
                    events::Union{PEtabEvent, AbstractVector, Nothing} = nothing,
                    verbose::Bool = false)::PEtabModel
    return _PEtabModel(sys, simulation_conditions, observables, measurements,
                       parameters, statemap, parametermap, events, verbose)
end
function PEtabModel(sys::ModelSystem, observables::Dict{String, <:PEtabObservable},
                    measurements::DataFrame, parameters::Vector{PEtabParameter};
                    statemap::Union{Nothing, AbstractVector} = nothing,
                    parametermap::Union{Nothing, AbstractVector} = nothing,
                    events::Union{PEtabEvent, AbstractVector, Nothing} = nothing,
                    verbose::Bool = false)::PEtabModel
    simulation_conditions = Dict("__c0__" => Dict())
    return _PEtabModel(sys, simulation_conditions, observables, measurements,
                       parameters, statemap, parametermap, events, verbose)
end

function _PEtabModel(sys::ModelSystem, simulation_conditions::Dict,
                     observables::Dict{String, <:PEtabObservable}, measurements::DataFrame,
                     parameters::Vector{PEtabParameter},
                     statemap::Union{Nothing, AbstractVector},
                     parametermap::Union{Nothing, AbstractVector},
                     events::Union{PEtabEvent, AbstractVector, Nothing},
                     verbose::Bool)::PEtabModel
    if sys isa ODESystem
        name = "ODESystemModel"
    else
        name = "ReactionSystemModel"
    end
    _logging(:Build_PEtabModel, verbose; name = name)

    # Convert the input to valid PEtab tables
    measurements_df = _measurements_to_table(measurements, simulation_conditions)
    observables_df = _observables_to_table(observables)
    conditions_df = _conditions_to_table(simulation_conditions)
    parameters_df = _parameters_to_table(parameters)
    petab_tables = Dict(:parameters => parameters_df, :conditions => conditions_df,
                        :observables => observables_df, :measurements => measurements_df)

    # Build the initial value map (initial values as parameters are set in the reaction sys_mutated)
    sys_mutated = deepcopy(sys)
    sys_mutated, statemap_use = _get_statemap(sys_mutated, conditions_df, statemap)
    parametermap_use = _get_parametermap(sys_mutated, parametermap)
    xindices = ParameterIndices(petab_tables, sys_mutated, parametermap_use, statemap_use)
    # Warn user if any variable is unassigned (and defaults to zero)
    _check_unassigned_variables(statemap_use, :specie, parameters_df, conditions_df)
    _check_unassigned_variables(parametermap_use, :parameter, parameters_df, conditions_df)

    _logging(:Build_u0_h_σ, verbose; exist = false)
    btime = @elapsed begin
        model_SBML = SBMLImporter.ModelSBML(name)
        hstr, u0!str, u0str, σstr = parse_observables(name, Dict{Symbol, String}(),
                                                      sys_mutated, observables_df,
                                                      xindices, statemap_use, model_SBML,
                                                      false)
        compute_h = @RuntimeGeneratedFunction(Meta.parse(hstr))
        compute_u0! = @RuntimeGeneratedFunction(Meta.parse(u0!str))
        compute_u0 = @RuntimeGeneratedFunction(Meta.parse(u0str))
        compute_σ = @RuntimeGeneratedFunction(Meta.parse(σstr))
    end
    _logging(:Build_u0_h_σ, verbose; time = btime)

    # The callback parsing is part of SBMLImporter. Basically, PEtabEvents are rewritten
    # to SBMLImporter.EventSBML, which then via a dummy ModelSBML (tmp) is parsed into
    # callback
    _logging(:Build_callbacks, verbose)
    btime = @elapsed begin
        sbml_events = parse_events(events, sys_mutated)
        model_SBML = SBMLImporter.ModelSBML(name; events = sbml_events)
        float_tspan = _xdynamic_in_event_cond(model_SBML, xindices, petab_tables) |> !
        psys = _get_sys_parameters(sys_mutated, statemap_use, parametermap_use) .|> string
        cbset = SBMLImporter.create_callbacks(sys_mutated, model_SBML, name;
                                              p_PEtab = psys, float_tspan = float_tspan)
    end
    _logging(:Build_callbacks, verbose; time = btime)

    # Path only applies when PEtab tables are provided
    paths = Dict{Symbol, String}()
    return PEtabModel(name, compute_h, compute_u0!, compute_u0, compute_σ,  float_tspan,
                      paths, sys, sys_mutated, parametermap_use, statemap_use, petab_tables,
                      cbset, true)
end
