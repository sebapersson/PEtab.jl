function PEtabModel(sys::Union{ReactionSystem, ODESystem}, simulation_conditions::Dict,
                    observables::Dict{String, <:PEtabObservable}, measurements::DataFrame,
                    petab_parameters::Vector{PEtabParameter};
                    statemap::Union{Nothing, AbstractVector} = nothing,
                    parametermap::Union{Nothing, AbstractVector} = nothing,
                    events::Union{PEtabEvent, AbstractVector, Nothing} = nothing,
                    verbose::Bool = false)::PEtabModel
    return _PEtabModel(sys, simulation_conditions, observables, measurements,
                       petab_parameters, statemap, parametermap, events, verbose)
end
function PEtabModel(sys::Union{ReactionSystem, ODESystem},
                    observables::Dict{String, <:PEtabObservable}, measurements::DataFrame,
                    petab_parameters::Vector{PEtabParameter};
                    statemap::Union{Nothing, AbstractVector} = nothing,
                    parametermap::Union{Nothing, AbstractVector} = nothing,
                    events::Union{PEtabEvent, AbstractVector, Nothing} = nothing,
                    verbose::Bool = false)::PEtabModel
    simulation_conditions = Dict("__c0__" => Dict())
    return _PEtabModel(sys, simulation_conditions, observables, measurements,
                       petab_parameters, statemap, parametermap, events, verbose)
end

function _PEtabModel(sys::Union{ReactionSystem, ODESystem}, simulation_conditions::Dict,
                     observables::Dict{String, <:PEtabObservable}, measurements::DataFrame,
                     petab_parameters::Vector{PEtabParameter},
                     statemap::Union{Nothing, AbstractVector},
                     parametermap::Union{Nothing, AbstractVector},
                     events::Union{PEtabEvent, AbstractVector, Nothing},
                     verbose::Bool)::PEtabModel
    if sys isa ODESystem
        modelname = "ODESystemModel"
    else
        modelname = "ReactionSystemModel"
    end
    _logging(:Build_PEtabModel, verbose; name = modelname)

    # Convert the input to valid PEtab tables
    measurements_df = _measurements_to_table(measurements, simulation_conditions)
    observables_df = _observables_to_table(observables)
    conditions_df = _conditions_to_table(simulation_conditions)
    parameters_df = _parameters_to_table(petab_parameters)
    petab_tables = Dict(:parameters => parameters_df, :conditions => conditions_df,
                        :observables => observables_df, :measurements => measurements_df)

    # Build the initial value map (initial values as parameters are set in the reaction sys_mutated)
    sys_mutated = deepcopy(sys)
    statemap_use = _get_statemap(sys_mutated, conditions_df, statemap)
    parametermap_use = _get_parametermap(sys_mutated, parametermap)
    # Warn user if any variable is unassigned (and defaults to zero)
    _check_unassigned_variables(statemap_use, :specie, parameters_df, conditions_df)
    _check_unassigned_variables(parametermap_use, :parameter, parameters_df, conditions_df)

    _logging(:Build_u0_h_σ, verbose; exist = false)
    btime = @elapsed begin
        h_str, u0!_str, u0_str, σ_str = create_u0_h_σ_file(modelname, sys_mutated,
                                                           conditions_df, measurements_df,
                                                           parameters_df, observables_df,
                                                           statemap_use)
        compute_h = @RuntimeGeneratedFunction(Meta.parse(h_str))
        compute_u0! = @RuntimeGeneratedFunction(Meta.parse(u0!_str))
        compute_u0 = @RuntimeGeneratedFunction(Meta.parse(u0_str))
        compute_σ = @RuntimeGeneratedFunction(Meta.parse(σ_str))
    end
    _logging(:Build_u0_h_σ, verbose; time = btime)

    _logging(:Build_∂_h_σ, verbose; exist = false)
    btime = @elapsed begin
        ∂h∂u_str, ∂h∂p_str, ∂σ∂u_str, ∂σ∂p_str = create_∂_h_σ_file(modelname, sys_mutated,
                                                                   conditions_df,
                                                                   measurements_df,
                                                                   parameters_df,
                                                                   observables_df,
                                                                   statemap_use)
        compute_∂h∂u! = @RuntimeGeneratedFunction(Meta.parse(∂h∂u_str))
        compute_∂h∂p! = @RuntimeGeneratedFunction(Meta.parse(∂h∂p_str))
        compute_∂σ∂σu! = @RuntimeGeneratedFunction(Meta.parse(∂σ∂u_str))
        compute_∂σ∂σp! = @RuntimeGeneratedFunction(Meta.parse(∂σ∂p_str))
    end
    _logging(:Build_∂_h_σ, verbose; time = btime)

    # For Callbacks. TODO: This functionality should live in SBMLImporter, basically
    # should rewrite PEtabEvent to SBMLEvent to not reuse code. Must be done after
    # updating SBMLImporter
    _logging(:Build_callbacks, verbose)
    btime = @elapsed begin
        parameter_info = parse_parameters(parameters_df)
        measurement_info = parse_measurements(measurements_df, observables_df)
        θ_indices = parse_conditions(parameter_info, measurement_info, sys_mutated,
                                     parametermap_use, statemap_use, conditions_df)
        cbset, compute_tstops, convert_tspan = process_petab_events(events, sys_mutated,
                                                                    θ_indices)
    end
    _logging(:Build_callbacks, verbose; time = btime)

    # Path only applies when PEtab tables are provided
    paths = Dict{Symbol, String}()
    petab_model = PEtabModel(modelname,
                             compute_h,
                             compute_u0!,
                             compute_u0,
                             compute_σ,
                             compute_∂h∂u!,
                             compute_∂σ∂σu!,
                             compute_∂h∂p!,
                             compute_∂σ∂σp!,
                             compute_tstops,
                             convert_tspan,
                             paths,
                             sys,
                             sys_mutated,
                             parametermap_use,
                             statemap_use,
                             petab_tables,
                             cbset,
                             Function[],
                             true)
    return petab_model
end
