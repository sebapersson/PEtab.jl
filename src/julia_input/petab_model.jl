function PEtabModel(sys::ModelSystem, observables::Dict{String, PEtabObservable},
                    measurements::DataFrame, parameters::Vector{PEtabParameter};
                    simulation_conditions::Union{Nothing, Dict} = nothing,
                    speciemap::Union{Nothing, AbstractVector} = nothing,
                    parametermap::Union{Nothing, AbstractVector} = nothing,
                    events::Union{PEtabEvent, AbstractVector, Nothing} = nothing,
                    verbose::Bool = false, nn::Union{Dict, Nothing} = nothing,
                    mapping_table::Union{Dict, Nothing} = nothing)::PEtabModel
    # One simulation condition is needed by the PEtab standard, if there is no such
    # creation a dummy is created
    if isnothing(simulation_conditions)
        simulation_conditions = Dict("__c0__" => Dict())
    end
    return _PEtabModel(sys, simulation_conditions, observables, measurements,
                       parameters, speciemap, parametermap, events, verbose, nn,
                       mapping_table)
end

function _PEtabModel(sys::ModelSystem, simulation_conditions::Dict,
                     observables::Dict{String, <:PEtabObservable}, measurements::DataFrame,
                     parameters::Vector{PEtabParameter},
                     speciemap::Union{Nothing, AbstractVector},
                     parametermap::Union{Nothing, AbstractVector},
                     events::Union{PEtabEvent, AbstractVector, Nothing},
                     verbose::Bool, nn::Union{Dict, Nothing},
                     mapping_table::Union{Dict, Nothing})::PEtabModel
    if sys isa ODESystem
        name = "ODESystemModel"
    elseif sys isa SDESystem
        name = "SDESystemModel"
    elseif sys isa ODEProblem
        name = "UDEProblemModel"
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
    # A mapping table is only needed for models where a neural network sets the values for
    # the model parameters
    mapping_df = !isnothing(mapping_table) ? _mapping_to_table(mapping_table) : DataFrame()
    petab_tables[:mapping_table] = mapping_df

    # Build the initial value map (initial values as parameters are set in the reaction sys_mutated)
    sys_mutated = deepcopy(sys)
    sys_mutated, speciemap_use = _get_speciemap(sys_mutated, conditions_df, speciemap)
    parametermap_use = _get_parametermap(sys_mutated, parametermap)
    xindices = ParameterIndices(petab_tables, sys_mutated, parametermap_use, speciemap_use, nn)
    # Warn user if any variable is unassigned (and defaults to zero)
    _check_unassigned_variables(sys, speciemap_use, speciemap, :specie, parameters_df,
                                conditions_df)
    _check_unassigned_variables(sys, parametermap_use, parametermap, :parameter,
                                parameters_df, conditions_df)

    _logging(:Build_u0_h_σ, verbose; exist = false)
    btime = @elapsed begin
        model_SBML = SBMLImporter.ModelSBML(name)
        hstr, u0!str, u0str, σstr = parse_observables(name, Dict{Symbol, String}(),
                                                      sys_mutated, observables_df,
                                                      xindices, speciemap_use, model_SBML,
                                                      mapping_df, false)
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
        if !isempty(sbml_events)
            model_SBML = SBMLImporter.ModelSBML(name; events = sbml_events)
            float_tspan = _xdynamic_in_event_cond(model_SBML, xindices, petab_tables) |> !
            psys = _get_sys_parameters(sys_mutated, speciemap_use, parametermap_use) .|>
                string
            cbset = SBMLImporter.create_callbacks(sys_mutated, model_SBML, name;
                                                  p_PEtab = psys, float_tspan = float_tspan)
        else
            cbset, float_tspan = CallbackSet(), true
        end
    end
    _logging(:Build_callbacks, verbose; time = btime)

    # Path only applies when PEtab tables are provided
    paths = Dict{Symbol, String}()
    return PEtabModel(name, compute_h, compute_u0!, compute_u0, compute_σ, float_tspan,
                      paths, sys, sys_mutated, parametermap_use, speciemap_use,
                      petab_tables, cbset, true, nn)
end
