function PEtabModel(sys::ModelSystem,
                    observables::Union{PEtabObservable, Vector{PEtabObservable}},
                    measurements::DataFrame, parameters::Vector{PEtabParameter};
                    simulation_conditions::Union{PEtabCondition, Vector{PEtabCondition}, Nothing} = nothing,
                    speciemap::Union{AbstractVector, Nothing} = nothing,
                    parametermap::Union{AbstractVector, Nothing} = nothing,
                    events::Union{PEtabEvent, Vector{PEtabEvent}, Nothing} = nothing,
                    verbose::Bool = false)::PEtabModel
    # One simulation condition is needed by the PEtab v1 standard. For downstream processing
    # easier if is a Vector
    # creation a dummy is created
    if isnothing(simulation_conditions)
        simulation_conditions = [PEtabCondition(:__c0__)]
    elseif simulation_conditions isa PEtabCondition
        simulation_conditions = [simulation_conditions]
    end

    # Downstream processing is easier if provided as a Vector for both events and
    # observables
    if isnothing(events)
        events = PEtabEvent[]
    elseif events isa PEtabEvent
        events = [events]
    end
    if observables isa PEtabObservable
        observables = [observables]
    end

    return _PEtabModel(sys, simulation_conditions, observables, measurements,
                       parameters, speciemap, parametermap, events, verbose)
end

function _PEtabModel(sys::ModelSystem, simulation_conditions::Vector{PEtabCondition},
                     observables::Vector{PEtabObservable}, measurements::DataFrame,
                     parameters::Vector{PEtabParameter},
                     speciemap::Union{Nothing, AbstractVector},
                     parametermap::Union{Nothing, AbstractVector},
                     events::Vector{PEtabEvent}, verbose::Bool)::PEtabModel
    if sys isa ODESystem
        name = "ODESystemModel"
    else
        name = "ReactionSystemModel"
    end

    _logging(:Build_PEtabModel, verbose; name = name)

    # Convert the input to valid PEtab tables
    measurements_df = _measurements_to_table(measurements, simulation_conditions)
    observables_df = _observables_to_table(observables)
    conditions_df = _conditions_to_table(simulation_conditions, sys)
    parameters_df = _parameters_to_table(parameters)
    petab_tables = Dict(:parameters => parameters_df, :conditions => conditions_df,
                        :observables => observables_df, :measurements => measurements_df)

    return _PEtabModel(sys, petab_tables, name, speciemap, parametermap, events, verbose)
end

function _PEtabModel(sys::ModelSystem, petab_tables::Dict{Symbol, DataFrame}, name,
                     speciemap, parametermap, events, verbose::Bool)::PEtabModel
    conditions_df, parameters_df = petab_tables[:conditions], petab_tables[:parameters]
    observables_df = petab_tables[:observables]

    # Build the initial value map (initial values as parameters are set in the reaction sys_mutated)
    sys_mutated = deepcopy(sys)
    sys_mutated, speciemap_model, speciemap_problem = _get_speciemap(sys_mutated,
                                                                     conditions_df,
                                                                     speciemap)
    sys_observables = _get_sys_observables(sys_mutated)
    sys_observable_ids = collect(keys(sys_observables))

    parametermap_problem = _get_parametermap(sys_mutated, parametermap)
    xindices = ParameterIndices(petab_tables, sys_mutated, parametermap_problem,
                                speciemap_problem)
    # Warn user if any variable is unassigned (and defaults to zero)
    _check_unassigned_variables(sys, speciemap_problem, speciemap, :specie, parameters_df,
                                conditions_df)
    _check_unassigned_variables(sys, parametermap_problem, parametermap, :parameter,
                                parameters_df, conditions_df)

    _logging(:Build_u0_h_σ, verbose; exist = false)
    btime = @elapsed begin
        model_SBML = SBMLImporter.ModelSBML(name)
        hstr, u0!str, u0str, σstr = parse_observables(name, Dict{Symbol, String}(),
                                                      sys_mutated, observables_df, xindices,
                                                      speciemap_problem, speciemap_model,
                                                      sys_observable_ids, model_SBML,
                                                      false)
        compute_h = @RuntimeGeneratedFunction(Meta.parse(hstr))
        compute_σ = @RuntimeGeneratedFunction(Meta.parse(σstr))
        # See comment on define petab_mode.jl for standard format input for why this is
        # needed
        _compute_u0! = @RuntimeGeneratedFunction(Meta.parse(u0!str))
        _compute_u0 = @RuntimeGeneratedFunction(Meta.parse(u0str))
        compute_u0! = let f_u0! = _compute_u0!
            (u0, p; __post_eq = false) -> f_u0!(u0, p, __post_eq)
        end
        compute_u0 = let f_u0 = _compute_u0
            (p; __post_eq = false) -> f_u0(p, __post_eq)
        end
    end
    _logging(:Build_u0_h_σ, verbose; time = btime)

    # The callback parsing is part of SBMLImporter, which rewrite any PEtabEvent into
    # EventSBML, which via a dummy ModelSBML (tmp) is parsed into callback. Events are
    # allowed to be condition specific
    _logging(:Build_callbacks, verbose)
    btime = @elapsed begin
        _set_trigger_time!(events)
        cbs, float_tspan = _parse_events(events, sys_mutated, speciemap_problem, parametermap_problem, name, xindices, petab_tables)
    end
    _logging(:Build_callbacks, verbose; time = btime)

    # Path only applies when PEtab tables are provided
    paths = Dict{Symbol, String}()
    return PEtabModel(name, compute_h, compute_u0!, compute_u0, compute_σ, float_tspan,
                      paths, sys, sys_mutated, parametermap_problem, speciemap_problem,
                      petab_tables, cbs, true, events, sys_observables)
end
