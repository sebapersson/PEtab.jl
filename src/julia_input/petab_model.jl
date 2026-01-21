# TODO: Refactor map handling!

"""
    PEtabModel(sys, observables, measurements::DataFrame, parameters; kwargs...)

Create a `PEtabModel` for parameter estimation from model system `sys`, `observables`
linking model output to `measurements`, and `parameters` to estimate.

If there are multiple `observables`, `parameters`, `simulation_conditions`, and/or `events`,
pass them as a `Vector` (e.g. `Vector{PEtabObservable}`).

For examples, see the online package documentation.

# Arguments
- `sys`: Model system (`ReactionSystem` or `ODESystem`).
- `observables`: `PEtabObservable`(s) linking model output to measurements.
- `measurements`: Measurement table (see documentation for required columns).
- `parameters`: `PEtabParameter`(s) to estimate.

# Keyword Arguments
- `simulation_conditions = nothing`: Optional `PEtabCondition`(s) specifying
  condition-specific overrides (initial values and/or model parameters).
- `events = nothing`: Optional `PEtabEvent`(s) defining model events/callbacks.
- `speciemap`: Optional vector of pairs `[:state_id => value, ...]` setting default initial
  values for model states/species. Only needed if values are not already defined in the
  model system (recommended) or provided elsewhere in the PEtab problem.
- `parametermap`: Like `speciemap`, but for model parameters.
- `verbose::Bool = false`: Print progress while building the model.

See also: [`PEtabObservable`](@ref), [`PEtabParameter`](@ref), [`PEtabCondition`](@ref),
and [`PEtabEvent`](@ref).
"""
function PEtabModel(
        sys::ModelSystem, observables::Union{PEtabObservable, Vector{PEtabObservable}},
        measurements::DataFrame, parameters::Union{UserParameter, Vector};
        simulation_conditions::Union{PEtabCondition, Vector{PEtabCondition}, Nothing} = nothing,
        speciemap::Union{AbstractVector, Nothing} = nothing,
        parametermap::Union{AbstractVector, Nothing} = nothing,
        events::Union{PEtabEvent, Vector{PEtabEvent}, Nothing} = nothing,
        verbose::Bool = false, ml_models::Union{MLModels, Nothing} = nothing
    )::PEtabModel

    # One simulation condition is needed by the PEtab v1 standard.
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
    if parameters isa PEtabParameter
        parameters = [parameters]
    end

    ml_models = isnothing(ml_models) ? Dict{Symbol, MLModel}() : ml_models

    return _PEtabModel(
        sys, simulation_conditions, observables, measurements, parameters, speciemap,
        parametermap, events, ml_models, verbose
    )
end

function _PEtabModel(
        sys::ModelSystem, simulation_conditions::Vector{PEtabCondition},
        observables::Vector{PEtabObservable}, measurements::DataFrame,
        parameters::Vector, speciemap::Union{Nothing, AbstractVector},
        parametermap::Union{Nothing, AbstractVector}, events::Vector{PEtabEvent},
        ml_models::MLModels, verbose::Bool
    )::PEtabModel

    if sys isa ODESystem
        name = "ODESystemModel"
    elseif sys isa SDESystem
        name = "SDESystemModel"
    elseif sys isa ODEProblem
        name = "UDEProblemModel"
    else
        name = "ReactionSystemModel"
    end
    ml_models = isnothing(ml_models) ? Dict{Symbol, MLModel}() : ml_models
    _set_ml_models_ps!(ml_models, parameters)

    _logging(:Build_PEtabModel, verbose; name = name)

    # Convert the input to valid PEtab tables
    measurements_df = _measurements_to_table(measurements, simulation_conditions)
    observables_df = _observables_to_table(observables)
    conditions_df = _conditions_to_table(simulation_conditions, sys, ml_models)
    parameters_df = _parameters_to_table(parameters)
    mappings_df = _mapping_to_table(ml_models)
    hybridization_df = _hybridization_to_table(ml_models, parameters_df, conditions_df)
    petab_tables = Dict{Symbol, Union{DataFrame, Dict}}(
        :parameters => parameters_df, :conditions => conditions_df,
        :observables => observables_df, :measurements => measurements_df,
        :mapping => mappings_df, :hybridization => hybridization_df, :yaml => Dict(),
        :experiments => DataFrame()
    )

    return _PEtabModel(sys, petab_tables, name, speciemap, parametermap, events, ml_models, verbose)
end

function _PEtabModel(sys::ModelSystem, petab_tables::PEtabTables, name, speciemap, parametermap, events, ml_models::MLModels, verbose::Bool; float_tspan::Union{Bool, Nothing} = nothing)::PEtabModel
    # Get initial value mappings
    sys_mutated = deepcopy(sys)
    sys_mutated, speciemap_model, speciemap_problem = _get_speciemap(
        sys_mutated, petab_tables, ml_models, speciemap
    )
    sys_mutated, parametermap_problem = _get_parametermap(
        sys_mutated, parametermap, ml_models
    )

    sys_observables = _get_sys_observables(sys_mutated)
    sys_observable_ids = collect(keys(sys_observables))

    paths = Dict{Symbol, String}()
    xindices = ParameterIndices(
        petab_tables, paths, sys_mutated, parametermap_problem, speciemap_problem, ml_models
    )

    # Warn user if any variable is unassigned (and defaults to zero)
    _check_unassigned_variables(
        sys, speciemap_problem, speciemap, :specie, petab_tables
    )
    _check_unassigned_variables(
        sys, parametermap_problem, parametermap, :parameter, petab_tables
    )

    _logging(:Build_u0_h_σ, verbose; exist = false)
    btime = @elapsed begin
        model_SBML = SBMLImporter.ModelSBML(name)
        hstr, u0!str, u0str, σstr = parse_observables(
            name, paths, sys_mutated, petab_tables, xindices, speciemap_problem,
            speciemap_model, sys_observable_ids, model_SBML, ml_models, false
        )
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
        cbs, float_tspan = _parse_events(
            events, sys_mutated, speciemap_problem, parametermap_problem, name, xindices,
            petab_tables
        )
    end
    _logging(:Build_callbacks, verbose; time = btime)

    # Path only applies when PEtab tables are provided
     return PEtabModel(
        name, compute_h, compute_u0!, compute_u0, compute_σ, float_tspan, paths, sys,
        sys_mutated, parametermap_problem, speciemap_problem, petab_tables, cbs, true,
        events, sys_observables, ml_models
    )
end
