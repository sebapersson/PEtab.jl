function _switch_condition(
        ode_problem::ODEProblem, experiment_id::Symbol, xdynamic_mech::AbstractVector,
        x_ml_models::Dict{Symbol, ComponentArray}, model_info::ModelInfo,
        cache::PEtabODEProblemCache,
        ml_models_pre_simulate::Dict{Symbol, Dict{Symbol, MLModelPreSimulate}},
        posteq_simulation::Bool; simulation_id::Union{Nothing, Symbol} = nothing
    )::ODEProblem
    @unpack xindices, model, nstates = model_info
    simulation_id = isnothing(simulation_id) ? experiment_id : simulation_id

    # Ensure ML-models inside the ODE know which condition it is, in case of condition
    # specific array input
    _set_condition_id_ml_models!(model_info, simulation_id)

    # Each simulation condition needs to have its own associated u0 and p vector, as these
    # vectors can be used in latter computations when computing the observables, hence
    # nothing is allowed to be over-written by another condition
    p = get_tmp(cache.pode[experiment_id], xdynamic_mech)
    u0 = get_tmp(cache.u0ode[experiment_id], xdynamic_mech)
    p .= _get_tunables(ode_problem.p, xindices.get_ps_mtk_parameters)

    @views u0 .= ode_problem.u0[1:length(u0)]

    # p must be set before u0, as u0 can depend on p
    condition_map! = xindices.condition_maps[simulation_id]
    condition_map!(p, xdynamic_mech)

    _set_ode_problem_ml_ps!(p, x_ml_models, model_info)

    # Potential ODE parameters which have their value assigned by a neural-net
    _set_ml_pre_simulate_ps!(
        p, xdynamic_mech, x_ml_models, simulation_id, xindices, ml_models_pre_simulate
    )

    # Initial state can depend on condition specific parameters. For a subset of models
    # remake must be made in two steps, otherwise u0 resets to zero
    model.u0!((@view u0[1:nstates]), p; __post_eq = posteq_simulation)
    ps = _get_ode_problem_ps(ode_problem, p, ode_problem.p, xindices)
    _ode_problem = remake(ode_problem, p = ps)
    _ode_problem = remake(_ode_problem, u0 = u0)

    return _ode_problem
end

function _get_tsave(
        save_observed_t::Bool, simulation_info::SimulationInfo, experiment_id::Symbol,
        ntimepoints_save::Integer
    )::Vector{Float64}
    tmax = simulation_info.tmaxs[experiment_id]
    if save_observed_t == true
        return simulation_info.tsaves[experiment_id]
    elseif ntimepoints_save > 0
        return collect(LinRange(0.0, tmax, ntimepoints_save))
    else
        return Float64[]
    end
end

function _is_dense(save_observed_t::Bool, dense_sol::Bool, ntimepoints_save::Integer)::Bool
    if save_observed_t == true
        return false
    elseif ntimepoints_save > 0
        return false
    else
        return dense_sol
    end
end

function _get_cbs(
        ::ODEProblem, simulation_info::SimulationInfo, simulation_id::Symbol, ::Any
    )::SciMLBase.DECallback
    return simulation_info.callbacks[simulation_id]
end

function _get_tspan(
        ode_problem::ODEProblem, tstart::Float64, tmax::Float64, solver::SciMLAlgorithm,
        float_tspan::Bool
    )::ODEProblem
    # When tmax=Inf and a multistep BDF Julia method, e.g. QNDF, is used tmax must be inf,
    # else if it is a large number such as 1e8 the dt_min is set to a large value making
    # the solver fail. Sundials solvers on the other hand are not compatible with
    # timespan = (0.0, Inf), hence for these we use timespan = (0.0, 1e8)
    # u0tmp needed as remake resets sensitivity initial values to zero
    u0tmp = ode_problem.u0 |> deepcopy
    tmax = _get_tmax(tmax, solver)
    if float_tspan == true
        _ode_problem = remake(ode_problem, tspan = (tstart, tmax))
    else
        ps = _get_tunables(ode_problem.p)
        _ode_problem = remake(
            ode_problem, tspan = convert.(eltype(ps), (tstart, tmax))
        )
    end
    _ode_problem.u0 .= u0tmp
    return _ode_problem
end

function _get_tmax(tmax::Float64, ::Union{CVODE_BDF, CVODE_Adams})::Float64
    return isinf(tmax) ? 1.0e8 : tmax
end
function _get_tmax(tmax::Float64, ::Union{Vector{Symbol}, SciMLAlgorithm})::Float64
    return tmax
end
function _get_tmax(
        condition::Union{ConditionExp, Nothing}, experiment::Union{ConditionExp, Nothing},
        model_info::ModelInfo
    )::Float64
    simulation_id = _get_simulation_id(condition, experiment, model_info)
    pre_equilibration_id = _get_pre_equilibration_id(condition, experiment, model_info)
    experiment_id = _get_experiment_id(simulation_id, pre_equilibration_id)
    return model_info.simulation_info.tmaxs[experiment_id]
end

function _get_start(
        condition::Union{ConditionExp, Nothing}, experiment::Union{ConditionExp, Nothing},
        model_info::ModelInfo
    )::Float64
    simulation_id = _get_simulation_id(condition, experiment, model_info)
    pre_equilibration_id = _get_pre_equilibration_id(condition, experiment, model_info)
    experiment_id = _get_experiment_id(simulation_id, pre_equilibration_id)
    return model_info.simulation_info.tstarts[experiment_id]
end

function _get_preeq_ids(simulation_info::SimulationInfo)::Vector{Symbol}
    return unique(simulation_info.conditionids[:pre_equilibration])
end

function _set_check_trigger_init!(cbs::SciMLBase.DECallback, value::Bool)::Nothing
    for cb in (cbs.discrete_callbacks..., cbs.continuous_callbacks...)
        isnothing(cb.initialize) && continue
        !hasproperty(cb.initialize, :_check_trigger_init) && continue
        cb.initialize._check_trigger_init[1] = value
    end
    return nothing
end

function _set_ml_pre_simulate_ps!(
        p::AbstractVector, xdynamic::AbstractVector,
        x_ml_models::Dict{Symbol, ComponentArray}, simulation_id::Symbol,
        xindices::ParameterIndices,
        ml_models_pre_simulate::Dict{Symbol, Dict{Symbol, MLModelPreSimulate}}
    )::Nothing
    !haskey(ml_models_pre_simulate, simulation_id) && return nothing

    maps_ml_pre_simulate = xindices.maps_ml_pre_simulate[simulation_id]
    for (ml_id, ml_model_pre_simulate) in ml_models_pre_simulate[simulation_id]
        map_ml_model_pre_simulate = maps_ml_pre_simulate[ml_id]
        # In case of neural nets being computed before the function call,
        # ml_model_pre_simulate.outputs is already computed
        outputs = get_tmp(ml_model_pre_simulate.outputs, p)
        if ml_model_pre_simulate.computed[1] == false
            # Only if neural net parameters are estimated, otherwise x_ml is not used to
            # set values in x (vector that might used for gradient computations)
            if haskey(x_ml_models, ml_id)
                x_ml = x_ml_models[ml_id]
                x = _get_ml_model_pre_simulate_x(
                    ml_model_pre_simulate, xdynamic, x_ml, map_ml_model_pre_simulate
                )
            else
                x = _get_ml_model_pre_simulate_x(
                    ml_model_pre_simulate, xdynamic, map_ml_model_pre_simulate
                )
            end
            ml_model_pre_simulate.forward!(outputs, x)
        end
        p[map_ml_model_pre_simulate.ix_sys_outputs] .= outputs
    end
    return nothing
end

function _set_ode_problem_ml_ps!(
        p::ComponentArray, x_ml_models::Dict{Symbol, ComponentArray}, model_info::ModelInfo
    )::Nothing
    for ml_id in model_info.xindices.ids[:ml_in_ode]
        !in(ml_id, model_info.xindices.ids[:ml_est]) && continue
        p[ml_id] .= x_ml_models[ml_id]
    end
    return nothing
end
function _set_ode_problem_ml_ps!(
        p::AbstractVector{<:Real}, x_ml_models::Dict{Symbol, ComponentArray},
        model_info::ModelInfo
    )::Nothing
    for ml_id in model_info.xindices.ids[:ml_in_ode]
        !in(ml_id, model_info.xindices.ids[:ml_est]) && continue
        p[model_info.xindices.indices_dynamic[Symbol("$(ml_id)_sys")]] .= x_ml_models[ml_id]
    end
    return nothing
end

function _nan_to_zero!(x::AbstractArray)::Nothing
    @views x[isnan.(x)] .= 0.0
    return nothing
end
function _nan_to_zero!(x::ModelingToolkitBase.MTKParameters)::Nothing
    @views x.tunable[isnan.(x.tunable)] .= 0.0
    return nothing
end
