function PEtabODEProblemCache(
        gradient_method::Symbol, hessian_method::Symbol, FIM_method::Symbol, sensealg,
        model_info::ModelInfo, ml_models::Union{MLModels}, split_over_conditions::Bool,
        oprob::ODEProblem
    )::PEtabODEProblemCache
    @unpack xindices, model, simulation_info = model_info
    @unpack petab_measurements, petab_parameters, petab_ml_parameters = model_info
    @unpack xids, indices_est = xindices

    n_estimate = _get_nx_estimate(xindices)
    n_states = model_info.nstates
    n_parameters_sys = _get_n_parameters_sys(model.sys_mutated)

    # DiffCache options
    chunk_size = n_estimate + n_estimate^2
    chunk_size = chunk_size > 100 ? 100 : chunk_size
    if hessian_method ∈ [:ForwardDiff, :BlockForwardDiff, :GaussNewton]
        level_cache = 2
    elseif gradient_method ∈ [:ForwardDiff, :ForwardEquations]
        level_cache = 1
    else
        level_cache = 0
    end

    # Pre-allocate cache for mechanistic parameters
    pre_xdynamic_mech = zeros(Float64, length(indices_est[:est_to_dynamic_mech]))
    pre_observable = zeros(Float64, length(xids[:observable]))
    pre_xnoise = zeros(Float64, length(xids[:noise]))
    pre_xnondynamic_mech = zeros(Float64, length(xids[:nondynamic_mech]))
    # On linear scale
    xdynamic_mech = _get_cache(pre_xdynamic_mech, chunk_size, level_cache)
    xobservable = _get_cache(pre_observable, chunk_size, level_cache)
    xnoise = _get_cache(pre_xnoise, chunk_size, level_cache)
    xnondynamic_mech = _get_cache(pre_xnondynamic_mech, chunk_size, level_cache)
    # On parameter scale
    xdynamic_mech_ps = _get_cache(pre_xdynamic_mech, chunk_size, level_cache)
    xobservable_ps = _get_cache(pre_observable, chunk_size, level_cache)
    xnoise_ps = _get_cache(pre_xnoise, chunk_size, level_cache)
    xnondynamic_mech_ps = _get_cache(pre_xnondynamic_mech, chunk_size, level_cache)

    # Pre-allocate ML model parameters. As x_ml is a Dict, during splitting the correct
    # type is moved to x_ml from the cache for estimated parameters
    x_ml_models_cache = Dict{Symbol, DiffCache}()
    x_ml_models = Dict{Symbol, ComponentArray}()
    x_ml_models_constant = Dict{Symbol, ComponentArray}()
    if !isnothing(ml_models)
        for ml_model in ml_models.ml_models
            ml_id = ml_model.ml_id
            ps = _get_ml_model_initialparameters(ml_model)
            if ml_id in xids[:ml_est]
                x_ml_models_cache[ml_id] = DiffCache(similar(ps); levels = level_cache)
                x_ml_models[ml_id] = ps
            else
                set_ml_model_ps!(ps, ml_model, model.paths)
                x_ml_models_constant[ml_id] = ps
            end
        end
    end

    # Arrays used in gradient computations
    n_dynamic_est = length(indices_est[:est_to_dynamic])
    xdynamic = _get_cache(zeros(Float64, n_dynamic_est), chunk_size, level_cache)
    xdynamic_grad = zeros(Float64, n_dynamic_est)
    x_not_system_grad = zeros(Float64, length(indices_est[:est_to_not_system]))
    grad_ml_pre_simulate_outputs = zeros(Float64, length(xids[:sys_ml_pre_simulate_outputs]))
    # Which arrays to allocate depend on gradient method used
    AD_gradient = gradient_method == :ForwardDiff
    GN_hess = hessian_method == :GaussNewton || FIM_method == :GaussNewton
    if !AD_gradient || GN_hess
        ∂h∂u = zeros(Float64, n_states)
        ∂σ∂u = zeros(Float64, n_states)
        ∂h∂p = zeros(Float64, n_parameters_sys)
        ∂σ∂p = zeros(Float64, n_parameters_sys)
        ∂G∂p = zeros(Float64, n_parameters_sys)
        ∂G∂p_ = zeros(Float64, n_parameters_sys)
        ∂G∂u = zeros(Float64, n_states)
        p = similar(oprob.p)
        u = zeros(Float64, n_states)
    else
        ∂h∂u = zeros(Float64, 0)
        ∂σ∂u = zeros(Float64, 0)
        ∂h∂p = zeros(Float64, 0)
        ∂σ∂p = zeros(Float64, 0)
        ∂G∂p = zeros(Float64, 0)
        ∂G∂p_ = zeros(Float64, 0)
        ∂G∂u = zeros(Float64, 0)
        p = zeros(Float64, 0)
        u = zeros(Float64, 0)
    end
    # In case forward sensitivities gradients are computed via AD
    forward_eqs_AD = (gradient_method === :ForwardEquations && sensealg === :ForwardDiff)
    if forward_eqs_AD || GN_hess
        nx_forward_eqs = _get_nx_forward_eqs(xindices, split_over_conditions)
        n_time_points_save = simulation_info.tsaves_no_cbs |>
            values .|>
            length |>
            sum
        S = zeros(Float64, n_time_points_save * n_states, nx_forward_eqs)
        forward_eqs_grad = zeros(Float64, nx_forward_eqs)
        odesols = zeros(Float64, n_states, n_time_points_save)
    else
        S = zeros(Float64, 0, 0)
        forward_eqs_grad = zeros(Float64, 0)
        odesols = zeros(Float64, 0, 0)
    end
    # In case a Gauss-Newton approximation is used
    if GN_hess
        jacobian_gn = zeros(Float64, n_estimate, length(petab_measurements.time))
        residuals_gn = zeros(Float64, length(petab_measurements.time))
    else
        jacobian_gn = zeros(Float64, (0, 0))
        residuals_gn = zeros(Float64, 0)
    end
    # In case adjoint sensitivity gradient method, for which the sensitivity matrix at t0
    # is needed
    if gradient_method === :Adjoint
        du = zeros(Float64, n_states)
        dp = zeros(Float64, n_parameters_sys)
        adjoint_grad = zeros(Float64, n_parameters_sys)
        St0 = zeros(Float64, n_states, n_parameters_sys)
    else
        du = zeros(Float64, 0)
        dp = zeros(Float64, 0)
        adjoint_grad = zeros(Float64, 0)
        St0 = zeros(Float64, 0, 0)
    end

    # Pre-allocate arrays for solving the ODE-model
    p_ode = Dict{Symbol, DiffCache}()
    u0_ode = Dict{Symbol, DiffCache}()
    if simulation_info.has_pre_equilibration == true
        condition_ids = unique(
            vcat(
                simulation_info.conditionids[:pre_equilibration],
                simulation_info.conditionids[:experiment]
            )
        )
    else
        condition_ids = unique(simulation_info.conditionids[:experiment])
    end
    for condition_id in condition_ids
        u0_ode[condition_id] = _get_cache(zeros(Float64, n_states), chunk_size, level_cache)
        p_ode[condition_id] = _get_cache(similar(oprob.p), chunk_size, level_cache)
    end

    return PEtabODEProblemCache(
        xdynamic_mech, xnoise, xobservable, xnondynamic_mech, xdynamic_mech_ps, xnoise_ps,
        xobservable_ps, xnondynamic_mech_ps, xdynamic_grad, x_not_system_grad, jacobian_gn,
        residuals_gn, forward_eqs_grad, adjoint_grad, St0, ∂h∂u, ∂σ∂u, ∂h∂p, ∂σ∂p, ∂G∂p,
        ∂G∂p_, ∂G∂u, dp, du, p, u, S, odesols, p_ode, u0_ode, x_ml_models_cache,
        x_ml_models, x_ml_models_constant, xdynamic, grad_ml_pre_simulate_outputs
    )
end

function _get_nx_forward_eqs(xindices::ParameterIndices, split_over_conditions::Bool)::Int64
    if split_over_conditions == false
        return length(xindices.indices_est[:est_to_dynamic])
    else
        return (
            length(xindices.indices_dynamic[:dynamic_to_mech]) +
            length(xindices.indices_dynamic[:dynamic_to_ml_sys]) +
            length(xindices.indices_dynamic[:sys_ml_pre_simulate_outputs])
        )
    end
end

_get_cache(x, chunk_size, levels) = DiffCache(similar(x), chunk_size, levels = levels)
