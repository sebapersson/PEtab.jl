function PEtabODEProblemCache(gradient_method::Symbol, hessian_method::Symbol, FIM_method::Symbol, sensealg, model_info::ModelInfo, ml_models::Union{MLModels}, split_over_conditions::Bool, oprob::ODEProblem)::PEtabODEProblemCache
    @unpack xindices, model, simulation_info, petab_measurements, petab_parameters, petab_net_parameters = model_info
    nxestimate = length(xindices.xids[:estimate])
    nstates = model_info.nstates
    if model.sys_mutated isa ODEProblem
        nxode = length(oprob.p)
    else
        nxode = parameters(model.sys_mutated) |> length
    end

    # Parameters for DiffCache
    chunksize = nxestimate + nxestimate^2
    chunksize = chunksize > 100 ? 100 : chunksize
    if hessian_method ∈ [:ForwardDiff, :BlockForwardDiff, :GaussNewton]
        level_cache = 2
    elseif gradient_method ∈ [:ForwardDiff, :ForwardEquations]
        level_cache = 1
    else
        level_cache = 0
    end
    # Parameters on linear scale
    _xdynamic_mech = zeros(Float64, length(xindices.xids[:dynamic_mech]))
    _xobservable = zeros(Float64, length(xindices.xids[:observable]))
    _xnoise = zeros(Float64, length(xindices.xids[:noise]))
    _xnondynamic_mech = zeros(Float64, length(xindices.xids[:nondynamic_mech]))
    xdynamic_mech = DiffCache(similar(_xdynamic_mech), chunksize, levels = level_cache)
    xobservable = DiffCache(similar(_xobservable), chunksize, levels = level_cache)
    xnoise = DiffCache(similar(_xnoise), chunksize, levels = level_cache)
    xnondynamic_mech = DiffCache(similar(_xnondynamic_mech), chunksize, levels = level_cache)
    # Parameters on parameter-scale
    xdynamic_mech_ps = DiffCache(similar(_xdynamic_mech), chunksize, levels = level_cache)
    xobservable_ps = DiffCache(similar(_xobservable), chunksize, levels = level_cache)
    xnoise_ps = DiffCache(similar(_xnoise), chunksize, levels = level_cache)
    xnondynamic_mech_ps = DiffCache(similar(_xnondynamic_mech), chunksize, levels = level_cache)

    # Parameters for potential neural-networks. First parameters to estimate, then
    # constant parameters (not to be estimated)
    xnn = Dict{Symbol, DiffCache}()
    xnn_dict = Dict{Symbol, ComponentArray}()
    xnn_constant = Dict{Symbol, ComponentArray}()
    if !isnothing(ml_models)
        for (ml_model_id, ml_model) in ml_models
            _p = _get_ml_model_initialparameters(ml_model)
            if ml_model_id in xindices.xids[:ml_est]
                xnn[ml_model_id] = DiffCache(similar(_p); levels = level_cache)
                xnn_dict[ml_model_id] = _p
            else
                set_ml_model_ps!(_p, ml_model_id, ml_model, model.paths)
                xnn_constant[ml_model_id] = _p
            end
        end
    end

    # For all dynamic parameters (mechanistic + nn parameters)
    nxdynamic_tot = _get_nxdynamic(xindices)
    _xdynamic_tot = zeros(Float64, nxdynamic_tot)
    xdynamic_tot = DiffCache(similar(_xdynamic_tot), chunksize, levels = level_cache)
    # For the gradient of parameters that are set via neural-network (needed for efficient
    # gradient of the neural network with the help of the chain rule)
    grad_nn_preode = zeros(Float64, length(xindices.xids[:ml_preode_outputs]))

    # Arrays needed in gradient compuations
    xdynamic_grad = zeros(Float64, nxdynamic_tot)
    xnotode_grad = zeros(Float64, length(xindices.xindices[:not_system_tot]))
    # For forward sensitivity equations and adjoint sensitivity analysis partial
    # derivatives are computed symbolically
    symbolic_needed_grads = gradient_method in [:Adjoint, :ForwardEquations]
    GN_hess = hessian_method == :GaussNewton || FIM_method == :GaussNewton
    if symbolic_needed_grads || GN_hess
        ∂h∂u = zeros(Float64, nstates)
        ∂σ∂u = zeros(Float64, nstates)
        ∂h∂p = zeros(Float64, nxode)
        ∂σ∂p = zeros(Float64, nxode)
        ∂G∂p = zeros(Float64, nxode)
        ∂G∂p_ = zeros(Float64, nxode)
        ∂G∂u = zeros(Float64, nstates)
        if oprob.p isa ComponentArray
            p = similar(oprob.p)
        else
            p = zeros(Float64, nxode)
        end
        u = zeros(Float64, nstates)
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

    # In case the sensitivites are computed via automatic differentitation we need to
    # pre-allocate a sensitivity matrix accross all conditions.
    forward_eqs_AD = gradient_method === :ForwardEquations && sensealg === :ForwardDiff
    nx_forwardeqs = _get_nx_forwardeqs(xindices, split_over_conditions)
    if forward_eqs_AD || GN_hess
        ntimepoints_save = simulation_info.tsaves |> values .|> length |> sum
        S = zeros(Float64, ntimepoints_save * nstates, nx_forwardeqs)
        forward_eqs_grad = zeros(Float64, nx_forwardeqs)
        odesols = zeros(Float64, nstates, ntimepoints_save)
    else
        S = zeros(Float64, 0, 0)
        forward_eqs_grad = zeros(Float64, 0)
        odesols = zeros(Float64, 0, 0)
    end

    # For Gauss-Newton the model residuals are computed
    if GN_hess
        nxtot = nxdynamic_tot + length(xindices.xindices[:not_system_tot])
        jacobian_gn = zeros(Float64, nxtot, length(petab_measurements.time))
        residuals_gn = zeros(Float64, length(petab_measurements.time))
    else
        jacobian_gn = zeros(Float64, (0, 0))
        residuals_gn = zeros(Float64, 0)
    end

    # For Adjoint sensitivity analysis intermediate gradient and state-vectors are
    # propegated, and the sensitivity matrix at t₀ is needed
    if gradient_method === :Adjoint
        du = zeros(Float64, nstates)
        dp = zeros(Float64, nxode)
        adjoint_grad = zeros(Float64, nxode)
        St0 = zeros(Float64, nstates, nxode)
    else
        du = zeros(Float64, 0)
        dp = zeros(Float64, 0)
        adjoint_grad = zeros(Float64, 0)
        St0 = zeros(Float64, 0, 0)
    end

    # Allocate arrays to track if xdynamic should be permuted prior and post gradient
    # compuations. This feature is used if PEtabODEProblem is remade (via remake) to
    # compute the gradient of a problem with reduced number of parameters where to run
    # fewer chunks with ForwardDiff.jl we only run enough chunks to reach nxdynamic
    # TODO: When get here
    xdynamic_input_order::Vector{Int64} = collect(1:length(_xdynamic_mech))
    xdynamic_output_order::Vector{Int64} = collect(1:length(_xdynamic_mech))
    nxdynamic::Vector{Int64} = Int64[length(_xdynamic_tot)]

    # Preallocate arrays used when solving the ODE model
    if simulation_info.has_pre_equilibration == true
        preeq_ids = simulation_info.conditionids[:pre_equilibration]
        sim_ids = simulation_info.conditionids[:experiment]
        condition_ids = vcat(preeq_ids, sim_ids) |> unique
    else
        condition_ids = simulation_info.conditionids[:experiment] |> unique
    end
    pode = Dict{Symbol, DiffCache}()
    u0ode = Dict{Symbol, DiffCache}()
    for cid in condition_ids
        u0ode[cid] = DiffCache(zeros(Float64, nstates), chunksize, levels = level_cache)
        if oprob.p isa ComponentArray
            pode[cid] = DiffCache(similar(oprob.p), chunksize, levels = level_cache)
        else
            pode[cid] = DiffCache(zeros(Float64, nxode), chunksize, levels = level_cache)
        end
    end

    return PEtabODEProblemCache(xdynamic_mech, xnoise, xobservable, xnondynamic_mech, xdynamic_mech_ps,
                                xnoise_ps, xobservable_ps, xnondynamic_mech_ps, xdynamic_grad,
                                xnotode_grad, jacobian_gn, residuals_gn, forward_eqs_grad,
                                adjoint_grad, St0, ∂h∂u, ∂σ∂u, ∂h∂p, ∂σ∂p, ∂G∂p, ∂G∂p_,
                                ∂G∂u, dp, du, p, u, S, odesols, pode, u0ode,
                                xdynamic_input_order, xdynamic_output_order, nxdynamic,
                                xnn, xnn_dict, xnn_constant, xdynamic_tot, grad_nn_preode)
end

function _get_nxdynamic(xindices::ParameterIndices)::Int64
    return length(xindices.xindices_dynamic[:xest_to_xdynamic])
end

function _get_nx_forwardeqs(xindices::ParameterIndices, split_over_conditions::Bool)::Int64
    if split_over_conditions == false
        return length(xindices.xindices_dynamic[:xest_to_xdynamic])
    else
        return (length(xindices.xindices_dynamic[:xdynamic_to_mech]) +
                length(xindices.xindices_dynamic[:nn_in_ode]) +
                length(xindices.xids[:ml_preode_outputs]))
    end
end
