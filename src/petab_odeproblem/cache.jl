function PEtabODEProblemCache(gradient_method::Symbol,
                              hessian_method::Union{Symbol, Nothing},
                              FIM_method::Symbol,
                              sensealg,
                              model_info::ModelInfo)::PEtabODEProblemCache
    @unpack xindices, model, simulation_info, petab_measurements = model_info
    nxestimate = length(xindices.xids[:estimate])
    nstates = model_info.nstates
    nxode = parameters(model.sys_mutated) |> length

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
    xdynamic = zeros(Float64, length(xindices.xindices[:dynamic]))
    xobservable = zeros(Float64, length(xindices.xindices[:observable]))
    xnoise = zeros(Float64, length(xindices.xindices[:noise]))
    xnondynamic = zeros(Float64, length(xindices.xindices[:nondynamic]))
    # Parameters on parameter-scale
    xdynamic_ps = DiffCache(similar(xdynamic), chunksize, levels = level_cache)
    xobservable_ps = DiffCache(similar(xobservable), chunksize, levels = level_cache)
    xnoise_ps = DiffCache(similar(xnoise), chunksize, levels = level_cache)
    xnondynamic_ps = DiffCache(similar(xnondynamic), chunksize, levels = level_cache)

    # Arrays needed in gradient compuations
    xdynamic_grad = zeros(Float64, length(xdynamic))
    xnotode_grad = zeros(Float64, length(xindices.xids[:not_system]))
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
        p = zeros(Float64, nxode)
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
    # pre-allocate a sensitivity matrix accross all conditions
    forward_eqs_AD = gradient_method === :ForwardEquations && sensealg === :ForwardDiff
    if forward_eqs_AD || GN_hess
        ntimepoints_save = simulation_info.tsaves |> values .|> length |> sum
        S = zeros(Float64, ntimepoints_save * nstates, length(xdynamic))
        odesols = zeros(Float64, nstates, ntimepoints_save)
    else
        S = zeros(Float64, 0, 0)
        odesols = zeros(Float64, 0, 0)
    end

    # For Gauss-Newton or Forward-Equations approach when acumulating the xdynamic
    # gradient, it is of size xdynamic
    if gradient_method === :ForwardEquations || GN_hess
        forward_eqs_grad = zeros(Float64, length(xdynamic))
    else
        forward_eqs_grad = zeros(Float64, 0)
    end

    # For Gauss-Newton the model residuals are computed
    if GN_hess
        jacobian_gn = zeros(Float64, nxestimate, length(petab_measurements.time))
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
    xdynamic_input_order::Vector{Int64} = collect(1:length(xdynamic))
    xdynamic_output_order::Vector{Int64} = collect(1:length(xdynamic))
    nxdynamic::Vector{Int64} = Int64[length(xdynamic)]

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
        pode[cid] = DiffCache(zeros(Float64, nxode), chunksize, levels = level_cache)
        u0ode[cid] = DiffCache(zeros(Float64, nstates), chunksize, levels = level_cache)
    end

    return PEtabODEProblemCache(xdynamic, xnoise, xobservable, xnondynamic, xdynamic_ps,
                                xnoise_ps, xobservable_ps, xnondynamic_ps, xdynamic_grad,
                                xnotode_grad, jacobian_gn, residuals_gn, forward_eqs_grad,
                                adjoint_grad, St0, ∂h∂u, ∂σ∂u, ∂h∂p, ∂σ∂p, ∂G∂p, ∂G∂p_,
                                ∂G∂u, dp, du, p, u, S, odesols, pode, u0ode,
                                xdynamic_input_order, xdynamic_output_order, nxdynamic)
end
