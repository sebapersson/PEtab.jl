#=
    Functions for computing forward-sensitivities with SciMLSensitivity
=#

function PEtab._get_odeproblem_gradient(odeproblem::ODEProblem, ::Symbol,
                                        sensealg::ForwardAlg)::ODEProblem
    return ODEForwardSensitivityProblem(odeproblem.f, odeproblem.u0, odeproblem.tspan,
                                        odeproblem.p, sensealg = sensealg)
end

function PEtab.solve_sensitivites!(
        ::PEtab.ModelInfo, _solve_conditions!::Function,
        xdynamic::Vector{<:AbstractFloat}, ::ForwardAlg, ::PEtab.PEtabODEProblemInfo,
        cids::Vector{Symbol}, ::Nothing
    )::Bool
    success = _solve_conditions!(xdynamic, cids)
    return success
end

function PEtab._grad_forward_eqs_cond!(
        grad::Vector{T}, xdynamic::Vector{T}, xnoise::Vector{T}, xobservable::Vector{T},
        xnondynamic_mech::Vector{T}, icid::Int64, ::ForwardAlg,
        probinfo::PEtab.PEtabODEProblemInfo, model_info::PEtab.ModelInfo
    )::Nothing where {T <: AbstractFloat}
    @unpack xindices, simulation_info, model = model_info
    @unpack petab_parameters, petab_measurements = model_info
    @unpack imeasurements_t, tsaves_no_cbs = simulation_info
    cache = probinfo.cache

    # Simulation ids
    cid = simulation_info.conditionids[:experiment][icid]
    simid = simulation_info.conditionids[:simulation][icid]
    sol = simulation_info.odesols_derivatives[cid]

    # Partial derivatives needed for computing the gradient (derived from the chain-rule)
    ∂G∂u!, ∂G∂p! = PEtab._get_∂G∂_!(model_info, cid, xnoise, xobservable, xnondynamic_mech, cache.x_ml_models, cache.x_ml_models_constant)

    p = sol.prob.p
    ∂G∂p, ∂G∂p_ = zeros(Float64, length(p)), zeros(Float64, length(p))
    ∂G∂u = zeros(Float64, length(PEtab._get_state_ids(model.sys_mutated)))
    _grad = zeros(Float64, length(p))
    for (it, tsave) in pairs(tsaves_no_cbs[cid])
        u, _S = extract_local_sensitivities(sol, it, true)
        ∂G∂u!(∂G∂u, u, p, tsave, it)
        ∂G∂p!(∂G∂p_, u, p, tsave, it)
        _grad .+= transpose(_S) * ∂G∂u
        ∂G∂p .+= ∂G∂p_
    end

    # In _grad the derivatives for all parameters in oprob.p are stored, if a subset
    # of these are the output of neural-net, they are the inner-derivative needed to
    # compute the gradient of the neural-net. As usual, the outer Jacobian derivative has
    # already been computed, so the only thing left is to combine them
    if !isempty(xindices.xids[:sys_ml_pre_simulate_outputs])
        ix = xindices.indices_dynamic[:sys_ml_pre_simulate_outputs]
        cache.grad_ml_pre_simulate_outputs .= _grad[ix]
        PEtab._set_grad_x_nn_pre_simulate!(grad, simid, probinfo, model_info)
    end

    # Adjust if gradient is non-linear scale (e.g. log and log10). TODO: Refactor
    # this function later
    PEtab.grad_to_xscale!(grad, _grad, ∂G∂p, xdynamic, xindices, simid,
                          sensitivities_AD = false)
    return nothing
end
