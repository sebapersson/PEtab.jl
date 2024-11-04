#=
    Functions for computing forward-sensitivities with SciMLSensitivity
=#

function PEtab._get_odeproblem_gradient(odeproblem::ODEProblem, gradient_method::Symbol,
                                        sensealg::ForwardAlg)::ODEProblem
    return ODEForwardSensitivityProblem(odeproblem.f, odeproblem.u0, odeproblem.tspan,
                                        odeproblem.p, sensealg = sensealg)
end

function PEtab.solve_sensitivites!(model_info::PEtab.ModelInfo,
                                   _solve_conditions!::Function,
                                   xdynamic_tot::Vector{<:AbstractFloat}, sensealg::ForwardAlg,
                                   probinfo::PEtab.PEtabODEProblemInfo,
                                   cids::Vector{Symbol}, cfg::Nothing,
                                   isremade::Bool = false)::Bool
    success = _solve_conditions!(xdynamic_tot, cids)
    return success
end

function PEtab._grad_forward_eqs_cond!(grad::Vector{T}, xdynamic_tot::Vector{T},
                                       xnoise::Vector{T}, xobservable::Vector{T},
                                       xnondynamic_mech::Vector{T}, icid::Int64,
                                       sensealg::ForwardAlg,
                                       probinfo::PEtab.PEtabODEProblemInfo,
                                       model_info::PEtab.ModelInfo)::Nothing where {T <:
                                                                                    AbstractFloat}
    @unpack xindices, simulation_info, model = model_info
    @unpack petab_parameters, petab_measurements = model_info
    @unpack imeasurements_t, tsaves = simulation_info
    cache = probinfo.cache

    # Simulation ids
    cid = simulation_info.conditionids[:experiment][icid]
    simid = simulation_info.conditionids[:simulation][icid]
    sol = simulation_info.odesols_derivatives[cid]

    # Partial derivatives needed for computing the gradient (derived from the chain-rule)
    ∂G∂u!, ∂G∂p! = PEtab._get_∂G∂_!(model_info, cid, xnoise, xobservable, xnondynamic_mech, cache.xnn_dict)

    p = sol.prob.p
    ∂G∂p, ∂G∂p_ = zeros(Float64, length(p)), zeros(Float64, length(p))
    ∂G∂u = zeros(Float64, length(PEtab._get_state_ids(model.sys_mutated)))
    _grad = zeros(Float64, length(p))
    for (it, tsave) in pairs(tsaves[cid])
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
    if !isempty(xindices.xids[:nn_pre_ode_outputs])
        ix = xindices.map_odeproblem.sys_to_nn_pre_ode_output
        cache.grad_nn_pre_ode .= _grad[ix]
        PEtab._grad_nn_pre_ode!(grad, simid, probinfo, model_info)
    end

    # Adjust if gradient is non-linear scale (e.g. log and log10). TODO: Refactor
    # this function later
    PEtab.grad_to_xscale!(grad, _grad, ∂G∂p, xdynamic_tot, xindices, simid,
                          sensitivites_AD = false)
    return nothing
end
