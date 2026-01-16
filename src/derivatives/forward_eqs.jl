function _grad_forward_eqs!(grad::Vector{T}, _solve_conditions!::Function,
                            probinfo::PEtabODEProblemInfo, model_info::ModelInfo,
                            cfg::Union{ForwardDiff.JacobianConfig, Nothing};
                            cids::Vector{Symbol} = [:all])::Nothing where {T <: AbstractFloat}
    @unpack cache, sensealg = probinfo
    @unpack xindices, simulation_info = model_info
    xnoise, xobservable, xnondynamic_mech, xdynamic = _get_x_not_nn(cache, 1.0)
    xnoise_ps = transform_x(xnoise, xindices, :xnoise, cache)
    xobservable_ps = transform_x(xobservable, xindices, :xobservable, cache)
    xnondynamic_mech_ps = transform_x(xnondynamic_mech, xindices, :xnondynamic_mech, cache)
    xdynamic_ps = transform_x(xdynamic, xindices, :xdynamic, cache)

    # Solve the expanded ODE system for the sensitivites
    success = solve_sensitivites!(model_info, _solve_conditions!, xdynamic_ps, sensealg,
                                  probinfo, cids, cfg)
    if success != true
        @warn "Failed to solve sensitivity equations"
        fill!(grad, 0.0)
        return nothing
    end
    if isempty(xdynamic_ps)
        return nothing
    end

    fill!(grad, 0.0)
    for icid in eachindex(simulation_info.conditionids[:experiment])
        if cids[1] != :all && !(imulation_info.conditionids[:experiment][cid] in cids)
            continue
        end
        _grad_forward_eqs_cond!(grad, xdynamic_ps, xnoise_ps, xobservable_ps,
                                xnondynamic_mech_ps, icid, sensealg, probinfo, model_info)
    end
    return nothing
end

function solve_sensitivites!(model_info::ModelInfo, _solve_conditions!::Function,
                             xdynamic::Vector{<:AbstractFloat}, ::Symbol,
                             probinfo::PEtabODEProblemInfo, ::Vector{Symbol},
                             cfg::ForwardDiff.JacobianConfig)::Bool
    @unpack split_over_conditions, cache = probinfo
    @unpack simulation_info, xindices = model_info

    # Need to track for each condition and ForwardDiff chunk if the ODE could be solved
    simulation_info.could_solve[1] = true
    @unpack S, odesols = cache
    fill!(S, 0.0)

    if split_over_conditions == false
        # Need ODE solution for gradient for the non xdynamic parameters even when
        # xdynamic is empty
        if !isempty(xdynamic)
            ForwardDiff.jacobian!(S, _solve_conditions!, odesols, xdynamic, cfg)
        else
            _solve_conditions!(odesols, xdynamic)
        end
    end

    # Most efficient if xdynamic contains many parameters specific to a certain condition
    if split_over_conditions == true
        Stmp = similar(S)
        for (i, cid) in pairs(simulation_info.conditionids[:experiment])
            simid = simulation_info.conditionids[:simulation][i]
            ixdynamic_simid = _get_ixdynamic_simid(simid, xindices; nn_pre_simulate = false)
            xinput = _get_xinput(simid, xdynamic, ixdynamic_simid, model_info, probinfo)
            _S_condition! = (odesols, _xinput) -> begin
                _split_xinput!(probinfo, simid, model_info, _xinput, ixdynamic_simid)
                _xdynamic = get_tmp(probinfo.cache.xdynamic, _xinput)
                _solve_conditions!(odesols, _xdynamic, [cid])
            end
            ix_S_simid = _get_ix_S_simid(ixdynamic_simid, split_over_conditions, model_info)
            @views ForwardDiff.jacobian!(Stmp[:, ix_S_simid], _S_condition!, odesols,
                                         xinput)
            @views S[:, ix_S_simid] .+= Stmp[:, ix_S_simid]
        end
    end
    return simulation_info.could_solve[1]
end

function _grad_forward_eqs_cond!(grad::Vector{T}, xdynamic::Vector{T}, xnoise::Vector{T},
                                 xobservable::Vector{T}, xnondynamic_mech::Vector{T},
                                 icid::Int64, ::Symbol,
                                 probinfo::PEtabODEProblemInfo,
                                 model_info::ModelInfo)::Nothing where {T <: AbstractFloat}
    @unpack xindices, simulation_info, model = model_info
    @unpack petab_parameters, petab_measurements = model_info
    @unpack imeasurements_t, tsaves_no_cbs, smatrixindices = simulation_info
    @unpack split_over_conditions, cache = probinfo

    # Simulation ids
    cid = simulation_info.conditionids[:experiment][icid]
    simid = simulation_info.conditionids[:simulation][icid]
    smatrixindices_cid = smatrixindices[cid]
    nn_pre_simulate = probinfo.split_over_conditions == false
    ixdynamic_simid = _get_ixdynamic_simid(simid, xindices, nn_pre_simulate = nn_pre_simulate)
    ix_S_simid = _get_ix_S_simid(ixdynamic_simid, split_over_conditions, model_info)
    sol = simulation_info.odesols_derivatives[cid]

    # Partial derivatives needed for computing the gradient (derived from the chain-rule)
    ∂G∂u!, ∂G∂p! = _get_∂G∂_!(model_info, cid, xnoise, xobservable, xnondynamic_mech,
                              cache.x_ml_models, cache.x_ml_models_constant)

    nstates = model_info.nstates
    cache.p .= sol.prob.p .|> SBMLImporter._to_float
    @unpack p, u, ∂G∂p, ∂G∂p_, ∂G∂u, S, forward_eqs_grad = cache
    fill!(forward_eqs_grad, 0.0)
    fill!(∂G∂p, 0.0)
    for (it, tsave) in pairs(tsaves_no_cbs[cid])
        u .= sol[:, it] .|> SBMLImporter._to_float
        ∂G∂u!(∂G∂u, u, p, tsave, it)
        ∂G∂p!(∂G∂p_, u, p, tsave, it)
        # Computations generate a big sensitivity matrix across all conditions, where each
        # row is an observation at a specific time point. Positions are precomputed in
        # smatrixindices_cid
        istart = (smatrixindices_cid[it] - 1) * nstates + 1
        iend = istart + nstates - 1
        _S = @view S[istart:iend, ix_S_simid]
        @views forward_eqs_grad[ix_S_simid] .+= transpose(_S) * ∂G∂u
        ∂G∂p .+= ∂G∂p_
    end

    # Gradient of ML parameters
    if split_over_conditions == true && !isempty(cache.grad_ml_pre_simulate_outputs)
        cache.grad_ml_pre_simulate_outputs .= forward_eqs_grad[(length(ixdynamic_simid)+1):end]
        _set_grad_x_nn_pre_simulate!(grad, simid, probinfo, model_info)
    end

    # Adjust if gradient is non-linear scale (e.g. log and log10).
    grad_to_xscale!(grad, forward_eqs_grad, ∂G∂p, xdynamic, xindices, simid;
                    sensitivities_AD = true, nn_pre_simulate = nn_pre_simulate)
    return nothing
end

function _get_ix_S_simid(ixdynamic_simid, split_over_conditions::Bool, model_info::ModelInfo)
    if split_over_conditions == false
        return ixdynamic_simid[:]
    end
    nx_forward_eqs = _get_nx_forward_eqs(model_info.xindices, split_over_conditions)
    nx_nn_pre_simulate_outputs = length(model_info.xindices.xids[:sys_ml_pre_simulate_outputs])
    istart = nx_forward_eqs - nx_nn_pre_simulate_outputs + 1
    return vcat(ixdynamic_simid, istart:nx_forward_eqs)
end
