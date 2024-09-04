function _grad_forward_eqs!(grad::Vector{T}, _solve_conditions!::Function,
                            probleminfo::PEtabODEProblemInfo, model_info::ModelInfo,
                            cfg::Union{ForwardDiff.JacobianConfig, Nothing};
                            cids::Vector{Symbol} = [:all], isremade::Bool = false)::Nothing where T <: AbstractFloat
    @unpack cache, sensealg = probleminfo
    @unpack θ_indices, simulation_info = model_info
    xnoise_ps = transform_x(cache.xnoise, θ_indices, :xnoise, cache)
    xobservable_ps = transform_x(cache.xobservable, θ_indices, :xobservable, cache)
    xnondynamic_ps = transform_x(cache.xnondynamic, θ_indices, :xnondynamic, cache)
    xdynamic_ps = transform_x(cache.xdynamic, θ_indices, :xdynamic, cache)

    # Solve the expanded ODE system for the sensitivites
    success = solve_sensitivites!(model_info, _solve_conditions!, xdynamic_ps, sensealg,
                                  probleminfo, cids, cfg, isremade)
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
        _grad_forward_eqs_cond!(grad, xdynamic_ps, xnoise_ps, xobservable_ps, xnondynamic_ps,
                                icid, sensealg, probleminfo, model_info)
    end
    return nothing
end

function solve_sensitivites!(model_info::ModelInfo, _solve_conditions!::Function,
                             xdynamic::Vector{<:AbstractFloat}, sensealg::Symbol,
                             probleminfo::PEtabODEProblemInfo, cids::Vector{Symbol},
                             cfg::ForwardDiff.JacobianConfig, isremade::Bool = false)::Bool
    @unpack split_over_conditions, cache = probleminfo
    @unpack simulation_info, θ_indices = model_info

    # Need to track for each condition and ForwardDiff chunk if the ODE could be solved
    simulation_info.could_solve[1] = true
    @unpack S, odesols, nxdynamic, xdynamic_grad = cache
    fill!(S, 0.0)

    if split_over_conditions == false
        # remade = false, no parameters in xdynamic are fixed, but for computations to
        # work nxdynamic must be set to default value temporarly
        if isremade == false || length(xdynamic_grad) == nxdynamic[1]
            tmp = cache.nxdynamic[1]
            cache.nxdynamic[1] = length(xdynamic)
            # Need ODE solution for gradient for the non xdynamic parameters even when
            # xdynamic is empty
            if !isempty(xdynamic)
                ForwardDiff.jacobian!(S, _solve_conditions!, odesols, xdynamic, cfg)
            else
                _solve_conditions!(cache.odesols, xdynamic)
            end
            cache.nxdynamic[1] = tmp
        end

        # Subset of dynamic parameters fixed, and Forward-AD can be run with fewer chunks
        if !(isremade == false || length(xdynamic_grad) == nxdynamic[1])
            # As above, must always obtain an ODESolution
            if cache.nxdynamic[1] != 0
                C = length(cfg.seeds)
                chunk = ForwardDiff.Chunk(C)
                nforward_passes = Int64(ceil(cache.nxdynamic[1] / C))
                _xdynamic = xdynamic[cache.xdynamic_input_order]
                forwarddiff_jacobian_chunks(_solve_conditions!, odesols, S, _xdynamic,
                                            chunk; nforward_passes = nforward_passes)
                @views S .= cache.S[:, cache.xdynamic_output_order]
            else
                _solve_conditions!(cache.odesols, xdynamic)
            end
        end
    end

    # Most efficient if xdynamic contains many parameters specific to a certain condition
    if split_over_conditions == true
        Stmp = similar(S)
        for (i, cid) in pairs(simulation_info.conditionids[:experiment])
            simid = simulation_info.conditionids[:simulation][i]
            ixdynamic_simid = _get_ixdynamic_simid(simid, θ_indices)
            _xinput = xdynamic[ixdynamic_simid]
            _S_condition! = (odesols, x) -> begin
                _xdynamic = convert.(eltype(x), xdynamic)
                _xdynamic[ixdynamic_simid] .= x
                _solve_conditions!(odesols, _xdynamic, [cid])
            end
            @views ForwardDiff.jacobian!(Stmp[:, ixdynamic_simid], _S_condition!, odesols,
                                         _xinput)
            @views S[:, ixdynamic_simid] .+= Stmp[:, ixdynamic_simid]
        end
    end
    return simulation_info.could_solve[1]
end

function _grad_forward_eqs_cond!(grad::Vector{T}, xdynamic::Vector{T}, xnoise::Vector{T},
                                 xobservable::Vector{T}, xnondynamic::Vector{T},
                                 icid::Int64, sensealg::Symbol, probleminfo::PEtabODEProblemInfo,
                                 model_info::ModelInfo)::Nothing where T <: AbstractFloat
    @unpack θ_indices, simulation_info, petab_model = model_info
    @unpack parameter_info, measurement_info = model_info
    @unpack imeasurements_t, tsaves, smatrixindices = simulation_info
    cache = probleminfo.cache

    # Simulation ids
    cid = simulation_info.conditionids[:experiment][icid]
    simid = simulation_info.conditionids[:simulation][icid]
    smatrixindices_cid = smatrixindices[cid]
    ixdynamic_simid = _get_ixdynamic_simid(simid, θ_indices)
    sol = simulation_info.odesols_derivatives[cid]

    # Partial derivatives needed for computing the gradient (derived from the chain-rule)
    ∂G∂u!, ∂G∂p! = _get_∂G∂_!(probleminfo, model_info, cid, xnoise, xobservable,
                              xnondynamic)

    nstates = length(unknowns(petab_model.sys_mutated))
    cache.p .= sol.prob.p .|> SBMLImporter._to_float
    @unpack p, u, ∂G∂p, ∂G∂p_, ∂G∂u, S, forward_eqs_grad = cache
    fill!(forward_eqs_grad, 0.0)
    fill!(∂G∂p, 0.0)
    for (it, tsave) in pairs(tsaves[cid])
        u .= sol[:, it] .|> SBMLImporter._to_float
        ∂G∂u!(∂G∂u, u, p, tsave, it)
        ∂G∂p!(∂G∂p_, u, p, tsave, it)
        # Computations generate a big sensitivity matrix across all conditions, where each
        # row is an observation at a specific time point. Positions are precomputed in
        # smatrixindices_cid
        istart = (smatrixindices_cid[it] - 1) * nstates + 1
        iend = istart + nstates - 1
        _S = @view S[istart:iend, ixdynamic_simid]
        @views forward_eqs_grad[ixdynamic_simid] .+= transpose(_S) * ∂G∂u
        ∂G∂p .+= ∂G∂p_
    end

    # Adjust if gradient is non-linear scale (e.g. log and log10). TODO: Refactor
    # this function later
    grad_to_xscale!(grad, forward_eqs_grad, ∂G∂p, xdynamic, θ_indices, simid,
                    sensitivites_AD = true)
    return nothing
end
