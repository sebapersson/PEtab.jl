# TODO: A lot similar with Sense equation
function _jac_residuals_xdynamic!(jac::AbstractMatrix, _solve_conditions!::Function,
                                  probleminfo::PEtabODEProblemInfo,
                                  model_info::ModelInfo, cfg::ForwardDiff.JacobianConfig;
                                  cids::Vector{Symbol} = [:all], isremade::Bool = false)::Nothing
    @unpack cache, sensealg, reuse_sensitivities = probleminfo
    @unpack θ_indices, simulation_info = model_info
    xnoise_ps = transform_x(cache.xnoise, θ_indices, :xnoise, cache)
    xobservable_ps = transform_x(cache.xobservable, θ_indices, :xobservable, cache)
    xnondynamic_ps = transform_x(cache.xnondynamic, θ_indices, :xnondynamic, cache)
    xdynamic_ps = transform_x(cache.xdynamic, θ_indices, :xdynamic, cache)

    if reuse_sensitivities == false
        success = solve_sensitivites!(model_info, _solve_conditions!, xdynamic_ps,
                                      :ForwardDiff, probleminfo, cids, cfg, isremade)
        if success != true
            @warn "Failed to solve sensitivity equations"
            fill!(jac, 0.0)
            return nothing
        end
    end
    if isempty(xdynamic_ps)
        fill!(jac, 0.0)
        return nothing
    end

    # Compute the gradient by looping through all experimental conditions.
    for icid in eachindex(simulation_info.conditionids[:experiment])
        if cids[1] != :all && !(imulation_info.conditionids[:experiment][icid] in cids)
            continue
        end
        _jac_residuals_cond!(jac, xdynamic_ps, xnoise_ps, xobservable_ps, xnondynamic_ps,
                             icid, probleminfo, model_info)
    end
    return nothing
end

function _jac_residuals_cond!(jac::AbstractMatrix{T}, xdynamic::Vector{T}, xnoise::Vector{T},
                              xobservable::Vector{T}, xnondynamic::Vector{T}, icid::Int64,
                              probleminfo::PEtabODEProblemInfo, model_info::ModelInfo) where T <: AbstractFloat
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
                              xnondynamic; residuals = true)

    nstates = length(states(petab_model.sys_mutated))
    cache.p .= sol.prob.p .|> dual_to_float
    @unpack p, u, ∂G∂p, ∂G∂p_, ∂G∂u, S, forward_eqs_grad = cache
    fill!(forward_eqs_grad, 0.0)
    fill!(∂G∂p, 0.0)
    for (it, tsave) in pairs(tsaves[cid])
        u .= sol[:, it] .|> dual_to_float
        istart = (smatrixindices_cid[it] - 1) * nstates + 1
        iend = istart + nstates - 1
        _S = @view cache.S[istart:iend, ixdynamic_simid]
        for imeasurement in imeasurements_t[cid][it]
            ∂G∂u!(∂G∂u, u, p, tsave, 1, [[imeasurement]])
            ∂G∂p!(∂G∂p, u, p, tsave, 1, [[imeasurement]])
            @views forward_eqs_grad[ixdynamic_simid] .= transpose(_S) * ∂G∂u
            _jac = @view jac[:, imeasurement]
            adjust_gradient_θ_transformed!(_jac, forward_eqs_grad, ∂G∂p, xdynamic,
                                           θ_indices, simid, autodiff_sensitivites = true)
        end
    end
    return nothing
end

# To compute the gradient for non-dynamic parameters
function residuals_not_solveode(residuals::T1, xnoise::T2, xobservable::T2, xnondynamic::T2,
                                probleminfo::PEtabODEProblemInfo, model_info::ModelInfo;
                                cids::Vector{Symbol} = [:all])::T1 where {T1 <: AbstractVector,
                                                                          T2 <: AbstractVector}
    @unpack θ_indices, simulation_info = model_info
    cache = probleminfo.cache
    xnoise_ps = transform_x(xnoise, θ_indices, :xnoise, cache)
    xobservable_ps = transform_x(xobservable, θ_indices, :xobservable, cache)
    xnondynamic_ps = transform_x(xnondynamic, θ_indices, :xnondynamic, cache)

    for cid in simulation_info.conditionids[:experiment]
        if cids[1] != :all && !(cid in cids)
            continue
        end
        _residuals_cond!(residuals, xnoise_ps, xobservable_ps, xnondynamic_ps, cid,
                         model_info)
        if success == false
            fill!(residuals, Inf)
            break
        end
    end
    return residuals
end

# For an experimental condition compute residuals
function _residuals_cond!(residuals::T1, xnoise::T2, xobservable::T2, xnondynamic::T2,
                          cid::Symbol, model_info::ModelInfo)::Bool where {T1 <: AbstractVector,
                                                                           T2 <: AbstractVector}
    @unpack θ_indices, simulation_info, measurement_info, parameter_info, petab_model = model_info
    sol = simulation_info.odesols_derivatives[cid]
    if !(sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Terminated)
        return false
    end

    @unpack time, measurement_transforms = measurement_info
    ys_transformed = measurement_info.measurementT
    @unpack imeasurements, imeasurements_t_sol = simulation_info
    for imeasurement in imeasurements[cid]
        t = time[imeasurement]
        u = sol[:, imeasurements_t_sol[imeasurement]] .|> dual_to_float
        p = sol.prob.p .|> dual_to_float

        y_transformed = ys_transformed[imeasurement]
        h = computeh(u, t, p, xobservable, xnondynamic, petab_model, imeasurement,
                       measurement_info, θ_indices, parameter_info)
        h_transformed = transform_measurement_or_h(h, measurement_transforms[imeasurement])
        σ = computeσ(u, t, p, xnoise, xnondynamic, petab_model, imeasurement,
                     measurement_info, θ_indices, parameter_info)
        residuals[imeasurement] = (h_transformed - y_transformed) / σ
    end
    return true
end
