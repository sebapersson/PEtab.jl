# TODO: A lot similar with Sense equation
function _jac_residuals_xdynamic!(jac::AbstractMatrix, _solve_conditions!::Function,
                                  probinfo::PEtabODEProblemInfo,
                                  model_info::ModelInfo, cfg::ForwardDiff.JacobianConfig;
                                  cids::Vector{Symbol} = [:all],
                                  isremade::Bool = false)::Nothing
    @unpack cache, sensealg, reuse_sensitivities = probinfo
    @unpack xindices, simulation_info = model_info
    xnoise_ps = transform_x(cache.xnoise, xindices, :xnoise, cache)
    xobservable_ps = transform_x(cache.xobservable, xindices, :xobservable, cache)
    xnondynamic_ps = transform_x(cache.xnondynamic, xindices, :xnondynamic, cache)
    xdynamic_ps = transform_x(cache.xdynamic, xindices, :xdynamic, cache)

    if reuse_sensitivities == false
        success = solve_sensitivities!(model_info, _solve_conditions!, xdynamic_ps,
                                      :ForwardDiff, probinfo, cids, cfg, isremade)
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
                             icid, probinfo, model_info)
    end
    return nothing
end

function _jac_residuals_cond!(jac::AbstractMatrix{T}, xdynamic::Vector{T},
                              xnoise::Vector{T},
                              xobservable::Vector{T}, xnondynamic::Vector{T}, icid::Int64,
                              probinfo::PEtabODEProblemInfo,
                              model_info::ModelInfo) where {T <: AbstractFloat}
    @unpack xindices, simulation_info, model = model_info
    @unpack petab_parameters, petab_measurements = model_info
    @unpack imeasurements_t, tsaves, smatrixindices = simulation_info
    cache = probinfo.cache

    # Simulation ids
    cid = simulation_info.conditionids[:experiment][icid]
    simid = simulation_info.conditionids[:simulation][icid]
    smatrixindices_cid = smatrixindices[cid]
    ixdynamic_simid = _get_ixdynamic_simid(simid, xindices)
    sol = simulation_info.odesols_derivatives[cid]

    # Partial derivatives needed for computing the gradient (derived from the chain-rule)
    ∂G∂u!, ∂G∂p! = _get_∂G∂_!(probinfo, model_info, cid, xnoise, xobservable,
                              xnondynamic; residuals = true)

    nstates = model_info.nstates
    cache.p .= sol.prob.p .|> SBMLImporter._to_float
    @unpack p, u, ∂G∂p, ∂G∂p_, ∂G∂u, S, forward_eqs_grad = cache
    fill!(forward_eqs_grad, 0.0)
    fill!(∂G∂p, 0.0)
    for (it, tsave) in pairs(tsaves[cid])
        u .= sol[:, it] .|> SBMLImporter._to_float
        istart = (smatrixindices_cid[it] - 1) * nstates + 1
        iend = istart + nstates - 1
        _S = @view cache.S[istart:iend, ixdynamic_simid]
        for imeasurement in imeasurements_t[cid][it]
            ∂G∂u!(∂G∂u, u, p, tsave, 1, [[imeasurement]])
            ∂G∂p!(∂G∂p, u, p, tsave, 1, [[imeasurement]])
            @views forward_eqs_grad[ixdynamic_simid] .= transpose(_S) * ∂G∂u
            _jac = @view jac[:, imeasurement]
            grad_to_xscale!(_jac, forward_eqs_grad, ∂G∂p, xdynamic, xindices, simid,
                            sensitivities_AD = true)
        end
    end
    return nothing
end

# To compute the gradient for non-dynamic parameters
function residuals_not_solveode(residuals::T1, xnoise::T2, xobservable::T2, xnondynamic::T2,
                                probinfo::PEtabODEProblemInfo, model_info::ModelInfo;
                                cids::Vector{Symbol} = [:all])::T1 where {
                                                                          T1 <:
                                                                          AbstractVector,
                                                                          T2 <:
                                                                          AbstractVector}
    @unpack xindices, simulation_info = model_info
    cache = probinfo.cache
    xnoise_ps = transform_x(xnoise, xindices, :xnoise, cache)
    xobservable_ps = transform_x(xobservable, xindices, :xobservable, cache)
    xnondynamic_ps = transform_x(xnondynamic, xindices, :xnondynamic, cache)

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
                          cid::Symbol,
                          model_info::ModelInfo)::Bool where {T1 <: AbstractVector,
                                                              T2 <: AbstractVector}
    @unpack xindices, simulation_info, petab_measurements, petab_parameters, model = model_info
    sol = simulation_info.odesols_derivatives[cid]
    if !(sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Terminated)
        return false
    end

    @unpack time, measurements_transformed, noise_distributions, observable_id = petab_measurements
    @unpack imeasurements, imeasurements_t_sol = simulation_info
    nominal_values = petab_parameters.nominal_value
    for im in imeasurements[cid]
        t, obsid = time[im], observable_id[im]
        u = sol[:, imeasurements_t_sol[im]] .|> SBMLImporter._to_float
        p = sol.prob.p .|> SBMLImporter._to_float

        # Model observable and noise
        xnoise_maps = xindices.xnoise_maps[im]
        xobservable_maps = xindices.xobservable_maps[im]
        h = _h(u, t, p, xobservable, xnondynamic, model, xobservable_maps, obsid,
               nominal_values)
        h_transformed = _transform_h(h, noise_distributions[im])
        σ = _sd(u, t, p, xnoise, xnondynamic, model, xnoise_maps, obsid,
                nominal_values)

        y_transformed = measurements_transformed[im]
        residuals[im] = (h_transformed - y_transformed) / σ
    end
    return true
end
