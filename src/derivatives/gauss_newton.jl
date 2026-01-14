# TODO: A lot similar with Sense equation
function _jac_residuals_xdynamic!(jac::AbstractMatrix, _solve_conditions!::Function,
                                  probinfo::PEtabODEProblemInfo,
                                  model_info::ModelInfo, cfg::ForwardDiff.JacobianConfig;
                                  cids::Vector{Symbol} = [:all])::Nothing
    @unpack cache, sensealg, reuse_sensitivities = probinfo
    @unpack xindices, simulation_info = model_info
    xnoise, xobservable, xnondynamic_mech, xdynamic = _get_x_not_nn(cache, 1.0)
    xnoise_ps = transform_x(xnoise, xindices, :xnoise, cache)
    xobservable_ps = transform_x(xobservable, xindices, :xobservable, cache)
    xnondynamic_mech_ps = transform_x(xnondynamic_mech, xindices, :xnondynamic_mech, cache)
    xdynamic_tot_ps = transform_x(xdynamic, xindices, :xdynamic_tot, cache)

    if reuse_sensitivities == false
        success = solve_sensitivites!(model_info, _solve_conditions!, xdynamic_tot_ps,
                                      :ForwardDiff, probinfo, cids, cfg)
        if success != true
            @warn "Failed to solve sensitivity equations"
            fill!(jac, 0.0)
            return nothing
        end
    end
    if isempty(xdynamic_tot_ps)
        fill!(jac, 0.0)
        return nothing
    end

    # Compute the gradient by looping through all experimental conditions.
    for icid in eachindex(simulation_info.conditionids[:experiment])
        if cids[1] != :all && !(imulation_info.conditionids[:experiment][icid] in cids)
            continue
        end
        _jac_residuals_cond!(jac, xdynamic_tot_ps, xnoise_ps, xobservable_ps,
                             xnondynamic_mech_ps, icid, probinfo, model_info)
    end
    return nothing
end

function _jac_residuals_cond!(jac::AbstractMatrix{T}, xdynamic_tot::Vector{T}, xnoise::Vector{T}, xobservable::Vector{T}, xnondynamic_mech::Vector{T}, icid::Int64, probinfo::PEtabODEProblemInfo, model_info::ModelInfo) where {T <: AbstractFloat}
    @unpack xindices, simulation_info, model = model_info
    @unpack petab_parameters, petab_measurements = model_info
    @unpack imeasurements_t, tsaves, smatrixindices = simulation_info
    @unpack cache, split_over_conditions = probinfo

    # Simulation ids
    cid = simulation_info.conditionids[:experiment][icid]
    simid = simulation_info.conditionids[:simulation][icid]
    smatrixindices_cid = smatrixindices[cid]
    nn_preode = probinfo.split_over_conditions == false
    ixdynamic_simid = _get_ixdynamic_simid(simid, xindices, nn_preode = nn_preode)
    ix_S_simid = _get_ix_S_simid(ixdynamic_simid, split_over_conditions, model_info)
    sol = simulation_info.odesols_derivatives[cid]

    # Partial derivatives needed for computing the gradient (derived from the chain-rule)
    ∂G∂u!, ∂G∂p! = _get_∂G∂_!(model_info, cid, xnoise, xobservable, xnondynamic_mech, cache.xnn_dict, cache.xnn_constant; residuals = true)

    nstates = model_info.nstates
    cache.p .= sol.prob.p .|> SBMLImporter._to_float
    @unpack p, u, ∂G∂p, ∂G∂p_, ∂G∂u, S, forward_eqs_grad = cache
    fill!(forward_eqs_grad, 0.0)
    fill!(∂G∂p, 0.0)
    for (it, tsave) in pairs(tsaves[cid])
        u .= sol[:, it] .|> SBMLImporter._to_float
        istart = (smatrixindices_cid[it] - 1) * nstates + 1
        iend = istart + nstates - 1
        _S = @view cache.S[istart:iend, ix_S_simid]
        for imeasurement in imeasurements_t[cid][it]
            ∂G∂u!(∂G∂u, u, p, tsave, 1, [[imeasurement]])
            ∂G∂p!(∂G∂p, u, p, tsave, 1, [[imeasurement]])
            @views forward_eqs_grad[ix_S_simid] .= transpose(_S) * ∂G∂u
            _jac = @view jac[:, imeasurement]
            # In contrast to gradient functions, need to compute gradient/sensitivity
            # for neural-net pre-ODE parameters per time-point to retreive a correct
            # Jacobian for Gauss-Newton
            if split_over_conditions == true
                ix = (length(ixdynamic_simid)+1):length(forward_eqs_grad)
                cache.grad_nn_preode .= forward_eqs_grad[ix]
                _set_grad_x_nn_preode!(_jac, simid, probinfo, model_info)
            end
            grad_to_xscale!(_jac, forward_eqs_grad, ∂G∂p, xdynamic_tot, xindices, simid,
                            sensitivities_AD = true, nn_preode = nn_preode)
        end
    end
    return nothing
end

# To compute the gradient for non-dynamic parameters
function residuals_not_solveode(residuals::T1, xnoise::T2, xobservable::T2, xnondynamic_mech::T2, xnn::Dict{Symbol, ComponentArray}, probinfo::PEtabODEProblemInfo, model_info::ModelInfo; cids::Vector{Symbol} = [:all])::T1 where {T1 <: AbstractVector, T2 <: AbstractVector}
    @unpack xindices, simulation_info = model_info
    cache = probinfo.cache
    xnoise_ps = transform_x(xnoise, xindices, :xnoise, cache)
    xobservable_ps = transform_x(xobservable, xindices, :xobservable, cache)
    xnondynamic_mech_ps = transform_x(xnondynamic_mech, xindices, :xnondynamic_mech, cache)

    for cid in simulation_info.conditionids[:experiment]
        if cids[1] != :all && !(cid in cids)
            continue
        end
        _residuals_cond!(residuals, xnoise_ps, xobservable_ps, xnondynamic_mech_ps, xnn, cache.xnn_constant, cid, model_info)
        if success == false
            fill!(residuals, Inf)
            break
        end
    end
    return residuals
end

# For an experimental condition compute residuals
function _residuals_cond!(residuals::T1, xnoise::T2, xobservable::T2, xnondynamic_mech::T2, xnn::Dict{Symbol, ComponentArray}, xnn_constant, cid::Symbol, model_info::ModelInfo)::Bool where {T1 <: AbstractVector, T2 <: AbstractVector}
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

        xnoise_maps = xindices.xnoise_maps[im]
        xobservable_maps = xindices.xobservable_maps[im]
        h = _h(u, t, p, xobservable, xnondynamic_mech, xnn, xnn_constant, model, xobservable_maps, obsid, nominal_values)
        h_transformed = _transform_h(h, noise_distributions[im])
        σ = _sd(u, t, p, xnoise, xnondynamic_mech, xnn, xnn_constant, model, xnoise_maps, obsid, nominal_values)

        y_transformed = measurements_transformed[im]
        residuals[im] = (h_transformed - y_transformed) / σ
    end
    return true
end
