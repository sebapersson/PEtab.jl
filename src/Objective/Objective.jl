function nllh(x::Vector{T}, probleminfo::PEtabODEProblemInfo, model_info::ModelInfo,
              cids::Vector{Symbol}, hess::Bool, residuals::Bool)::T where {T <: Real}
    # TODO: Use DiffCache for highest level
    xdynamic, xobservable, xnoise, xnondynamic = split_x(x, model_info.θ_indices)
    nllh = nllh_solveode(xdynamic, xnoise, xobservable, xnondynamic, probleminfo,
                         model_info; hess = hess, residuals = residuals, cids = cids)

    # TODO : Must refactor prior handling
    @unpack θ_indices, prior_info = model_info
    if prior_info.has_priors == true && hess == false
        x_ps = transform_x(x, θ_indices.xids[:estimate], θ_indices)
        nllh -= compute_priors(x, x_ps, θ_indices.xids[:estimate], prior_info) # We work with -loglik
    end
    return nllh
end

function nllh_solveode(xdynamic::T1, xnoise::T2, xobservable::T2, xnondynamic::T2,
                       probleminfo::PEtabODEProblemInfo, model_info::ModelInfo;
                       hess::Bool = false, residuals::Bool = false, cids = [:all],
                       grad_xdynamic::Bool = false)::Real where {T1 <: AbstractVector,
                                                                 T2 <: AbstractVector}
    θ_indices, cache = model_info.θ_indices, probleminfo.cache
    # If the problem has been remade (e.g. for PEtab-select) the parameter order in
    # xdynamic must be corrected
    if grad_xdynamic == true && cache.nxdynamic[1] != length(xdynamic)
        _xdynamic = xdynamic[cache.xdynamic_output_order]
        xdynamic_ps = transform_x(_xdynamic, θ_indices, :xdynamic, cache)
    else
        xdynamic_ps = transform_x(xdynamic, θ_indices, :xdynamic, cache)
    end
    xnoise_ps = transform_x(xnoise, θ_indices, :xnoise, cache)
    xobservable_ps = transform_x(xobservable, θ_indices, :xobservable, cache)
    xnondynamic_ps = transform_x(xnondynamic, θ_indices, :xnondynamic, cache)

    derivative = hess || grad_xdynamic
    success = solve_conditions!(model_info, xdynamic_ps, probleminfo; cids = cids,
                                dense_sol = false, save_observed_t = true,
                                derivative = derivative)
    if success != true
        if probleminfo.solver.verbose == true
            @warn "Failed to solve ODE model."
        end
        return Inf
    end
    return _nllh(xnoise_ps, xobservable_ps, xnondynamic_ps, model_info, cids; hess = hess, grad_xdynamic = grad_xdynamic, residuals = residuals)
end

function nllh_not_solveode(xnoise::T1, xobservable::T1, xnondynamic::T1,
                           probleminfo::PEtabODEProblemInfo, model_info::ModelInfo;
                           grad_forward_AD::Bool = false, grad_adjoint::Bool = false,
                           grad_forward_eqs::Bool = false, cids = [:all])::Real where T1 <: AbstractVector
    θ_indices, cache = model_info.θ_indices, probleminfo.cache
    xnoise_ps = transform_x(xnoise, θ_indices, :xnoise, cache)
    xobservable_ps = transform_x(xobservable, θ_indices, :xobservable, cache)
    xnondynamic_ps = transform_x(xnondynamic, θ_indices, :xnondynamic, cache)

    return _nllh(xnoise_ps, xobservable_ps, xnondynamic_ps, model_info, cids,
                 grad_forward_AD = grad_forward_AD, grad_adjoint = grad_adjoint,
                 grad_forward_eqs = grad_forward_eqs)
end

function _nllh(xnoise::T, xobservable::T, xnondynamic::T, model_info::ModelInfo,
               cids::Vector{Symbol}; hess::Bool = false, grad_xdynamic::Bool = false,
               residuals::Bool = false, grad_forward_AD::Bool = false,
               grad_adjoint::Bool = false, grad_forward_eqs::Bool = false)::Real where T <: AbstractVector
    simulation_info = model_info.simulation_info
    derivative = any((hess, grad_adjoint, grad_forward_AD, grad_forward_eqs, grad_xdynamic))
    if derivative == true
        odesols = simulation_info.odesols_derivatives
    else
        odesols = simulation_info.odesols
    end

    nllh = 0.0
    for cid in simulation_info.conditionids[:experiment]
        if cids[1] != :all && !(cid in cids)
            continue
        end

        sol = odesols[cid]
        nllh += _nllh_cond(sol, xnoise, xobservable, xnondynamic, cid, model_info;
                           grad_adjoint = grad_adjoint, grad_forward_AD = grad_forward_AD,
                           grad_forward_eqs = grad_forward_eqs, residuals = residuals)
        if isinf(nllh)
            return nllh
        end
    end
    return nllh
end

function _nllh_cond(sol::ODESolution, xnoise::T, xobservable::T, xnondynamic::T,
                    cid::Symbol, model_info::ModelInfo; residuals::Bool = false,
                    grad_forward_AD::Bool = false, grad_adjoint::Bool = false,
                    grad_forward_eqs::Bool = false)::Real where T <: AbstractVector
    @unpack θ_indices, simulation_info, measurement_info, parameter_info, petab_model = model_info
    if !(sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Terminated)
        return Inf
    end

    @unpack time, measurement_transforms = measurement_info
    ys_transformed = measurement_info.measurementT
    @unpack imeasurements, imeasurements_t_sol = simulation_info
    # TODO: Should live in PEtabModel
    nstates = length(states(petab_model.sys_mutated))
    nllh = 0.0
    for imeasurement in imeasurements[cid]
        t = time[imeasurement]
        # grad_forward_eqs and grad_forward_AD are only true when we compute the gradient
        # via the nllh_not_solve (gradient for not ODE-system parameters), in this setting
        # only the ODESolution is required as Float, hence any dual must be converted
        if grad_forward_eqs || grad_forward_AD
            it = imeasurements_t_sol[imeasurement]
            u = sol[1:nstates, it] .|> dual_to_float
            p = sol.prob.p .|> dual_to_float
        # For adjoint sensitivity analysis the ODESolution is dense
        elseif grad_adjoint == true
            # In case we only have sol.t = 0.0 (or similar) interpolation does not work
            u = length(sol.t) > 1 ? sol(t) : sol[1]
            p = sol.prob.p
        # Default nice case
        else
            it = imeasurements_t_sol[imeasurement]
            u = sol[:, it]
            p = sol.prob.p
        end

        # TODO Ideally refactor when get to computeh
        y_transformed = ys_transformed[imeasurement]
        h = computeh(u, t, p, xobservable, xnondynamic, petab_model, imeasurement,
                       measurement_info, θ_indices, parameter_info)
        h_transformed = transform_measurement_or_h(h, measurement_transforms[imeasurement])
        σ = computeσ(u, t, p, xnoise, xnondynamic, petab_model, imeasurement,
                     measurement_info, θ_indices, parameter_info)
        residual = (h_transformed - y_transformed) / σ

        # TODO: Should not belong here, but refactor later
        update_measurement_info!(measurement_info, h, h_transformed, σ, residual, imeasurement)

        # By default a positive ODE solution is not enforced. Therefore it is possible
        # to get negative numbers in h_transformed which would throw an error
        if isinf(h_transformed)
            @warn "Transformed observable is non-finite for measurement $imeasurement"
            return Inf
        end
        if σ < 0.0
            @warn "Measurement noise σ is smaller than 0. Consider changing the noise " *
                  "formula so it cannot go below zero. This issue likelly happens due " *
                  "to numerical noise when solving the ODE" maxlog=10
            return Inf
        end

        # For residuals == true we only care about residuals
        if residuals == false
            if measurement_transforms[imeasurement] === :lin
                nllh += log(σ) + 0.5 * log(2π) + 0.5 * residual^2
            elseif measurement_transforms[imeasurement] === :log10
                nllh += log(σ) + 0.5 * log(2π) + log(log(10)) + log(10) * y_transformed + 0.5 * residual^2
            elseif measurement_transforms[imeasurement] === :log
                nllh += log(σ) + 0.5 * log(2π) + y_transformed + 0.5 * residual^2
            end
        else
            nllh += residual
        end
    end
    return nllh
end

function update_measurement_info!(measurement_info::MeasurementsInfo, h::T, hT::T, σ::T,
                                  res::T, imeasurement::Integer)::Nothing where {T <: AbstractFloat}
    ChainRulesCore.@ignore_derivatives begin
        measurement_info.simulated_values[imeasurement] = h
        mT = measurement_info.measurementT
        measurement_info.chi2_values[imeasurement] = (hT - mT[imeasurement])^2 / σ^2
        measurement_info.residuals[imeasurement] = res
    end
    return nothing
end
function update_measurement_info!(measurement_info::MeasurementsInfo, h, hT, σ, residual,
                                  imeasurement)::Nothing
    return nothing
end
