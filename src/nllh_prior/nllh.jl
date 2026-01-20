function nllh(x::Vector{T}, probinfo::PEtabODEProblemInfo, model_info::ModelInfo,
              cids::Vector{Symbol}, hess::Bool, residuals::Bool)::T where {T <: Real}
    xdynamic, xobservable, xnoise, xnondynamic = split_x(x, model_info.xindices)
    nllh = nllh_solveode(xdynamic, xnoise, xobservable, xnondynamic, probinfo,
                         model_info; hess = hess, residuals = residuals, cids = cids)
    return nllh
end

function nllh_solveode(xdynamic::T1, xnoise::T2, xobservable::T2, xnondynamic::T2,
                       probinfo::PEtabODEProblemInfo, model_info::ModelInfo;
                       hess::Bool = false, residuals::Bool = false, cids = [:all],
                       grad_xdynamic::Bool = false)::Real where {T1 <: AbstractVector,
                                                                 T2 <: AbstractVector}
    xindices, cache = model_info.xindices, probinfo.cache
    # If the problem has been remade (e.g. for PEtab-select) the parameter order in
    # xdynamic must be corrected
    if grad_xdynamic == true && cache.nxdynamic[1] != length(xdynamic)
        _xdynamic = xdynamic[cache.xdynamic_output_order]
        xdynamic_ps = transform_x(_xdynamic, xindices, :xdynamic, cache)
    else
        xdynamic_ps = transform_x(xdynamic, xindices, :xdynamic, cache)
    end
    xnoise_ps = transform_x(xnoise, xindices, :xnoise, cache)
    xobservable_ps = transform_x(xobservable, xindices, :xobservable, cache)
    xnondynamic_ps = transform_x(xnondynamic, xindices, :xnondynamic, cache)

    derivative = hess || grad_xdynamic
    success = solve_conditions!(model_info, xdynamic_ps, probinfo; cids = cids,
                                dense_sol = false, save_observed_t = true,
                                derivative = derivative)
    if success != true
        if probinfo.solver.verbose == true
            @warn "Failed to solve ODE model."
        end
        return Inf
    end
    return _nllh(xnoise_ps, xobservable_ps, xnondynamic_ps, model_info, cids; hess = hess,
                 grad_xdynamic = grad_xdynamic, residuals = residuals)
end

function nllh_not_solveode(xnoise::T1, xobservable::T1, xnondynamic::T1,
                           probinfo::PEtabODEProblemInfo, model_info::ModelInfo;
                           grad_forward_AD::Bool = false, grad_adjoint::Bool = false,
                           grad_forward_eqs::Bool = false,
                           cids = [:all])::Real where {T1 <: AbstractVector}
    xindices, cache = model_info.xindices, probinfo.cache
    xnoise_ps = transform_x(xnoise, xindices, :xnoise, cache)
    xobservable_ps = transform_x(xobservable, xindices, :xobservable, cache)
    xnondynamic_ps = transform_x(xnondynamic, xindices, :xnondynamic, cache)

    return _nllh(xnoise_ps, xobservable_ps, xnondynamic_ps, model_info, cids,
                 grad_forward_AD = grad_forward_AD, grad_adjoint = grad_adjoint,
                 grad_forward_eqs = grad_forward_eqs)
end

function _nllh(xnoise::T, xobservable::T, xnondynamic::T, model_info::ModelInfo,
               cids::Vector{Symbol}; hess::Bool = false, grad_xdynamic::Bool = false,
               residuals::Bool = false, grad_forward_AD::Bool = false,
               grad_adjoint::Bool = false,
               grad_forward_eqs::Bool = false)::Real where {T <: AbstractVector}
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
                    grad_forward_eqs::Bool = false)::Real where {T <: AbstractVector}
    @unpack xindices, simulation_info, petab_measurements, petab_parameters, model = model_info
    if !(sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Terminated)
        return Inf
    end

    @unpack time, measurements, measurements_transformed, observable_id, noise_distributions = petab_measurements
    @unpack imeasurements, imeasurements_t_sol = simulation_info
    nominal_values = petab_parameters.nominal_value
    nllh = 0.0
    for imeasurement in imeasurements[cid]
        t = time[imeasurement]
        obsid = observable_id[imeasurement]
        noise_distribution = noise_distributions[imeasurement]

        # TODO: This must enter inside the _h and _sd function. Then I can meta-program
        # TODO: retreival of the observable itself.
        # grad_forward_eqs and grad_forward_AD are only true when we compute the gradient
        # via the nllh_not_solve (gradient for not ODE-system parameters), in this setting
        # only the ODESolution is required as Float, hence any dual must be converted
        if grad_forward_eqs || grad_forward_AD
            it = imeasurements_t_sol[imeasurement]
            u = sol[1:(model_info.nstates), it] .|> SBMLImporter._to_float
            p = sol.prob.p .|> SBMLImporter._to_float
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

        # Model observable and noise
        xnoise_maps = xindices.xnoise_maps[imeasurement]
        xobservable_maps = xindices.xobservable_maps[imeasurement]
        h = _h(u, t, p, xobservable, xnondynamic, model, xobservable_maps, obsid,
               nominal_values)
        h_transformed = _transform_h(h, noise_distribution)
        σ = _sd(u, t, p, xnoise, xnondynamic, model, xnoise_maps, obsid, nominal_values)

        residual = (h_transformed - measurements_transformed[imeasurement]) / σ
        update_petab_measurements!(petab_measurements, h, h_transformed, σ, residual,
                                   imeasurement)

        # By default a positive ODE solution is not enforced. Therefore it is possible
        # to get negative numbers in h_transformed which would throw an error
        if isinf(h_transformed)
            @warn "Transformed observable is non-finite for measurement \
                $imeasurement"  maxlog=20
            return Inf
        end
        if σ ≤ 0.0
            @warn "Computed noise σ ≤ 0 detected. σ must be > 0 for a valid likelihood, \
                 so Inf will be returned. It is recommended to adjust the noise formula \
                 in the PEtab observable to ensure σ > 0. This warning is likely due to \
                 ODE-solver round-off error and, if it occurs only rarely, can usually be \
                 safely ignored." maxlog=20
            return Inf
        end

        # For residuals == true we only care about residuals
        if residuals == false
            nllh += _nllh_obs(h, σ, measurements[imeasurement], noise_distribution)
        else
            nllh += residual
        end
    end
    return nllh
end

function _nllh_obs(h::Real, σ::Real, y::Float64, distribution::Symbol)::Real
    @unpack dist, transform = NOISE_DISTRIBUTIONS[distribution]
    _dist = dist(transform(h), σ)
    return logpdf(_dist, y) .* -1
end

function update_petab_measurements!(petab_measurements::PEtabMeasurements, h::T, hT::T,
                                    σ::Real, res::T,
                                    imeasurement::Integer)::Nothing where {T <:
                                                                           AbstractFloat}
    petab_measurements.simulated_values[imeasurement] = h
    mT = petab_measurements.measurements_transformed[imeasurement]
    petab_measurements.chi2_values[imeasurement] = (hT - mT)^2 / σ^2
    petab_measurements.residuals[imeasurement] = res
    return nothing
end
function update_petab_measurements!(::PEtabMeasurements, ::Any, ::Any, ::Any, ::Any, ::Any)::Nothing
    return nothing
end
