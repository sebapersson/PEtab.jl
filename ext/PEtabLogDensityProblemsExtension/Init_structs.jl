function PEtab.PEtabLogDensity(petab_problem::PEtabODEProblem)::PEtab.PEtabLogDensity
    inference_info = PEtab.InferenceInfo(petab_problem)
    @unpack nllh, nllh_grad, nparameters_estimate = petab_problem

    # For via autodiff compute the gradient of the prior and Jacobian correction
    _prior_correction = (x_inference) -> let inference_info = inference_info
        prior = PEtab.compute_prior(x_inference, inference_info)
        correction = Bijectors.logabsdetjac(inference_info.inv_bijectors, x_inference)
        return prior + correction
    end

    logtarget = (x_inference) -> let nllh = nllh, inference_info = inference_info
        _logtarget(x_inference, nllh, inference_info)
    end

    logtarget_gradient = (x_inference) -> let nllh_grad = nllh_grad,
        inference_info = inference_info, _prior_correction = _prior_correction

        _logtarget_gradient(x_inference, nllh_grad, _prior_correction, inference_info)
    end

    initial_value = Vector{Float64}(undef, nparameters_estimate)

    return PEtab.PEtabLogDensity(inference_info, logtarget, logtarget_gradient,
                                 initial_value, nparameters_estimate)
end

function PEtab.InferenceInfo(petab_problem::PEtabODEProblem)::PEtab.InferenceInfo
    @unpack model_info, lower_bounds, upper_bounds, xnominal = petab_problem
    @unpack priors, petab_parameters = model_info

    parameter_names = Symbol.(ComponentArrays.labels(xnominal))
    n_parameters = length(parameter_names)

    priors_dist = Vector{PEtab.ContDistribution}(undef, n_parameters)
    bijectors = Vector(undef, n_parameters)
    priors_scale = similar(parameter_names)
    parameters_scale = similar(parameter_names)

    for (ix, θ) in pairs(parameter_names)
        # ML parameters are always on linear scale
        if ix in model_info.xindices.indices_est[:est_to_mech]
            iθ = findfirst(x -> x == θ, petab_parameters.parameter_id)
            parameters_scale[ix] = petab_parameters.parameter_scale[iθ]
        else
            parameters_scale[ix] = :lin
        end

        # In case the parameter lacks a defined prior we default to a Uniform
        # on parameter scale with lb and ub as bounds
        if !in(ix, priors.ix_prior)
            if abs(isinf(lower_bounds[ix])) || isinf(upper_bounds[ix])
                @warn "Lower or upper bounds for parameter $(parameter_names[ix]) is \
                    -inf and/or inf. Assigning Uniform(1e-3, 1e3) prior"
                priors_dist[ix] = Uniform(1e-3, 1e3)
            else
                priors_dist[ix] = Uniform(lower_bounds[ix], upper_bounds[ix])
            end
            priors_scale[ix] = :lin
        else
            jx = findfirst(x -> x == ix, priors.ix_prior)
            priors_dist[ix] = priors.distributions[jx]
            priors_scale[ix] = priors.priors_on_parameter_scale[jx] ? :parameter_scale : :lin
        end
        bijectors[ix] = Bijectors.bijector(priors_dist[ix])
    end

    inv_bijectors = Bijectors.Stacked(Bijectors.inverse.(bijectors))
    bijectors = Bijectors.Stacked(bijectors)
    tpriors = Bijectors.transformed.(priors_dist)

    return PEtab.InferenceInfo(
        priors_dist, tpriors, bijectors, inv_bijectors, priors_scale, parameters_scale,
        parameter_names
    )
end
