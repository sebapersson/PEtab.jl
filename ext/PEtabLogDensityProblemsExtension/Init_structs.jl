function PEtab.PEtabLogDensity(petab_problem::PEtabODEProblem)::PEtab.PEtabLogDensity
    inference_info = PEtab.InferenceInfo(petab_problem)
    @unpack nllh, nllh_grad, nparameters_esimtate = petab_problem

    # For via autodiff compute the gradient of the prior and Jacobian correction
    _prior_correction = (x_inference) -> let inference_info = inference_info
        prior = PEtab.compute_prior(x_inference, inference_info)
        correction = Bijectors.logabsdetjac(inference_info.inv_bijectors, x_inference)
        return prior + correction
    end

    logtarget = (x_inference) -> let nllh = nllh,
        inference_info = inference_info
        _logtarget(x_inference, nllh, inference_info)
    end

    logtarget_gradient = (x_inference) -> let nllh_grad = nllh_grad,
        inference_info = inference_info, _prior_correction = _prior_correction
        _logtarget_gradient(x_inference, nllh_grad, _prior_correction,
                            inference_info)
    end

    initial_value = Vector{Float64}(undef, nparameters_esimtate)

    return PEtab.PEtabLogDensity(inference_info, logtarget, logtarget_gradient,
                                 initial_value, nparameters_esimtate)
end

function PEtab.InferenceInfo(petab_problem::PEtabODEProblem)::PEtab.InferenceInfo
    @unpack model_info, xnames, lower_bounds, upper_bounds = petab_problem
    @unpack priors, petab_parameters = model_info
    priors_dist = Vector{Distribution{Univariate, Continuous}}(undef, length(xnames))
    bijectors = Vector(undef, length(xnames))
    priors_scale, parameters_scale = similar(xnames), similar(xnames)

    for (i, θ) in pairs(xnames)
        iθ = findfirst(x -> x == θ, petab_parameters.parameter_id)
        parameters_scale[i] = petab_parameters.parameter_scale[iθ]

        # In case the parameter lacks a defined prior we default to a Uniform
        # on parameter scale with lb and ub as bounds
        if !haskey(priors.distribution, θ)
            priors_dist[i] = Uniform(lower_bounds[i], upper_bounds[i])
            priors_scale[i] = :parameter_scale
        else
            priors_dist[i] = priors.distribution[θ]
            priors_scale[i] = priors.prior_on_parameter_scale[θ] ? :parameter_scale :
                              :lin
        end
        bijectors[i] = Bijectors.bijector(priors_dist[i])
    end

    inv_bijectors = Bijectors.Stacked(Bijectors.inverse.(bijectors))
    bijectors = Bijectors.Stacked(bijectors)
    tpriors = Bijectors.transformed.(priors_dist)

    return PEtab.InferenceInfo(priors_dist, tpriors, bijectors, inv_bijectors, priors_scale,
                               parameters_scale, xnames)
end
