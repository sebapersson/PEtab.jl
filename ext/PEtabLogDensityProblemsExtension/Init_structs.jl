function PEtab.PEtabLogDensity(petab_problem::PEtabODEProblem)::PEtab.PEtabLogDensity
    inference_info = PEtab.InferenceInfo(petab_problem)
    @unpack compute_nllh, compute_nllh_and_gradient, n_parameters_esimtate = petab_problem

    # For via autodiff compute the gradient of the prior and Jacobian correction
    _prior_correction = (x_inference) -> let inference_info = inference_info
        prior = PEtab.compute_prior(x_inference, inference_info)
        correction = Bijectors.logabsdetjac(inference_info.inv_bijectors, x_inference)
        return prior + correction
    end

    logtarget = (x_inference) -> let compute_nllh = compute_nllh,
        inference_info = inference_info

        _logtarget(x_inference, compute_nllh, inference_info)
    end

    logtarget_gradient = (x_inference) -> let compute_nllh_and_gradient = compute_nllh_and_gradient,
        inference_info = inference_info, _prior_correction = _prior_correction

        _logtarget_gradient(x_inference, compute_nllh_and_gradient, _prior_correction,
                            inference_info)
    end

    initial_value = Vector{Float64}(undef, n_parameters_esimtate)

    return PEtab.PEtabLogDensity(inference_info, logtarget, logtarget_gradient,
                                 initial_value, n_parameters_esimtate)
end

function PEtab.InferenceInfo(petab_problem::PEtabODEProblem)::PEtab.InferenceInfo
    @unpack prior_info, parameter_info, θ_names, lower_bounds, upper_bounds = petab_problem

    priors = Vector{Distribution{Univariate, Continuous}}(undef, length(θ_names))
    bijectors = Vector(undef, length(θ_names))
    priors_scale, parameters_scale = similar(θ_names), similar(θ_names)

    for (i, θ) in pairs(θ_names)
        iθ = findfirst(x -> x == θ, parameter_info.parameter_id)
        parameters_scale[i] = parameter_info.parameter_scale[iθ]

        # In case the parameter lacks a defined prior we default to a Uniform
        # on parameter scale with lb and ub as bounds
        if !haskey(prior_info.distribution, θ)
            priors[i] = Uniform(lower_bounds[i], upper_bounds[i])
            priors_scale[i] = :parameter_scale
        else
            priors[i] = prior_info.distribution[θ]
            priors_scale[i] = prior_info.prior_on_parameter_scale[θ] ? :parameter_scale :
                              :lin
        end
        bijectors[i] = Bijectors.bijector(priors[i])
    end

    inv_bijectors = Bijectors.Stacked(Bijectors.inverse.(bijectors))
    bijectors = Bijectors.Stacked(bijectors)
    tpriors = Bijectors.transformed.(priors)

    return PEtab.InferenceInfo(priors, tpriors, bijectors, inv_bijectors, priors_scale,
                               parameters_scale, θ_names)
end
