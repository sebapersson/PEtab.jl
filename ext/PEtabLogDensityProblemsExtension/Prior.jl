function PEtab.compute_prior(
        x_inference, inference_info::PEtab.InferenceInfo
    )
    logpdf_prior = 0.0
    x_prior = inference_info.inv_bijectors(x_inference)
    for (i, prior) in pairs(inference_info.priors)
        logpdf_prior += logpdf(prior, x_prior[i])
    end
    return logpdf_prior
end
