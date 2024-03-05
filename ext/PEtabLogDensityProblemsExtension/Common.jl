function to_nllh_scale(x_inference::AbstractVector,
                       inference_info::PEtab.InferenceInfo)::AbstractVector

    # Transform x into θ - the scale for the priors
    @unpack inv_bijectors, priors_scale, parameters_scale = inference_info
    x_nllh = inference_info.inv_bijectors(x_inference)

    for i in eachindex(x_nllh)

        # If the prior is on the same scale as the parameter in the
        # likelihood function no further transformations are needed
        if priors_scale[i] === :parameter_scale
            continue
        end
        if priors_scale[i] === :lin && parameters_scale[i] === :lin
            continue
        end

        # If the prior is on a different scale than the parameter in the
        # likelihood x_nllh must be transformed to likelihood scale
        if parameters_scale[i] === :log10
            x_nllh[i] = log10(x_nllh[i])
            continue
        elseif parameters_scale[i] === :log
            x_nllh[i] = log(x_nllh[i])
            continue
        end

        # TODO: Obviously
        @error "We should not be here at all - ajaj"
    end

    return x_nllh
end

function PEtab.to_prior_scale(x_nllh::AbstractVector,
                              target::PEtab.PEtabLogDensity)::AbstractVector
    inference_info = target.inference_info
    @unpack parameters_scale, priors_scale = inference_info
    x_prior = similar(x_nllh)
    for (i, x) in pairs(x_nllh)
        if priors_scale[i] == :parameter_scale
            x_prior[i] = x
            continue
        end
        if priors_scale[i] === :lin && parameters_scale[i] === :lin
            x_prior[i] = x
            continue
        end

        if parameters_scale[i] === :log10
            x_prior[i] = exp10(x)
        elseif parameters_scale[i] === :log
            x_prior[i] = exp(x)
        else
            @error "Ajajajaj"
        end
    end
    return x_prior
end
