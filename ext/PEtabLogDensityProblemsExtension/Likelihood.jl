function PEtab.compute_llh(
        x_inference::AbstractVector{T}, nllh::Function, inference_info::PEtab.InferenceInfo
    )::T where {T <: Real}
    x = to_nllh_scale(x_inference, inference_info)
    return nllh(x; prior = false) * -1
end

function PEtab.correct_gradient!(
        grad::T, x_inference::T, x_nllh::T,
        inference_info::PEtab.InferenceInfo
    )::Nothing where {
        T <:
        AbstractVector,
    }
    grad .*= -1
    # Gradient needs to be transformed back to the scale which inference
    # is performed on. Two-step,
    # 1 : From parameter to prior-scale
    # 2 : From prior to inference scale
    @unpack inv_bijectors, priors_scale, parameters_scale = inference_info
    for i in eachindex(grad)

        # 1 parameter to prior scale (unless they are on the same scale)
        if !(
                priors_scale[i] === :parameter_scale ||
                    (priors_scale[i] === :lin && parameters_scale[i] === :lin)
            )
            if parameters_scale[i] === :log10
                grad[i] *= exp(Bijectors.logabsdetjac(log10, exp10(x_nllh[i])))
            elseif parameters_scale[i] === :log
                grad[i] *= exp(Bijectors.logabsdetjac(log, exp(x_nllh[i])))
            end
        end

        # 2 from prior to inference scale
        inv_bijector = inv_bijectors.bs[i]
        grad[i] *= exp(Bijectors.logabsdetjac(inv_bijector, x_inference[i]))
    end

    return nothing
end
