LogDensityProblems.logdensity(p::PEtabLogDensity, θ) = p.logtarget(θ)

LogDensityProblems.dimension(p::PEtabLogDensity) = p.dim

LogDensityProblems.capabilities(::PEtabLogDensity) = LogDensityProblems.LogDensityOrder{1}()

LogDensityProblems.logdensity_and_gradient(p::PEtabLogDensity, x) = p.logtarget_gradient(x)

function _logtarget(x_inference::AbstractVector{T}, compute_nllh::Function,
                    inference_info::PEtab.InferenceInfo)::T where {T <: Real}
    # Logposterior with Jacobian correction for transformed parameters
    logtarget = compute_llh(x_inference, compute_nllh, inference_info)
    logtarget += compute_prior(x_inference, inference_info)
    logtarget += Bijectors.logabsdetjac(inference_info.inv_bijectors, x_inference)
    return logtarget
end

function _logtarget_gradient(x_inference::AbstractVector{T}, _nllh_gradient::Function,
                             _prior_correction::Function,
                             inference_info::PEtab.InferenceInfo)::Tuple{T,
                                                                         Vector{T}} where {T <:
                                                                                           Real}
    x_nllh = to_nllh_scale(x_inference, inference_info)
    nllh, logtarget_grad = _nllh_gradient(x_nllh)

    # Logposterior with Jacobian correction for transformed parameters
    logtarget = nllh * -1 + compute_prior(x_inference, inference_info)
    logtarget += Bijectors.logabsdetjac(inference_info.inv_bijectors, x_inference)

    # Gradient with transformation correction
    correct_gradient!(logtarget_grad, x_inference, x_nllh, inference_info)
    logtarget_grad .+= ForwardDiff.gradient(_prior_correction, x_inference)

    return logtarget, logtarget_grad
end
