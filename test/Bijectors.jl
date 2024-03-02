using Test
using ModelingToolkit
using PEtab
using ForwardDiff
using LogDensityProblems, LogDensityProblemsAD, Bijectors

function test_custom_llh_and_gradient(x_nllh, x_inference, inference_info)

    # Setup a test dataset (mixture-normal)
    dist1_data = Normal(-2.0, 1.0)
    dist2_data = Normal(3.0, 1.0)
    data = vcat(rand(dist1_data, 10), rand(dist2_data, 10))

    # For Likelihood
    compute_nllh_ref = (x) -> _compute_nllh(x, inference_info, data)
    compute_llh_check = (x) -> PEtab.compute_llh(x, compute_nllh_ref, inference_info)
    llh_ref = compute_nllh_ref(x_nllh) * -1
    llh_check = compute_llh_check(x_inference)
    @test llh_ref ≈ llh_check atol=1e-10

    # For gradient
    grad_ref = ForwardDiff.gradient(compute_llh_check, x_inference)
    grad_check = ForwardDiff.gradient(compute_nllh_ref, x_nllh)
    PEtab.correct_gradient!(grad_check, x_inference, x_nllh, inference_info)
    @test all(grad_ref .- grad_check .< 1e-10)

    return nothing
end


function _compute_nllh(θ::Vector{T}, inference_info::PEtab.InferenceInfo, data::Vector{Float64})::T where T<:Real

    x = similar(θ)
    @unpack priors_scale, parameters_scale = inference_info
    for i in eachindex(x)
        if parameters_scale[i] === :lin
            x[i] = θ[i]
        elseif parameters_scale[i] === :log10
            x[i] = exp10(θ[i])
        elseif parameters_scale[i] === :log
            x[i] = exp(θ[i])
        end
    end

    μ1, μ2 = x
    _logpdf = sum(logpdf(Normal(μ1, 1.0), data[1:10]))
    _logpdf += sum(logpdf(Normal(μ2, 1.0), data[11:20]))
    return _logpdf * -1
end

@testset "Likelihood and gradient transformation" begin

    # log10 and lin scale, bounded priors
    prior1, scale1, prior_scale1 = Uniform(-10, 0), :lin, :lin
    prior2, scale2, prior_scale2 = Gamma(1.0, 1.0), :log10, :lin
    _inference_info = PEtab.InferenceInfo([prior1, prior2], Bijectors.transformed.([prior1, prior2]),
                            Bijectors.Stacked(Bijectors.bijector.([prior1, prior2])),
                            Bijectors.Stacked(Bijectors.inverse.(Bijectors.bijector.([prior1, prior2]))),
                            [prior_scale1, prior_scale2], [scale1, scale2],
                            [:μ1, :μ2])
    x_nllh = [-3.0, log10(2.0)]
    x_inference = _inference_info.bijectors([-3.0, 2.0])
    test_custom_llh_and_gradient(x_nllh, x_inference, _inference_info)

    # log10 and lin scale, unbounded priors
    prior1, scale1, prior_scale1 = Normal(-2.0, 1.0), :lin, :lin
    prior2, scale2, prior_scale2 = Normal(1.0, 1.0), :log10, :lin
    _inference_info = PEtab.InferenceInfo([prior1, prior2], Bijectors.transformed.([prior1, prior2]),
                            Bijectors.Stacked(Bijectors.bijector.([prior1, prior2])),
                            Bijectors.Stacked(Bijectors.inverse.(Bijectors.bijector.([prior1, prior2]))),
                            [prior_scale1, prior_scale2], [scale1, scale2],
                            [:μ1, :μ2])
    x_nllh = [-3.0, log10(2.0)]
    x_inference = _inference_info.bijectors([-3.0, 2.0])
    test_custom_llh_and_gradient(x_nllh, x_inference, _inference_info)

    # log and lin scale, bounded priors
    prior1, scale1, prior_scale1 = Uniform(-10., 0.0), :lin, :lin
    prior2, scale2, prior_scale2 = Gamma(1.0, 1.0), :log, :lin
    _inference_info = PEtab.InferenceInfo([prior1, prior2], Bijectors.transformed.([prior1, prior2]),
                            Bijectors.Stacked(Bijectors.bijector.([prior1, prior2])),
                            Bijectors.Stacked(Bijectors.inverse.(Bijectors.bijector.([prior1, prior2]))),
                            [prior_scale1, prior_scale2], [scale1, scale2],
                            [:μ1, :μ2])
    x_nllh = [-3.0, log(2.0)]
    x_inference = _inference_info.bijectors([-3.0, 2.0])
    test_custom_llh_and_gradient(x_nllh, x_inference, _inference_info)

    # Prior on parameter scale, bounded priors
    prior1, scale1, prior_scale1 = Uniform(-10., 0.0), :lin, :parameter_scale
    prior2, scale2, prior_scale2 = Gamma(1.0, 1.0), :log, :parameter_scale
    _inference_info = PEtab.InferenceInfo([prior1, prior2], Bijectors.transformed.([prior1, prior2]),
                            Bijectors.Stacked(Bijectors.bijector.([prior1, prior2])),
                            Bijectors.Stacked(Bijectors.inverse.(Bijectors.bijector.([prior1, prior2]))),
                            [prior_scale1, prior_scale2], [scale1, scale2],
                            [:μ1, :μ2])
    x_nllh = [-3.0, log(2.0)]
    x_inference = _inference_info.bijectors([-3.0, log(2.0)])
    test_custom_llh_and_gradient(x_nllh, x_inference, _inference_info)

    # Prior on parameter scale, unbounded priors
    prior1, scale1, prior_scale1 = Normal(-1., 1.0), :lin, :parameter_scale
    prior2, scale2, prior_scale2 = Normal(1.0, 1.0), :log, :parameter_scale
    _inference_info = PEtab.InferenceInfo([prior1, prior2], Bijectors.transformed.([prior1, prior2]),
                            Bijectors.Stacked(Bijectors.bijector.([prior1, prior2])),
                            Bijectors.Stacked(Bijectors.inverse.(Bijectors.bijector.([prior1, prior2]))),
                            [prior_scale1, prior_scale2], [scale1, scale2],
                            [:μ1, :μ2])
    x_nllh = [-3.0, log(2.0)]
    x_inference = _inference_info.bijectors([-3.0, log(2.0)])
    test_custom_llh_and_gradient(x_nllh, x_inference, _inference_info)
end
