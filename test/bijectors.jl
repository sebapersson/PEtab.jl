using LogDensityProblems, LogDensityProblemsAD, Bijectors, DataFrames, PEtab,
    FiniteDifferences, ModelingToolkit, Catalyst, ForwardDiff, Test

function test_custom_llh_and_gradient(x_nllh, x_inference, inference_info)
    # Setup a test dataset (mixture-normal)
    dist1_data = Normal(-2.0, 1.0)
    dist2_data = Normal(3.0, 1.0)
    data = vcat(rand(dist1_data, 10), rand(dist2_data, 10))

    # For Likelihood
    compute_nllh_ref = (x; prior = true) -> _compute_nllh(x, inference_info, data)
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

# Test PEtabLogDensity computes everything correctly for non-uniform priors on unbounded
# domain without need for corrections
@testset "PEtabLogDensity" begin
    rs = @reaction_network begin
        (k1, k2), X1 <--> X2
    end
    u0 = [:X1 => 1.0]
    @unpack X1 = rs
    observables = PEtabObservable(:obs_X1, X1, 0.5)
    par_k1 = PEtabParameter(:k1; scale = :lin, prior = Normal(1.0, 1.0), value = 1.1, lb = -Inf, ub = Inf)
    par_k2 = PEtabParameter(:k2; scale = :lin, prior = Normal(0.5, 3.0), value = 0.9, lb = -Inf, ub = Inf)
    params = [par_k1, par_k2]
    measurements = DataFrame(obs_id="obs_X1", time=[1.0, 2.0, 3.0], measurement=[1.1, 1.2, 1.3])
    model = PEtabModel(rs, observables, measurements, params; speciemap=u0)
    prob = PEtabODEProblem(model; verbose = false)
    prob_density = PEtabLogDensity(prob)
    x = get_x(prob)
    xinference = prob_density.inference_info.bijectors(x)
    llh_prior = prob.nllh(x) * -1
    llh_prior_grad = prob.grad(x) .* -1
    log_target, log_target_grad = prob_density.logtarget_gradient(xinference)
    @test llh_prior == prob_density.logtarget(xinference)
    @test llh_prior ≈ log_target atol=1e-6
    @test all(.≈(llh_prior_grad[:], log_target_grad; atol=1e-6))
    g_ref = FiniteDifferences.grad(central_fdm(5, 1), prob_density.logtarget, xinference)[1]
    @test all(.≈(g_ref, log_target_grad; atol=1e-6))
end
