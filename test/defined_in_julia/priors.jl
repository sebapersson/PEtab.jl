#=
    Recrating PEtab test-suite for Catalyst integration to robustly test that
    we support a wide arrange of features for PEtab integration, and test that
    Hessian of the prior is computed correctly for block and non block approach
=#

# Compute prior for testing
function _compute_prior(θ)
    a0, b0, _k1, _k2, sigma = θ
    k1, k2 = exp10(_k1), exp10(_k2)

    prior_sigma = LogNormal(0.6, 1.0)
    prior_k1 = truncated(Normal(-0.8, 0.2), exp10(-0.8), exp10(0.2))
    prior_k2 = LogNormal(0.6, 1.0)

    logprior = logpdf(prior_sigma, sigma)
    logprior += logpdf(prior_k1, k1)
    logprior += logpdf(prior_k2, k2)
    return logprior * -1
end

# Define reaction network model
rn = @reaction_network begin
    @parameters a0 b0
    @species A(t) = a0 B(t) = b0
    (k1, k2), A <--> B
end

# Measurement data
measurements = DataFrame(
    simulation_id = ["c0", "c0"],
    obs_id = ["obs_a", "obs_a"],
    time = [0, 10.0],
    measurement = [0.7, 0.1],
    noise_parameters = 0.5
)

# Single experimental condition
simulation_conditions = PEtabCondition(:c0)

# Observable equation
@unpack A = rn
@parameters sigma
observables = PEtabObservable("obs_a", A, sigma)

# PEtab-parameter to "estimate"
parameters = [
    PEtabParameter(:sigma, value = 1.0, scale = :lin, prior = LogNormal(0.6, 1.0)),
    PEtabParameter(:a0, value = 1.0, scale = :lin),
    PEtabParameter(:b0, value = 0.0, scale = :lin),
    PEtabParameter(:k1, value = 0.8, scale = :log10, lb = exp10(-0.8), ub = exp10(0.2), prior = Normal(-0.8, 0.2)),
    PEtabParameter(:k2, value = 0.6, scale = :log10, prior = LogNormal(0.6, 1.0)),
]

# Create a PEtabODEProblem ReactionNetwork
model_rn = PEtabModel(
    rn, observables, measurements, parameters;
    simulation_conditions = simulation_conditions
)
petab_prob_rn = PEtabODEProblem(model_rn)

# Compute gradient + hessian for nllh and prior
x = get_x(petab_prob_rn)
prior_val = _compute_prior(x)
prior_grad = ForwardDiff.gradient(_compute_prior, x)
prior_hess = ForwardDiff.hessian(_compute_prior, x)

# Compute nllh as well as nllh + prior
nllh = petab_prob_rn.nllh(x; prior = false)
nllh_grad = petab_prob_rn.grad(x; prior = false)
nllh_hess = petab_prob_rn.hess(x; prior = false)
obj = petab_prob_rn.nllh(x)
grad = petab_prob_rn.grad(x)
hess = petab_prob_rn.hess(x)

# Test it all adds up
@test obj ≈ prior_val + nllh
@test all(.≈(grad, nllh_grad + prior_grad; atol = 1.0e-3))
@test all(.≈(hess, nllh_hess + prior_hess; atol = 1.0e-3))

# The same test but for the Blockhessian approximation, and another gradient method
petab_prob_rn = PEtabODEProblem(
    model_rn, verbose = false, hessian_method = :BlockForwardDiff,
    gradient_method = :ForwardEquations
)
nllh = petab_prob_rn.nllh(x; prior = false)
nllh_grad = petab_prob_rn.grad(x; prior = false)
nllh_hess = petab_prob_rn.hess(x; prior = false)
obj = petab_prob_rn.nllh(x)
grad = petab_prob_rn.grad(x)
hess = petab_prob_rn.hess(x)
@test obj ≈ prior_val + nllh
@test all(.≈(grad, nllh_grad + prior_grad; atol = 1.0e-3))
@test all(.≈(hess[1:4, 1:4], nllh_hess[1:4, 1:4] + prior_hess[1:4, 1:4]; atol = 1.0e-3))
@test all(.≈(hess[5, 5], nllh_hess[5, 5] + prior_hess[5, 5]; atol = 1.0e-3))

# On topic of distributions, check PEtabObservable throws correctly
@test_throws PEtab.PEtabFormatError begin
    PEtabObservable(:obs1, "A + 3.0", 3.0; distribution = Distributions.Gamma)
end
