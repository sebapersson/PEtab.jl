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
    prior_k1 = Normal(-0.8, 0.2)
    prior_k2 = LogNormal(0.6, 1.0)

    logprior = logpdf(prior_sigma, sigma)
    logprior += logpdf(prior_k1, _k1)
    logprior += logpdf(prior_k2, k2)
    return logprior * -1
end

# Define reaction network model
rn = @reaction_network begin
    @parameters a0 b0
    @species A(t)=a0 B(t)=b0
    (k1, k2), A <--> B
end

@parameters a0 b0 k1 k2
@variables t A(t) B(t)
D = Differential(t)
eqs = [
    D(A) ~ -k1*A + k2*B
    D(B) ~ k1*A - k2*B
]
@named sys = ODESystem(eqs; defaults=Dict(A => a0, B => b0))

# Measurement data
measurements = DataFrame(simulation_id=["c0", "c0"],
                         obs_id=["obs_a", "obs_a"],
                         time=[0, 10.0],
                         measurement=[0.7, 0.1],
                         noise_parameters=0.5)

# Single experimental condition
simulation_conditions = Dict("c0" => Dict())

# Observable equation
@unpack A = rn
@parameters sigma
observables = Dict("obs_a" => PEtabObservable(A, sigma))

# PEtab-parameter to "estimate"
petab_parameters = [PEtabParameter(:sigma, value=1.0, scale=:lin, prior=LogNormal(0.6, 1.0)),
                    PEtabParameter(:a0, value=1.0, scale=:lin),
                    PEtabParameter(:b0, value=0.0, scale=:lin),
                    PEtabParameter(:k1, value=0.8, scale=:log10, prior=Normal(-0.8, 0.2), prior_on_linear_scale=false),
                    PEtabParameter(:k2, value=0.6, scale=:log10, prior=LogNormal(0.6, 1.0))]

# Create a PEtabODEProblem ReactionNetwork
petab_model_rn = PEtabModel(rn, simulation_conditions, observables, measurements,
                            petab_parameters, verbose=false)
petab_problem_rn = PEtabODEProblem(petab_model_rn, verbose=false)

# Compute gradient + hessian for nllh and prior
prior_val = _compute_prior(petab_problem_sys.θ_nominalT)
prior_grad = ForwardDiff.gradient(_compute_prior, petab_problem_sys.θ_nominalT)
prior_hess = ForwardDiff.hessian(_compute_prior, petab_problem_sys.θ_nominalT)
nllh = petab_problem_rn.compute_nllh(petab_problem_sys.θ_nominalT)
nllh_grad = ForwardDiff.gradient(petab_problem_rn.compute_nllh, petab_problem_sys.θ_nominalT)
nllh_hess = ForwardDiff.hessian(petab_problem_rn.compute_nllh, petab_problem_sys.θ_nominalT)
# Compute petab objective gradient and hessian
obj = petab_problem_rn.compute_cost(petab_problem_sys.θ_nominalT)
grad = petab_problem_rn.compute_gradient(petab_problem_sys.θ_nominalT)
hess = petab_problem_rn.compute_hessian(petab_problem_sys.θ_nominalT)
# Test it all adds up
@test obj ≈ prior_val + nllh
@test norm(grad - (nllh_grad + prior_grad)) < 1e-8
@test norm(hess - (nllh_hess + prior_hess)) < 1e-8

# The same test but for the Blockhessian approximation, and another gradient method
petab_problem_rn = PEtabODEProblem(petab_model_rn, verbose=false, hessian_method=:BlockForwardDiff, gradient_method=:ForwardEquations)
nllh = petab_problem_rn.compute_nllh(petab_problem_sys.θ_nominalT)
nllh_grad = ForwardDiff.gradient(petab_problem_rn.compute_nllh, petab_problem_sys.θ_nominalT)
nllh_hess = ForwardDiff.hessian(petab_problem_rn.compute_nllh, petab_problem_sys.θ_nominalT)
obj = petab_problem_rn.compute_cost(petab_problem_sys.θ_nominalT)
grad = petab_problem_rn.compute_gradient(petab_problem_sys.θ_nominalT)
hess = petab_problem_rn.compute_hessian(petab_problem_sys.θ_nominalT)
@test obj ≈ prior_val + nllh
@test norm(grad - (nllh_grad + prior_grad)) < 1e-8
@test norm(hess - (nllh_hess + prior_hess)) < 1e-8
