#=
    Recrating PEtab test-suite for Catalyst integration to robustly test that
    we support a wide arrange of features for PEtab integration
=#

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
measurements = DataFrame(obs_id=["obs_a", "obs_a"],
                         time=[0, 10.0],
                         measurement=[0.7, 0.1],
                         noise_parameters=0.5)

# PEtab-parameter to "estimate"
petab_parameters = [PEtabParameter(:a0, value=1.0, scale=:lin),
                    PEtabParameter(:b0, value=0.0, scale=:lin),
                    PEtabParameter(:k1, value=0.8, scale=:lin),
                    PEtabParameter(:k2, value=0.6, scale=:lin)]

# Observable equation
@unpack A = rn
observables = Dict("obs_a" => PEtabObservable(A, 0.5))

# Create a PEtabODEProblem ReactionNetwork
petab_model_rn = PEtabModel(rn, observables, measurements,
                            petab_parameters, verbose=false)
petab_problem_rn = PEtabODEProblem(petab_model_rn, verbose=false)
# Create a PEtabODEProblem ODESystem
petab_model_sys = PEtabModel(sys, observables, measurements,
                             petab_parameters, verbose=false)
petab_problem_sys = PEtabODEProblem(petab_model_sys, verbose=false)

# Compute negative log-likelihood
nll_rn = petab_problem_rn.compute_cost(petab_problem_rn.θ_nominalT)
nll_sys = petab_problem_sys.compute_cost(petab_problem_sys.θ_nominalT)
@test nll_rn ≈ 0.84750169713188 atol=1e-3
@test nll_sys ≈ 0.84750169713188 atol=1e-3
