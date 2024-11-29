#=
    Test 0001 from the PEtab test-suite recreated in Julia
=#

rn = @reaction_network begin
    @species A(t)=1.0 B(t)=0.0
    (k1, k2), A <--> B
end

# Measurement data
measurements = DataFrame(obs_id=["obs_a", "obs_a"],
                         time=[0, 10.0],
                         measurement=[0.7, 0.1],
                         noise_parameters=0.5)

# PEtab-parameter to "estimate"
parameters = [PEtabParameter(:k1, value=0.8, scale=:log2),
              PEtabParameter(:k2, value=0.6, scale=:log2)]

# Observable equation
@unpack A = rn
observables = Dict("obs_a" => PEtabObservable(A, 0.5, transformation=:log2))

# Create a PEtabODEProblem ReactionNetwork
model_rn = PEtabModel(rn, observables, measurements, parameters)
petab_problem_rn = PEtabODEProblem(model_rn; gradient_method = :ForwardEquations)
x = get_x(petab_problem_rn)

# Reference nllh computed by "hand"
m = measurements.measurement
res2 = ((log2.(petab_problem_rn.simulated_values(x)) - log2.(m)) ./ 0.5).^2
nllh_ref = sum(0.5*log(2π) + log(log(2)) + log(0.5) .+ log.(m) .+ 0.5 .* res2)

# Compute negative log-likelihood
nllh_rn = petab_problem_rn.nllh(x)
@test nllh_ref ≈ nllh_rn atol=1e-3

# Check that gradients are transformed correctly
gref = ForwardDiff.gradient(petab_problem_rn.nllh, x) |> collect
gtest = petab_problem_rn.grad(x) |> collect
@test all(.≈(gref, gtest; atol=1e-3))
