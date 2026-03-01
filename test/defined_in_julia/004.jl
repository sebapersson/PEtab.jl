#=
    Test 0004 from the PEtab test-suite recreated in Julia
=#

rn = @reaction_network begin
    @parameters a0 b0
    @species A(t) = a0 B(t) = b0
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
@parameters k1 k2
@variables A(t) = a0 B(t) = b0
equations = [
    D(A) ~ -k1 * A + k2 * B
    D(B) ~ k1 * A - k2 * B
]
@named sys_model = System(equations, t)
sys = ModelingToolkitBase.mtkcompile(sys_model)

# Measurement data
measurements = DataFrame(
    simulation_id = ["c0", "c0"],
    obs_id = ["obs_a", "obs_a"],
    time = [0, 10.0],
    measurement = [0.7, 0.1],
    observable_parameters = ["0.5;2", "0.5;2"]
)

# Single experimental condition
simulation_conditions = [PEtabCondition(:c0)]

# PEtab-parameter to "estimate"
parameters = [
    PEtabParameter(:a0, value = 1.0, scale = :lin),
    PEtabParameter(:b0, value = 0.0, scale = :lin),
    PEtabParameter(:k1, value = 0.8, scale = :lin),
    PEtabParameter(:k2, value = 0.6, scale = :lin),
    PEtabParameter(:scaling_A, value = 0.5, scale = :lin),
    PEtabParameter(:offset_A, value = 2.0, scale = :lin),
]

# Observable equation
@unpack A = rn
@parameters scaling_A offset_A
observables = PEtabObservable("obs_a", scaling_A * A + offset_A, 1.0)

# Create a PEtabODEProblem ReactionNetwork
model_rn = PEtabModel(
    rn, observables, measurements, parameters;
    simulation_conditions = simulation_conditions
)
petab_prob_rn = PEtabODEProblem(model_rn)
# Create a PEtabODEProblem ODESystem
model_sys = PEtabModel(
    sys, observables, measurements, parameters,
    simulation_conditions = simulation_conditions
)
petab_prob_sys = PEtabODEProblem(model_sys)

# Compute negative log-likelihood
nll_rn = petab_prob_rn.nllh(get_x(petab_prob_rn))
nll_sys = petab_prob_sys.nllh(get_x(petab_prob_sys))
@test nll_rn ≈ 5.69297960953693 atol = 1.0e-3
@test nll_sys ≈ 5.69297960953693 atol = 1.0e-3
