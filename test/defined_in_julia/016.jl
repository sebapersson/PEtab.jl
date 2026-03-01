#=
    Test 0016 from the PEtab test-suite recreated in Julia
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

measurements = DataFrame(
    simulation_id = ["c0", "c0"],
    obs_id = ["obs_a", "obs_b"],
    time = [10.0, 10.0],
    measurement = [0.2, 0.8]
)

simulation_conditions = PEtabCondition("c0")

parameters = [
    PEtabParameter(:k1, value = 0.8, scale = :lin),
    PEtabParameter(:k2, value = 0.6, scale = :lin),
    PEtabParameter(:a0, value = 1.0, scale = :lin),
    PEtabParameter(:b0, value = 0.0, scale = :lin),
]

@unpack A, B = rn
observables = [
    PEtabObservable(:obs_a, A, 0.5),
    PEtabObservable(:obs_b, B, 0.7; distribution = LogNormal),
]

model_rn = PEtabModel(
    rn, observables, measurements, parameters;
    simulation_conditions = simulation_conditions
)
petab_prob_rn = PEtabODEProblem(model_rn)
model_sys = PEtabModel(
    sys, observables, measurements, parameters,
    simulation_conditions = simulation_conditions
)
petab_prob_sys = PEtabODEProblem(model_sys)

nll_rn = petab_prob_rn.nllh(get_x(petab_prob_rn))
nll_sys = petab_prob_sys.nllh(get_x(petab_prob_sys))
@test nll_rn ≈ 0.78492623889606 atol = 1.0e-3
@test nll_sys ≈ 0.78492623889606 atol = 1.0e-3
