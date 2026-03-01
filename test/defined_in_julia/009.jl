#=
    Test 0009 from the PEtab test-suite recreated in Julia
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
    pre_eq_id = ["preeq_c0", "preeq_c0"],
    simulation_id = ["c0", "c0"],
    obs_id = ["obs_a", "obs_a"],
    time = [1.0, 10.0],
    measurement = [0.7, 0.1]
)

simulation_conditions = [
    PEtabCondition(:c0, :k1 => 0.8)
    PEtabCondition(:preeq_c0, :k1 => 0.3)
]

parameters = [
    PEtabParameter(:a0, value = 1.0, scale = :lin),
    PEtabParameter(:b0, value = 0.0, scale = :lin),
    PEtabParameter(:k2, value = 0.6, scale = :lin),
]

@unpack A = rn
observables = PEtabObservable("obs_a", A, 0.5)

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
@test nll_rn ≈ 0.75799668259765 atol = 1.0e-3
@test nll_sys ≈ 0.75799668259765 atol = 1.0e-3
