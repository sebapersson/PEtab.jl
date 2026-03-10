#=
    Test 0010 from the PEtab test-suite recreated in Julia
=#

rn = @reaction_network begin
    (k1, k2), A <--> B
end
speciemap = [:A => 1.0]

t = default_t()
D = default_time_deriv()
ps = @parameters k1 k2
sps = @variables A(t) = 1.0 B(t)
equations = [
    D(A) ~ -k1 * A + k2 * B
    D(B) ~ k1 * A - k2 * B
]
@named sys_model = System(equations, t, sps, ps)
sys = ModelingToolkitBase.mtkcompile(sys_model)

measurements = DataFrame(
    pre_eq_id = ["preeq_c0", "preeq_c0"],
    simulation_id = ["c0", "c0"],
    obs_id = ["obs_a", "obs_a"],
    time = [1.0, 10.0],
    measurement = [0.7, 0.1]
)

simulation_conditions = [
    PEtabCondition(:c0, :k1 => 0.8, :B => 1.0)
    PEtabCondition(:preeq_c0, "k1" => "0.3", "B" => 0.0)
]

parameters = PEtabParameter(:k2, value = 0.6, scale = :lin)

@unpack A = rn
observables = PEtabObservable("obs_a", A, 0.5)

model_rn = PEtabModel(
    rn, observables, measurements, parameters; speciemap = speciemap,
    simulation_conditions = simulation_conditions,
)
petab_prob_rn = PEtabODEProblem(model_rn)
model_sys = PEtabModel(
    sys, observables, measurements, parameters,
    simulation_conditions = simulation_conditions
)
petab_prob_sys = PEtabODEProblem(model_sys)

nll_rn = petab_prob_rn.nllh(get_x(petab_prob_rn))
nll_sys = petab_prob_sys.nllh(get_x(petab_prob_sys))
@test nll_rn ≈ 1.20628941926143 atol = 1.0e-3
@test nll_sys ≈ 1.20628941926143 atol = 1.0e-3
