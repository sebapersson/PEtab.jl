#=
    Extra test to verify setting constant parameters work
=#

rn = @reaction_network begin
    @parameters a0 b0
    @species A(t) = a0 B(t) = b0
    (k1, k2), A <--> B
end
parametermap = [:k2 => 1.6]

t = default_t()
D = default_time_deriv()
ps = @parameters k1 k2 = 1.6 a0 b0
sps = @variables A(t) = a0 B(t) = b0
equations = [
    D(A) ~ -k1 * A + k2 * B
    D(B) ~ k1 * A - k2 * B
]
@named sys_model = System(equations, t, sps, ps)
sys = ModelingToolkitBase.mtkcompile(sys_model)

measurements = DataFrame(
    simulation_id = ["c0", "c0"],
    obs_id = ["obs_a", "obs_a"],
    time = [0, 10.0],
    measurement = [0.7, 0.1],
    noise_parameters = 0.5
)

simulation_conditions = PEtabCondition(:c0)

parameters = [
    PEtabParameter(:a0, value = 1.0, scale = :lin),
    PEtabParameter(:b0, value = 0.0, scale = :lin),
    PEtabParameter(:k1, value = 0.8, scale = :log10),
]

@unpack A = rn
observables = PEtabObservable("obs_a", A, 0.5)

model_rn = PEtabModel(
    rn, observables, measurements, parameters, parametermap = parametermap,
    simulation_conditions = simulation_conditions
)
petab_prob_rn = PEtabODEProblem(model_rn)
# TODO: Fix failure with parameter-map here!!
model_sys = PEtabModel(
    sys, observables, measurements, parameters;
    simulation_conditions = simulation_conditions
)
petab_prob_sys = PEtabODEProblem(model_sys)

nll_rn = petab_prob_rn.nllh(get_x(petab_prob_rn))
nll_sys = petab_prob_sys.nllh(get_x(petab_prob_sys))
@test nll_rn ≈ 1.2738049275398435 atol = 1.0e-3
@test nll_sys ≈ 1.2738049275398435 atol = 1.0e-3
