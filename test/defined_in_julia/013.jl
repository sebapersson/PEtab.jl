#=
    Test 0013 from the PEtab test-suite recreated in Julia
=#

rn = @reaction_network begin
    @parameters par
    (k1, k2), A <--> B
end
speciemap = [:A => 1.0]

t = default_t()
D = default_time_deriv()
@parameters k1 k2 par
@variables A(t)=1.0 B(t)
equations = [
    D(A) ~ -k1 * A + k2 * B
    D(B) ~ k1 * A - k2 * B
]
@named sys_model = ODESystem(equations, Catalyst.default_t(), [A, B], [k1, k2, par])

measurements = DataFrame(
    simulation_id = ["c0", "c0"],
    obs_id = ["obs_a", "obs_a"],
    time = [0.0, 10.0],
    measurement = [0.7, 0.1]
)

simulation_conditions = PEtabCondition(:c0, :B => :par)

parameters = [
    PEtabParameter(:k1, value = 0.8, scale = :lin),
    PEtabParameter(:k2, value = 0.6, scale = :lin),
    PEtabParameter(:par, value = 7.0, scale = :lin),
]

@unpack A = rn
observables = PEtabObservable(:obs_a, A, 0.5)

model_rn = PEtabModel(
    sys, observables, measurements, parameters; speciemap = speciemap,
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
@test nll_rn ≈ 22.79033132827511 atol = 1.0e-3
@test nll_sys ≈ 22.79033132827511 atol = 1.0e-3
