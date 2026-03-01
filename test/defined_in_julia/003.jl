#=
    Test 0003 from the PEtab test-suite recreated in Julia
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

function f_ode3!(du, u, p, t)
    A, B = u
    @unpack a0, b0, k1, k2 = p
    du[1] = -k1 * A + k2 * B # A
    du[2] = k1 * A - k2 * B # B
    return nothing
end
u0 = ComponentArray(A = 0.0, B = 0.0)
ps = ComponentArray(a0 = 0.0, b0 = 0.0, k1 = 0.0, k2 = 0.0)
specie_map = [:A => :a0, :B => :b0]
ode_prob3 = ODEProblem(f_ode3!, u0, (0.0, 10.0), ps)

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
]

# Observable equation
@unpack A = rn
@parameters observableParameter1_obs_a observableParameter2_obs_a
observables = PEtabObservable(:obs_a, observableParameter1_obs_a * A + observableParameter2_obs_a, 0.5)

# Create a PEtabODEProblem ReactionNetwork
model_rn = PEtabModel(
    rn, observables, measurements, parameters;
    simulation_conditions = simulation_conditions
)
petab_prob_rn = PEtabODEProblem(model_rn)
# Create a PEtabODEProblem ODESystem
model_sys = PEtabModel(
    sys, observables, measurements, parameters;
    simulation_conditions = simulation_conditions
)
petab_prob_sys = PEtabODEProblem(model_sys)
# Create PEtabODEProblem ODEProblem
model_ode = PEtabModel(
    ode_prob3, observables, measurements, parameters; speciemap = specie_map,
    simulation_conditions = simulation_conditions
)
petab_prob_ode = PEtabODEProblem(model_ode)

# Compute negative log-likelihood
nll_rn = petab_prob_rn.nllh(get_x(petab_prob_rn))
nll_sys = petab_prob_sys.nllh(get_x(petab_prob_sys))
nll_prob = petab_prob_ode.nllh(get_x(petab_prob_ode))
@test nll_rn ≈ 15.87199287779978 atol = 1.0e-3
@test nll_sys ≈ 15.87199287779978 atol = 1.0e-3
@test nll_prob ≈ 15.87199287779978 atol = 1.0e-3
