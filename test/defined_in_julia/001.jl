#=
    Test 0001 from the PEtab test-suite recreated in Julia
=#

rn = @reaction_network begin
    @parameters a0 b0
    @species begin
        A(t) = a0
        B(t) = b0
    end
    @observables obs_a ~ A
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
ps = @parameters k1 k2 a0 b0
states = @variables A(t)=a0 B(t)=b0 obs_a(t)
equations = [
    D(A) ~ -k1 * A + k2 * B
    D(B) ~ k1 * A - k2 * B
    obs_a ~ A
]
@named sys_model = System(equations, t, states, ps)
sys = ModelingToolkitBase.mtkcompile(sys_model)

function f_ode1!(du, u, p, t)
    A, B = u
    @unpack a0, b0, k1, k2 = p
    du[1] = -k1 * A + k2 * B # A
    du[2] = k1 * A - k2 * B # B
    return nothing
end
u0 = (A = :a0, B = :b0)
ps = (a0 = 0.0, b0 = 0.0, k1 = 0.0, k2 = 0.0)
ode_prob1 = ODEProblem(f_ode1!, u0, (0.0, 10.0), ps)

# Measurement data
measurements = DataFrame(
    obs_id = ["obs_a", "obs_a"],
    time = [0, 10.0],
    measurement = [0.7, 0.1],
    noise_parameters = 0.5
)

# PEtab-parameter to "estimate"
parameters = [
    PEtabParameter(:a0, value = 1.0, scale = :lin),
    PEtabParameter(:b0, value = 0.0, scale = :lin),
    PEtabParameter(:k1, value = 0.8, scale = :lin),
    PEtabParameter(:k2, value = 0.6, scale = :lin),
]

# Observable equation
petab_observables = PEtabObservable("obs_a", :obs_a, 0.5)
petab_observables_prob = PEtabObservable("obs_a", :A, 0.5)

# Create a PEtabODEProblem ReactionNetwork
model_rn = PEtabModel(rn, petab_observables, measurements, parameters)
petab_prob_rn = PEtabODEProblem(model_rn)
# Create a PEtabODEProblem ODESystem
model_sys = PEtabModel(sys, petab_observables, measurements, parameters)
petab_prob_sys = PEtabODEProblem(model_sys)
# Create a PEtabODEProblem ODEProblem
model_ode = PEtabModel(ode_prob1, petab_observables_prob, measurements, parameters)
petab_prob_ode = PEtabODEProblem(model_ode)

# Compute negative log-likelihood
nll_rn = petab_prob_rn.nllh(get_x(petab_prob_rn))
nll_sys = petab_prob_sys.nllh(get_x(petab_prob_sys))
nll_prob = petab_prob_ode.nllh(get_x(petab_prob_ode))
@test nll_rn ≈ 0.84750169713188 atol = 1.0e-3
@test nll_sys ≈ 0.84750169713188 atol = 1.0e-3
@test nll_prob ≈ 0.84750169713188 atol = 1.0e-3
