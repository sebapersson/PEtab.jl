#=
    Math expression when defining conditions
=#

rn = @reaction_network begin
    @species A(t)=1.0 B(t)=1.0
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
@mtkmodel SYS_v2_29 begin
    @parameters begin
        k1
        k2
    end
    @variables begin
        A(t) = 1.0
        B(t) = 1.0
    end
    @equations begin
        D(A) ~ -k1*A + k2*B
        D(B) ~ k1*A - k2*B
    end
end
@mtkbuild sys = SYS_v2_29()

# Measurement data
measurements = DataFrame(simulation_id=["e1", "e1"],
                         obs_id=["obs_a", "obs_a"],
                         time=[5.0, 10.0],
                         measurement=[0.01, 0.1])

# Single experimental condition
simulation_conditions = PEtabCondition(:e1, "", ""; t0 = 5.0)

# PEtab-parameter to "estimate"
parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
              PEtabParameter(:k2, value=0.6, scale=:lin)]

# Observable equation
observables = Dict("obs_a" => PEtabObservable("A", 1.0))

# Create a PEtabODEProblem ReactionNetwork
model_rn = PEtabModel(rn, observables, measurements, parameters;
                      simulation_conditions = simulation_conditions)
petab_problem_rn = PEtabODEProblem(model_rn)
# Create a PEtabODEProblem ODESystem
model_sys = PEtabModel(sys, observables, measurements, parameters;
                       simulation_conditions = simulation_conditions)
petab_problem_sys = PEtabODEProblem(model_sys)

nll_rn = petab_problem_rn.nllh(get_x(petab_problem_rn))
nll_sys = petab_problem_sys.nllh(get_x(petab_problem_sys))
@test nll_rn ≈ 2.61465836008652 atol=1e-3
@test nll_sys ≈ 2.61465836008652 atol=1e-3
