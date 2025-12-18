#=
    Value for specie only assigned for a subset of conditions.
=#

rn = @reaction_network begin
    @species A(t)=0.8 B(t)=1.0
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
@mtkmodel SYS_v2_2 begin
    @parameters begin
        k1
        k2
    end
    @variables begin
        A(t) = 0.8
        B(t) = 1.0
    end
    @equations begin
        D(A) ~ -k1*A + k2*B
        D(B) ~ k1*A - k2*B
    end
end
@mtkbuild sys = SYS_v2_2()

# Measurement data
measurements = DataFrame(simulation_id=["e1", "e1", "e2", "e2", "e1", "e2"],
                         obs_id=["obs_a", "obs_a", "obs_a", "obs_a", "obs_b", "obs_b"],
                         time=[0, 10.0, 0, 10.0, 0, 0],
                         measurement=[0.01, 0.1, 0.02, 0.2, 0.01, 0.01])

# Single experimental condition
simulation_conditions = [PEtabCondition(:e1, "", ""),
                         PEtabCondition(:e2, :A, 0.9)]

# PEtab-parameter to "estimate"
parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
              PEtabParameter(:k2, value=0.6, scale=:lin)]

# Observable equation
observables = Dict("obs_a" => PEtabObservable("A", 1.0),
                   "obs_b" => PEtabObservable("B", 1.0))

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
@test nll_rn ≈ 7.60706289161541 atol=1e-3
@test nll_sys ≈ 7.60706289161541 atol=1e-3
