#=
    Math expression when defining conditions
=#

rn = @reaction_network begin
    @parameters a0 b0
    @species A(t)=a0 B(t)=b0
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
@mtkmodel SYS_v2_26 begin
    @parameters begin
        a0
        b0
        k1
        k2
    end
    @variables begin
        A(t) = a0
        B(t) = b0
    end
    @equations begin
        D(A) ~ -k1*A + k2*B
        D(B) ~ k1*A - k2*B
    end
end
@mtkbuild sys = SYS_v2_26()

# Measurement data
measurements = DataFrame(simulation_id=["e1", "e1", "e1", "e1"],
                         obs_id=["obs_a", "obs_a", "obs_b", "obs_b"],
                         time=[0, 10.0, 0, 10.0],
                         measurement=[0.7, 0.1, 0.7, 0.1])

# Single experimental condition
@parameters initial_B1, initial_B2
simulation_conditions = PEtabCondition(:e1,  [:a0, :b0],
                                       ["initial_A1 + initial_A2", initial_B1 / initial_B2])

# PEtab-parameter to "estimate"
parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
              PEtabParameter(:k2, value=0.6, scale=:lin),
              PEtabParameter(:initial_A1, value=0.5, scale=:lin),
              PEtabParameter(:initial_A2, value=1.5, scale=:lin, estimate = false),
              PEtabParameter(:initial_B1, value=9.0, scale=:lin, estimate = false),
              PEtabParameter(:initial_B2, value=3.0, scale=:lin, estimate = false)]

# Observable equation
observables = Dict("obs_a" => PEtabObservable("A", 0.5),
                   "obs_b" => PEtabObservable("B", 0.5))

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
@test nll_rn ≈ 38.41336983161109 atol=1e-3
@test nll_sys ≈ 38.41336983161109 atol=1e-3
