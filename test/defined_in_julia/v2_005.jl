#=
    Value for specie only assigned for a subset of conditions.
=#

rn = @reaction_network begin
    @parameters begin
        offset_A = 2.0
    end
    @species begin
        A(t) = 1.0
        B(t) = 0.0
    end
    @observables begin
        obs_a ~ A + offset_A
    end
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
@mtkmodel SYS_v2_5 begin
    @parameters begin
        k1
        k2
        offset_A = 2.0
    end
    @variables begin
        A(t) = 1.0
        B(t) = 0.0
        # Observable
        obs_a(t)
    end
    @equations begin
        D(A) ~ -k1*A + k2*B
        D(B) ~ k1*A - k2*B
        obs_a ~ A + offset_A
    end
end
@mtkbuild sys = SYS_v2_5()

# Measurement data
measurements = DataFrame(simulation_id=["e1", "e2"],
                         obs_id=["obs_a", "obs_a"],
                         time=[10.0, 10.0],
                         measurement=[2.1, 3.2])

# Single experimental condition
simulation_conditions = Dict("e1" => Dict(),
                             "e2" => Dict(:offset_A => "offset_A_c1"))

# PEtab-parameter to "estimate"
parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
              PEtabParameter(:k2, value=0.6, scale=:lin),
              PEtabParameter(:offset_A_c1, value=3.0, scale=:lin)]

# Observable equation
observables = Dict("obs_a" => PEtabObservable(:obs_a, 1.0))

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
@test nll_rn ≈ 1.91797937195749 atol=1e-3
@test nll_sys ≈ 1.91797937195749 atol=1e-3
