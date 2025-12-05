#=
    Test 0020 from the PEtab test-suite recreated in Julia
=#

rn = @reaction_network begin
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
@mtkmodel SYS begin
    @parameters begin
        k1
        k2
    end
    @variables begin
        A(t) = 1.0
        B(t) = 3.0
    end
    @equations begin
        D(A) ~ -k1*A + k2*B
        D(B) ~ k1*A - k2*B
    end
end
@mtkbuild sys = SYS()
speciemap = [:A => 1.0, :B => 3.0] # Constant initial value for B

# Measurement data
measurements = DataFrame(simulation_id=["c0", "c0"],
                         obs_id=["obs_a", "obs_a"],
                         time=[0, 10.0],
                         measurement=[0.7, 0.1])

# Single experimental condition
simulation_conditions = Dict("c0" => Dict(:A => "initial_A", :B => NaN))

# PEtab-parameter to "estimate"
parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
              PEtabParameter(:k2, value=0.6, scale=:lin),
              PEtabParameter(:initial_A, value=2.0, scale=:log10)]

# Observable equation
@unpack A = rn
observables = Dict("obs_a" => PEtabObservable(A, 0.5))

# Create a PEtabODEProblem ReactionNetwork
model_rn = PEtabModel(sys, observables, measurements, parameters; speciemap = speciemap,
                      simulation_conditions = simulation_conditions)
petab_problem_rn = PEtabODEProblem(model_rn, verbose=false)
# Create a PEtabODEProblem ODESystem
model_sys = PEtabModel(sys, observables, measurements, parameters;
                       simulation_conditions = simulation_conditions)
petab_problem_sys = PEtabODEProblem(model_sys, verbose=false)

# Compute negative log-likelihood
nll_rn = petab_problem_rn.nllh(petab_problem_rn.xnominal_transformed)
nll_sys = petab_problem_sys.nllh(petab_problem_sys.xnominal_transformed)
@test nll_rn ≈ 12.17811234685187 atol=1e-3
@test nll_sys ≈ 12.17811234685187 atol=1e-3
