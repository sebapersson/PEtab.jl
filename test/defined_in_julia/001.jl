#=
    Test 0001 from the PEtab test-suite recreated in Julia
=#

rn = @reaction_network begin
    @parameters a0 b0
    @species A(t)=a0 B(t)=b0
    @observables obs_a ~ A
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
@mtkmodel SYS1 begin
    @parameters begin
        a0
        b0
        k1
        k2
    end
    @variables begin
        A(t) = a0
        B(t) = b0
        # Observables
        obs_a(t)
    end
    @equations begin
        D(A) ~ -k1*A + k2*B
        D(B) ~ k1*A - k2*B
        obs_a ~ A
    end
end
@mtkbuild sys = SYS1()

# Measurement data
measurements = DataFrame(obs_id=["obs_a", "obs_a"],
                         time=[0, 10.0],
                         measurement=[0.7, 0.1],
                         noise_parameters=0.5)

# PEtab-parameter to "estimate"
parameters = [PEtabParameter(:a0, value=1.0, scale=:lin),
              PEtabParameter(:b0, value=0.0, scale=:lin),
              PEtabParameter(:k1, value=0.8, scale=:lin),
              PEtabParameter(:k2, value=0.6, scale=:lin)]

# Observable equation
petab_observables = Dict("obs_a" => PEtabObservable(:obs_a, 0.5))

# Create a PEtabODEProblem ReactionNetwork
model_rn = PEtabModel(rn, petab_observables, measurements, parameters)
petab_problem_rn = PEtabODEProblem(model_rn)
# Create a PEtabODEProblem ODESystem
model_sys = PEtabModel(sys, petab_observables, measurements, parameters)
petab_problem_sys = PEtabODEProblem(model_sys)

# Compute negative log-likelihood
nll_rn = petab_problem_rn.nllh(petab_problem_rn.xnominal_transformed)
nll_sys = petab_problem_sys.nllh(petab_problem_sys.xnominal_transformed)
@test nll_rn ≈ 0.84750169713188 atol=1e-3
@test nll_sys ≈ 0.84750169713188 atol=1e-3
