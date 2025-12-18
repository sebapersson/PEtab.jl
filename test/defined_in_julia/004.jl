#=
    Test 0004 from the PEtab test-suite recreated in Julia
=#

rn = @reaction_network begin
    @parameters a0 b0
    @species A(t)=a0 B(t)=b0
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
@mtkmodel SYS4 begin
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
@mtkbuild sys = SYS4()

# Measurement data
measurements = DataFrame(simulation_id=["c0", "c0"],
                         obs_id=["obs_a", "obs_a"],
                         time=[0, 10.0],
                         measurement=[0.7, 0.1],
                         observable_parameters=["0.5;2", "0.5;2"])

# Single experimental condition
simulation_conditions = [PEtabCondition(:c0, "", "")]

# PEtab-parameter to "estimate"
parameters = [PEtabParameter(:a0, value=1.0, scale=:lin),
                    PEtabParameter(:b0, value=0.0, scale=:lin),
                    PEtabParameter(:k1, value=0.8, scale=:lin),
                    PEtabParameter(:k2, value=0.6, scale=:lin),
                    PEtabParameter(:scaling_A, value=0.5, scale=:lin),
                    PEtabParameter(:offset_A, value=2.0, scale=:lin)]

# Observable equation
@unpack A = rn
@parameters scaling_A offset_A
observables = Dict("obs_a" => PEtabObservable(scaling_A * A + offset_A, 1.0))

# Create a PEtabODEProblem ReactionNetwork
model_rn = PEtabModel(rn, observables, measurements, parameters;
                      simulation_conditions = simulation_conditions)
petab_problem_rn = PEtabODEProblem(model_rn, verbose=false)
# Create a PEtabODEProblem ODESystem
model_sys = PEtabModel(sys, observables, measurements, parameters,
     simulation_conditions = simulation_conditions)
petab_problem_sys = PEtabODEProblem(model_sys, verbose=false)

# Compute negative log-likelihood
nll_rn = petab_problem_rn.nllh(get_x(petab_problem_rn))
nll_sys = petab_problem_sys.nllh(get_x(petab_problem_sys))
@test nll_rn ≈ 5.69297960953693 atol=1e-3
@test nll_sys ≈ 5.69297960953693 atol=1e-3
