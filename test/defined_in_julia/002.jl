#=
    Test 0002 from the PEtab test-suite recreated in Julia
=#

rn = @reaction_network begin
    @parameters a0
    @species A(t)=a0
    (k1, k2), A <--> B
end

t = default_t()
D = default_time_deriv()
@mtkmodel SYS begin
    @parameters begin
        a0
        k1
        k2
    end
    @variables begin
        A(t) = a0
        B(t) = 1.0
    end
    @equations begin
        D(A) ~ -k1*A + k2*B
        D(B) ~ k1*A - k2*B
    end
end
@mtkbuild sys = SYS()

speciemap = [:B => 1.0] # Constant initial value for B

# Measurement data
measurements = DataFrame(simulation_id=["c0", "c0", "c1", "c1"],
                         obs_id=["obs_a", "obs_a", "obs_a", "obs_a"],
                         time=[0, 10.0, 0, 10.0],
                         measurement=[0.7, 0.1, 0.8, 0.2])

# Single experimental condition
simulation_conditions = Dict("c0" => Dict(:a0 => 0.8),
                             "c1" => Dict(:a0 => 0.9))

# PEtab-parameter to "estimate"
parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
              PEtabParameter(:k2, value=0.6, scale=:lin)]

# Observable equation
@unpack A = rn
observables = Dict("obs_a" => PEtabObservable(A, 1.0))

# Create a PEtabODEProblem ReactionNetwork
model_rn = PEtabModel(sys, observables, measurements, parameters; speciemap = speciemap,
                     simulation_conditions = simulation_conditions)
petab_prob_rn = PEtabODEProblem(model_rn, verbose=false)
# Create a PEtabODEProblem ODESystem
model_sys = PEtabModel(sys, observables, measurements, parameters;
                       simulation_conditions = simulation_conditions)
petab_prob_sys = PEtabODEProblem(model_sys, verbose=false)

# Compute negative log-likelihood
nll_rn = petab_prob_rn.nllh(get_x(petab_prob_rn))
nll_sys = petab_prob_sys.nllh(get_x(petab_prob_sys))
@test nll_rn ≈ 4.09983582520606 atol=1e-3
@test nll_sys ≈ 4.09983582520606 atol=1e-3
