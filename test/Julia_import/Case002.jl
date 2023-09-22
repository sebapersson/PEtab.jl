#=
    Recrating PEtab test-suite for Catalyst integration to robustly test that
    we support a wide arrange of features for PEtab integration
=#


# Define reaction network model 
rn = @reaction_network begin
    @parameters a0
    @species A(t)=a0
    (k1, k2), A <--> B
end

@parameters a0 k1 k2
@variables t A(t) B(t)
D = Differential(t)
eqs = [
    D(A) ~ -k1*A + k2*B
    D(B) ~ k1*A - k2*B
]
@named sys = ODESystem(eqs; defaults=Dict(A => a0))

state_map = [:B => 1.0] # Constant initial value for B 

# Measurement data 
measurements = DataFrame(simulation_id=["c0", "c0", "c1", "c1"],
                         obs_id=["obs_a", "obs_a", "obs_a", "obs_a"],
                         time=[0, 10.0, 0, 10.0],
                         measurement=[0.7, 0.1, 0.8, 0.2])

# Single experimental condition                          
simulation_conditions = Dict("c0" => Dict(:a0 => 0.8), 
                             "c1" => Dict(:a0 => 0.9))

# PEtab-parameter to "estimate"
petab_parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
                    PEtabParameter(:k2, value=0.6, scale=:lin)]

# Observable equation                     
@unpack A = rn
observables = Dict("obs_a" => PEtabObservable(A, 1.0))

# Create a PEtabODEProblem ReactionNetwork
petab_model_rn = PEtabModel(rn, simulation_conditions, observables, measurements,
                            petab_parameters, verbose=false, state_map=state_map)
petab_problem_rn = PEtabODEProblem(petab_model_rn, verbose=false)
# Create a PEtabODEProblem ODESystem
petab_model_sys = PEtabModel(sys, simulation_conditions, observables, measurements,
                             petab_parameters, verbose=false, state_map=state_map)
petab_problem_sys = PEtabODEProblem(petab_model_sys, verbose=false)

# Compute negative log-likelihood 
nll_rn = petab_problem_rn.compute_cost(petab_problem_rn.θ_nominalT)
nll_sys = petab_problem_sys.compute_cost(petab_problem_sys.θ_nominalT)
@test nll_rn ≈ 4.09983582520606 atol=1e-3
@test nll_sys ≈ 4.09983582520606 atol=1e-3
