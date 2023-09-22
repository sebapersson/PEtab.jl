#=
    Recrating PEtab test-suite for Catalyst integration to robustly test that
    we support a wide arrange of features for PEtab integration
=#


# Define reaction network model 
rn = @reaction_network begin
    (k1, k2), A <--> B
end

@parameters k1 k2
@variables t A(t) B(t)
D = Differential(t)
eqs = [
    D(A) ~ -k1*A + k2*B
    D(B) ~ k1*A - k2*B
]
@named sys = ODESystem(eqs)

# Measurement data 
measurements = DataFrame(simulation_id=["c0", "c0", "c0", "c0"],
                         pre_eq_id = ["preeq_c0", "preeq_c0", "preeq_c0", "preeq_c0"],
                         obs_id=["obs_a", "obs_a", "obs_a", "obs_b"],
                         time=[0.0, 1.0, 10.0, 0.0],
                         measurement=[0.1, 0.7, 0.1, 0.1])

# Single experimental condition                          
simulation_conditions = Dict("preeq_c0" => Dict(:k1 => 0.3, :A => 0.0, :B => 2.0), 
                             "c0" => Dict(:k1 => 0.8, :A => 1.0, :B => NaN))

# PEtab-parameter to "estimate"
petab_parameters = [PEtabParameter(:k2, value=0.6, scale=:lin)]

# Observable equation                     
@unpack A, B = rn
observables = Dict("obs_a" => PEtabObservable(A, 0.5), 
                   "obs_b" => PEtabObservable(B, 0.2))

# Create a PEtabODEProblem ReactionNetwork
petab_model_rn = PEtabModel(rn, simulation_conditions, observables, measurements,
                             petab_parameters, verbose=false)
petab_problem_rn = PEtabODEProblem(petab_model_rn, verbose=false)
# Create a PEtabODEProblem ODESystem
petab_model_sys = PEtabModel(sys, simulation_conditions, observables, measurements,
                             petab_parameters, verbose=false)
petab_problem_sys = PEtabODEProblem(petab_model_sys, verbose=false)

# Compute negative log-likelihood 
nll_rn = petab_problem_rn.compute_cost(petab_problem_rn.θ_nominalT)
nll_sys = petab_problem_sys.compute_cost(petab_problem_sys.θ_nominalT)
@test nll_rn ≈ 6.3898204385477 atol=1e-3
@test nll_sys ≈ 6.3898204385477 atol=1e-3