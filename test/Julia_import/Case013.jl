#=
    Recrating PEtab test-suite for Catalyst integration to robustly test that
    we support a wide arrange of features for PEtab integration. 
    Test not applicable for ODE-system (unless an additional parameter would 
    be added). Also, in real life set B=>par as default
=#


# Define reaction network model 
rn = @reaction_network begin
    @parameters par
    (k1, k2), A <--> B
end
state_map = [:A => 1.0]

# Measurement data 
measurements = DataFrame(simulation_id=["c0", "c0"],
                         obs_id=["obs_a", "obs_a"],
                         time=[0.0, 10.0],
                         measurement=[0.7, 0.1])

# Single experimental condition                          
simulation_conditions = Dict("c0" => Dict(:B => :par))

# PEtab-parameter to "estimate"
petab_parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
                    PEtabParameter(:k2, value=0.6, scale=:lin), 
                    PEtabParameter(:par, value=7.0, scale=:lin)]

# Observable equation                     
@unpack A = rn
observables = Dict("obs_a" => PEtabObservable(A, 0.5))

# Create a PEtabODEProblem ReactionNetwork
petab_model_rn = PEtabModel(rn, simulation_conditions, observables, measurements,
                             petab_parameters, verbose=false, state_map=state_map)
petab_problem_rn = PEtabODEProblem(petab_model_rn, verbose=false)

# Compute negative log-likelihood 
nll_rn = petab_problem_rn.compute_cost(petab_problem_rn.θ_nominalT)
@test nll_rn ≈ 22.79033132827511 atol=1e-3
