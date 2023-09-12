#=
    Recrating PEtab test-suite for Catalyst integration to robustly test that
    we support a wide arrange of features for PEtab integration
=#


# Define reaction network model 
rn = @reaction_network begin
    (k1, k2), A <--> B
end
state_map = [:A => 1.0]

# Measurement data 
measurements = DataFrame(pre_eq_id=["preeq_c0", "preeq_c0"],
                         simulation_id=["c0", "c0"],
                         obs_id=["obs_a", "obs_a"],
                         time=[1.0, 10.0],
                         measurement=[0.7, 0.1])

# Single experimental condition                          
simulation_conditions = Dict("c0" => Dict(:k1 => 0.8, :B => 1.0), 
                             "preeq_c0" => Dict(:k1 => 0.3, :B => 0.0))

# PEtab-parameter to "estimate"
petab_parameters = [PEtabParameter(:k2, value=0.6, scale=:lin)]

# Observable equation                     
@unpack A = rn
observables = Dict("obs_a" => PEtabObservable(A, :lin, 0.5))

# Create a PEtabODEProblem 
petab_model = readPEtabModel(rn, simulation_conditions, observables, measurements,
                            petab_parameters, verbose=false, stateMap=state_map)
petab_problem = createPEtabODEProblem(petab_model, verbose=false)

# Compute negative log-likelihood 
nll = petab_problem.computeCost(petab_problem.θ_nominalT)
@test nll ≈ 1.20628941926143 atol=1e-3

