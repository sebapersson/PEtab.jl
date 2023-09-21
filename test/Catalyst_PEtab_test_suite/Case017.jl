#=
    Recrating PEtab test-suite for Catalyst integration to robustly test that
    we support a wide arrange of features for PEtab integration
=#


# Define reaction network model 
rn = @reaction_network begin
    (k1, k2), A <--> B
end

# Measurement data 
measurements = DataFrame(simulation_id=["c0", "c0"],
                         pre_eq_id = ["preeq_c0", "preeq_c0"],
                         obs_id=["obs_a", "obs_a"],
                         time=[1.0, 10.0],
                         measurement=[0.7, 0.1])

# Single experimental condition                          
simulation_conditions = Dict("preeq_c0" => Dict(:k1 => 0.3, :A => 0.0, :B => 2.0), 
                             "c0" => Dict(:k1 => 0.8, :A => 1.0, :B => NaN))

# PEtab-parameter to "estimate"
petab_parameters = [PEtabParameter(:k2, value=0.6, scale=:lin)]

# Observable equation                     
@unpack A = rn
observables = Dict("obs_a" => PEtabObservable(A, 0.5))

# Create a PEtabODEProblem 
petab_model = PEtabModel(rn, simulation_conditions, observables, measurements,
                            petab_parameters, verbose=false)
petab_problem = PEtabODEProblem(petab_model, verbose=false)

# Compute negative log-likelihood 
nll = petab_problem.compute_cost(petab_problem.θ_nominalT)
@test nll ≈ 1.22063957624351 atol=1e-3