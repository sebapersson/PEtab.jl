#=
    Recrating PEtab test-suite for Catalyst integration to robustly test that
    we support a wide arrange of features for PEtab integration
=#


using Test
include(joinpath(@__DIR__, "..", "Catalyst_functions.jl"))

# Define reaction network model 
rn = @reaction_network begin
    (k1, k2), A <--> B
end

# Measurement data 
measurements = DataFrame(exp_id=["c0", "c0", "c0", "c0"],
                         pre_eq_id = ["preeq_c0", "preeq_c0", "preeq_c0", "preeq_c0"],
                         obs_id=["obs_a", "obs_a", "obs_a", "obs_b"],
                         time_point=[0.0, 1.0, 10.0, 0.0],
                         value=[0.1, 0.7, 0.1, 0.1])

# Single experimental condition                          
experimental_conditions = Dict("preeq_c0" => PEtabExperimentalCondition(Dict(:k1 => 0.3, :A => 0.0, :B => 2.0)), 
                               "c0" => PEtabExperimentalCondition(Dict(:k1 => 0.8, :A => 1.0, :B => NaN)))

# PEtab-parameter to "estimate"
petab_parameters = [PEtabParameter(:k2, value=0.6, scale=:lin)]

# Observable equation                     
@unpack A, B = rn
observables = Dict("obs_a" => PEtabObservable(A, :lin, 0.5), 
                   "obs_b" => PEtabObservable(B, :lin, 0.2))

# Create a PEtabODEProblem 
petab_model = readPEtabModel(rn, experimental_conditions, observables, measurements,
                            petab_parameters, verbose=true)
petab_problem = createPEtabODEProblem(petab_model)

# Compute negative log-likelihood 
nll = petab_problem.computeCost(petab_problem.θ_nominalT)
@test nll ≈ 6.3898204385477 atol=1e-3