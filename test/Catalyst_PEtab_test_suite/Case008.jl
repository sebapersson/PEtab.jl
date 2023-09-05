#=
    Recrating PEtab test-suite for Catalyst integration to robustly test that
    we support a wide arrange of features for PEtab integration
=#

using Test
include(joinpath(@__DIR__, "..", "Catalyst_functions.jl"))

# Define reaction network model 
rn = @reaction_network begin
    @parameters a0 b0 offset_A
    @species A(t)=a0 B(t)=b0
    (k1, k2), A <--> B
end

# Measurement data 
measurements = DataFrame(exp_id=["c0", "c0", "c0"],
                         obs_id=["obs_a", "obs_a", "obs_a"],
                         time_point=[0.0, 10.0, 10.0],
                         value=[0.7, 0.1, 0.2])

# Single experimental condition                          
experimental_conditions = Dict("c0" => PEtabExperimentalCondition(Dict()))

# PEtab-parameter to "estimate"
petab_parameters = [PEtabParameter(:a0, value=1.0, scale=:lin),
                    PEtabParameter(:b0, value=0.0, scale=:lin),
                    PEtabParameter(:k1, value=0.8, scale=:lin),
                    PEtabParameter(:k2, value=0.6, scale=:lin)]

# Observable equation                     
@unpack A = rn
observables = Dict("obs_a" => PEtabObservable(A, :lin, 0.5))

# Create a PEtabODEProblem 
petab_model = readPEtabModel(rn, experimental_conditions, observables, measurements,
                            petab_parameters, verbose=true)
petab_problem = createPEtabODEProblem(petab_model)

# Compute negative log-likelihood 
nll = petab_problem.computeCost(petab_problem.θ_nominalT)
@test nll ≈ 1.17778328012676 atol=1e-3