#=
    Recrating PEtab test-suite for Catalyst integration to robustly test that
    we support a wide arrange of features for PEtab integration
=#


# Define reaction network model 
rn = @reaction_network begin
    @parameters a0 b0
    @species A(t)=a0 B(t)=b0
    (k1, k2), A <--> B
end

# Measurement data 
measurements = DataFrame(simulation_id=["c0", "c0"],
                         obs_id=["obs_a", "obs_a"],
                         time=[0, 10.0],
                         measurement=[0.7, 0.1],
                         noise_parameters=0.5)

# Single experimental condition                          
simulation_conditions = Dict("c0" => Dict())

# PEtab-parameter to "estimate"
petab_parameters = [PEtabParameter(:a0, value=1.0, scale=:lin),
                    PEtabParameter(:b0, value=0.0, scale=:lin),
                    PEtabParameter(:k1, value=0.8, scale=:log10, prior=Normal(-0.8, 0.2), prior_on_linear_scale=false),
                    PEtabParameter(:k2, value=0.6, scale=:log10, prior=LogNormal(0.6, 1.0))]

# Observable equation                     
@unpack A = rn
observables = Dict("obs_a" => PEtabObservable(A, 0.5))

# Create a PEtabODEProblem 
petab_model = PEtabModel(rn, simulation_conditions, observables, measurements,
                            petab_parameters, verbose=false)
petab_problem = PEtabODEProblem(petab_model, verbose=false)

# Compute negative log-likelihood 
nll = petab_problem.compute_cost(petab_problem.θ_nominalT)
@test nll ≈ 7.361276133152829 atol=1e-3
