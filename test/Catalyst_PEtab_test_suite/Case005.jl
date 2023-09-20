#=
    Recrating PEtab test-suite for Catalyst integration to robustly test that
    we support a wide arrange of features for PEtab integration
=#


# Define reaction network model 
rn = @reaction_network begin
    @parameters a0 b0 offset_A
    @species A(t)=a0 B(t)=b0
    (k1, k2), A <--> B
end

# Measurement data 
measurements = DataFrame(simulation_id=["c0", "c1"],
                         obs_id=["obs_a", "obs_a"],
                         time=[10.0, 10.0],
                         measurement=[2.1, 3.2])

# Single experimental condition                          
simulation_conditions = Dict("c0" => Dict(:offset_A => :offset_A_c0), 
                             "c1" => Dict(:offset_A => :offset_A_c1))

# PEtab-parameter to "estimate"
petab_parameters = [PEtabParameter(:a0, value=1.0, scale=:lin),
                    PEtabParameter(:b0, value=0.0, scale=:lin),
                    PEtabParameter(:k1, value=0.8, scale=:lin),
                    PEtabParameter(:k2, value=0.6, scale=:lin), 
                    PEtabParameter(:offset_A_c0, value=2.0, scale=:lin),
                    PEtabParameter(:offset_A_c1, value=3.0, scale=:lin)]

# Observable equation                     
@unpack A = rn
@parameters offset_A
observables = Dict("obs_a" => PEtabObservable(A + offset_A, 1.0))

# Create a PEtabODEProblem 
petab_model = readPEtabModel(rn, simulation_conditions, observables, measurements,
                             petab_parameters, verbose=false)
petab_problem = createPEtabODEProblem(petab_model, verbose=false)

# Compute negative log-likelihood 
nll = petab_problem.computeCost(petab_problem.θ_nominalT)
@test nll ≈ 1.91797937195749 atol=1e-3
