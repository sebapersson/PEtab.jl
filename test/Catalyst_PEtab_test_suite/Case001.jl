#= 
    Recrating PEtab test-suite for Catalyst integration to robustly test that 
    we support a wide arrange of features for PEtab integration 
=#

include(joinpath(@__DIR__, "..", "Catalyst_functions.jl"))

rn = @reaction_network begin
    @parameters a0 b0
    (k1), A --> B
    (k2), B --> A
end

@unpack A, B, k1, k2, a0, b0 = rn
state_map = [A => a0, B => b0]

experimental_conditions = Dict(["c0" => PEtabExperimentalCondition(Dict())])

measurements = DataFrame(exp_id=["c0", "c0"],
                         obs_id=["obs_a", "obs_a"], 
                         time_point=[0, 10.0],
                         value=[0.7, 0.1], 
                         noise_parameter=0.5)

petab_parameters = [PEtabParameter(a0, true, 1.0, nothing, nothing, nothing, :lin), 
                    PEtabParameter(b0, true, 0.0, nothing, nothing, nothing, :lin), 
                    PEtabParameter(k1, true, 0.8, nothing, nothing, nothing, :lin),
                    PEtabParameter(k2, true, 0.6, nothing, nothing, nothing, :lin)]                         

obs1 = PEtabObservable(A, :lin, 0.5)
observables = Dict(["obs_a" => obs1])
                    
# Creates the model
petab_model = readPEtabModel(rn, experimental_conditions, observables, measurements, 
                             petab_parameters, state_map, verbose=true)

petab_problem = createPEtabODEProblem(petab_model)
nll = petab_problem.computeCost(petab_problem.θ_nominalT)
@test nll ≈ 0.84750169713188 atol=1e-3