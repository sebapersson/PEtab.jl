include(joinpath(@__DIR__, "..", "Catalyst_functions.jl"))

using Optim
using QuasiMonteCarlo
using PyCall
using Ipopt

# Define reaction network model 
rn = @reaction_network begin
    @parameters a0
    @species A(t)=a0
    (k1, k2), A <--> B
end
state_map = [:B => 1.0] # Constant initial value for B 

# Measurement data 
measurements = DataFrame(exp_id=["c0", "c0", "c1", "c1"],
                         obs_id=["obs_a", "obs_a", "obs_a", "obs_a"],
                         time_point=[0, 10.0, 0, 10.0],
                         value=[0.7, 0.1, 0.8, 0.2])

# Single experimental condition                          
experimental_conditions = Dict("c0" => PEtabExperimentalCondition(Dict(:a0 => 0.8)), 
                               "c1" => PEtabExperimentalCondition(Dict(:a0 => 0.9)))

# PEtab-parameter to "estimate"
petab_parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
                    PEtabParameter(:k2, value=0.6, scale=:lin)]

# Observable equation                     
@unpack A = rn
observables = Dict(["obs_a" => PEtabObservable(A, :lin, 1.0)])

# Create a PEtabODEProblem 
petab_model = readPEtabModel(rn, experimental_conditions, observables, measurements,
                            petab_parameters, verbose=false, stateMap=state_map)
petab_problem = createPEtabODEProblem(petab_model, verbose=false)

# Callibrate the model with different methods 
saveTrace = true
p0 = petab_problem.θ_nominalT .* 0.4
res = callibrateModel(petab_problem, p0, Fides(verbose=true); 
                      saveTrace=true)
res = callibrateModel(petab_problem, p0, Fides(:BFGS, verbose=true); 
                      saveTrace=true)           
res = callibrateModel(petab_problem, p0, Optim.IPNewton(); 
                      saveTrace=true)                                 
res = callibrateModel(petab_problem, p0, IpoptOptimiser(approximateHessian=true); 
                      saveTrace=true, options=IpoptOptions(print_level=5))
res = callibrateModel(petab_problem, p0, IpoptOptimiser(approximateHessian=true); 
                      saveTrace=true, options=IpoptOptions(print_level=5))                                                

