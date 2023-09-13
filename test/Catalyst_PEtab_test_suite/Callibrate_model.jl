using Optim
using PyCall
using Ipopt
using QuasiMonteCarlo
using CSV
using Catalyst
using DataFrames
using Distributions
using PEtab

# Define reaction network model
rn = @reaction_network begin
    @parameters a0
    @species A(t)=a0
    (k1, k2), A <--> B
end
state_map = [:B => 1.0] # Constant initial value for B

# Setup PEtab problem 
measurements = DataFrame(simulation_id=["c0", "c0", "c1", "c1"],
                         obs_id=["obs_a", "obs_a", "obs_a", "obs_a"],
                         time=[0, 10.0, 0, 10.0],
                         measurement=[0.7, 0.1, 0.8, 0.2])
simulation_conditions = Dict("c0" => Dict(:a0 => 0.8),
                             "c1" => Dict(:a0 => 0.9))
petab_parameters = [PEtabParameter(:k1, value=0.8, scale=:lin),
                    PEtabParameter(:k2, value=0.6, scale=:lin)]
@unpack A = rn
observables = Dict(["obs_a" => PEtabObservable(A, :lin, 1.0)])
petab_model = readPEtabModel(rn, simulation_conditions, observables, measurements,
                            petab_parameters, verbose=false, stateMap=state_map)
petab_problem = createPEtabODEProblem(petab_model, verbose=false)

# Callibrate the model with different methods
saveTrace = true
p0 = petab_problem.θ_nominalT .* 0.4
# Fides = NewtonTrustRegion with box-constraints
res = callibrateModel(petab_problem, p0, Fides(verbose=true);
                      saveTrace=true)
res = callibrateModel(petab_problem, p0, Fides(:BFGS, verbose=true);
                      saveTrace=true)
# Optim Interior point Newton                      
res = callibrateModel(petab_problem, p0, Optim.IPNewton();
                      saveTrace=true)
# Ipopt Interior point Newton                       
res = callibrateModel(petab_problem, p0, IpoptOptimiser(approximateHessian=false);
                      saveTrace=true, options=IpoptOptions(print_level=5))
res = callibrateModel(petab_problem, p0, IpoptOptimiser(approximateHessian=true);
                      saveTrace=true, options=IpoptOptions(print_level=5))

# Mutlistart optimisation using LathinHypercube sampling, performing 100 multistarts 
dirSave = joinpath(@__DIR__, "Callibration")
res = callibrateModelMultistart(petab_problem, Optim.IPNewton(), 100, nothing, saveTrace=true)
