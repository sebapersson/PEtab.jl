using ModelingToolkit
using Catalyst
using OrdinaryDiffEq
using Distributions
using DataFrames
using Plots
using PEtab
using CSV
using Tables
using Printf


include(joinpath(@__DIR__, "Catalyst_functions.jl"))

# Prepares the model simulation conditions.
rn = @reaction_network begin
    (p,d), 0 <--> X
    (1,k), X <--> Y
end


### Performs Experiment (simulate data) ###
u0 = [1.0, 0.0]
tspan = (0.0, 10.0)
p = [1.0, 0.2, 0.5]
meassurment_times = 2.0:2.0:10.0
meassurment_error_dist = Normal(0.0, 0.4)
oprob = ODEProblem(rn, u0, tspan, p)
sol = solve(oprob, Tsit5(); tstops=meassurment_times)
meassured_values_true = [sol[2, findfirst(sol.t .== mt)] for mt in meassurment_times]
meassured_values = meassured_values_true .+ rand(meassurment_error_dist, length(meassured_values_true))


### Prepares the PEtab Structure ###
@unpack X, Y, p, d, k = rn

# The system field.
system = rn

# The experimental conditions field.
exp1 = PEtabExperimentalCondition(Dict([d=>0.2, k=>2.0]))
experimental_conditions = Dict(["Exp1" => exp1])

# The observables field.
obs1 = PEtabObservable((X + Y) / X, :lin, 0.4)
observables = Dict(["Obs1" => obs1])

# The meassurments field
nM = length(meassured_values)
meassurments = DataFrame(exp_id=fill("Exp1", nM), obs_id=fill("Obs1", nM), value=meassured_values, time_point=meassurment_times, noise_parameter=fill(0.4,nM))

# The parameters field (we are going to create a nice constructor here)
par_p = PEtabParameter(p, true, nothing, 1e-2, 1e2, nothing, :log10)
par_d = PEtabParameter(d, false, 0.2, nothing, nothing, nothing, nothing)
par_k = PEtabParameter(k, false, 2.0, nothing, nothing, nothing, nothing)
petab_parameters = [par_p, par_d, par_k]

# The initial conditions field.
state_map = [X => 1.0, Y => 0.0]

# Creates the model
petab_model = readPEtabModel(system, experimental_conditions, observables, meassurments, 
                             petab_parameters, state_map, verbose=true)

# Can now easily be made into PEtabODEProblem
petabProblem = createPEtabODEProblem(petab_model)
p = [log10(20)]
f = petabProblem.computeCost(p)
∇f = petabProblem.computeGradient(p)
Δf = petabProblem.computeHessian(p)


#= 
    Important things to discuss:

    1. If a parameter species an initial value that parameter needs to be a part of 
       of parameters(systems) - otherwise gradients will not be computed correctly. 
       Currently we must use @parameters a0 in the systems, and in the state-map 
       set [A => a0] - feels a bit redundant. (see how I had to do it in the 
       first test-case). 

    2. Setting default values already in the system, can this be done? Otherwise 
       all constant parameters would have to be set in PEtab-parameters. 
=#