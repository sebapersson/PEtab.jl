#=
    Showing all different things we can handle with PEtab.jl 
=#

using Distributions
include(joinpath(@__DIR__, "..", "Catalyst_functions.jl"))

# Define reaction network model 
rn = @reaction_network begin
    @parameters se0
    @species SE(t)=se0 # s0 = initial value for S
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end
#= 
Set a constant state (in this case E and P) and parameter (in this case c1) 
values via a state- and parameter-map respectively with a vector on the form 
Symbol(variable) => value
=#
state_map =  [:E => 1.0, :P => 0.0]
parameter_map = [:c1 => 1.0]

#=
For each simulation condtion we can specify values for states (in this case S) 
and/or model parameters (in this case :c2) via a Dict. For these controll-
parameters values must be set for each simulation condition (I must add a check 
for this). Note - c0_pre corresponds to a pre-equilibration - which is a steady-
state simulation carried out prior to matching the model against data (to mimic 
that cells are at steady-state at time zero).
=#
simulation_conditions = Dict("c0_pre" => Dict(:S => 2.0, :c2 => 2.0),
                             "c0" => Dict(:S => 10.0, :c2 => 2.0))

#=
A PEtabObservable has fields;
    observable formula : (formula for what is observed, e.g S + S)
    measurment noise distribution . (either :lin (normal) or log-normal (:log))
    noise-formula : formula for measurement noise σ

Parameters on the form noiseParameter... maps to the parameter or values specifed 
under noise_parameter in the measurement data, and parameters on the form 
observableParameter... maps to the parameters or values specifed in the 
observable_parameters column in the measurement data. This allows the user to 
specify different, for example, noise parameters per measurement to mimic a setting 
where the same observable (e.g S) might been observed in different experiments, or 
with different assays.
=#
@unpack S, P = rn
@parameters noiseParameter1_obs_S observableParameter1_obs_P observableParameter2_obs_P
observables = Dict("obs_S" => PEtabObservable(S, noiseParameter1_obs_S * S, transformation=:log), 
                   "obs_P" => PEtabObservable(observableParameter1_obs_P*P + observableParameter2_obs_P, 1.0))

#=
The column pre_eq_id corresponds to the pre-equilibration criteria, that is under which 
simulation-condition setting to carry out a steady-state simulation prior to matching the 
model against data (e.g to mimic that cells are at steady-state at time zero). The column 
observable_parameters holds the values for the observable-parameters specified in the 
observable formula, and these can be parameters (specified in PEtabParameters) or constant 
values (e.g 1.0). In case of several values these should be separated by ;. Similiar holds 
for noise_parameter - these map to the noiseParameter... values in the noise-formula, and 
these can be parameters or values. In case there are now noise-parameters, or observable-
parameters these columns can be left out (they are optional).
=#
measurements = DataFrame(simulation_id=["c0", "c0", "c0", "c0"],
                         pre_eq_id=["c0_pre", "c0_pre", "c0_pre", "c0_pre"], # Steady-state pre-eq simulations 
                         obs_id=["obs_S", "obs_S", "obs_P", "obs_P"],
                         time=[0.0, 10.0, 1.0, 20.0],
                         measurement=[0.7, 0.1, 1.0, 1.5], 
                         observable_parameters=["", "", "scale_P;offset_P", "scale_P;offset_P"],
                         noise_parameters=["noise", "noise", "", ""])

#=
For a PEtab parameter to estimate the user must specify id. Then the user can choose scale 
to estimate the parameter on (:log10 (default), :lin and :log). For each parameter any 
univariate cont. prior can be set to turn the parameter estimation problem into a 
maximum a posteriori problem. If prior_on_linear_scale=true (default) the prior applies 
on the linear-scale, while if false it applies on the parameter scale (e.g log10 in case 
the user choose scale=:log10). In case the parameter is not be estimated the user can 
set estimate=false (default true). lb and ub (lower and upper bounds) can also be 
set for each parameter.
=#
petab_parameters = [PEtabParameter(:c3, scale=:log10, prior=Normal(0.0, 2.0), prior_on_linear_scale=false),
                    PEtabParameter(:se0, scale=:lin, prior=LogNormal(1.0, 0.5)), 
                    PEtabParameter(:scale_P, scale=:log10),
                    PEtabParameter(:offset_P, scale=:log10),
                    PEtabParameter(:noise, scale=:log10)]

# Given all this it is easy to set a PEtabODEProblem and then compute the cost, gradient 
# and Hessian.
petab_model = readPEtabModel(rn, simulation_conditions, observables, measurements,
                            petab_parameters, stateMap=state_map, parameterMap=parameter_map, 
                            verbose=true)
petab_problem = createPEtabODEProblem(petab_model)

# Compute negative log-likelihood 
θ = [1.0, 1.0, 1.0, 1.0, 1.0]
f = petab_problem.computeCost(θ)
∇f = petab_problem.computeGradient(θ)
Δf = petab_problem.computeHessian(θ)