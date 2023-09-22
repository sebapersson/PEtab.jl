# [Pre-Equilibration (Steady-State Simulations)](@id define_with_ss)

In certain scenarios, such as during perturbation experiments, the model should be at a steady state at time zero before it is matched against data. This can be achieved by defining pre-equilibration simulation conditions. To demonstrate how to use these, let us consider the same enzyme kinetics model as used in the [Creating a PEtab Parameter Estimation Problem in Julia](@ref define_in_julia) tutorial.

```julia
using Catalyst 
using Distributions
using PEtab

rn = @reaction_network begin
    @parameters se0
    @species SE(t) = se0  # se0 = initial value for S
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end

# Define state and parameter maps
state_map =  [:E => 1.0, :P => 0.0]
parameter_map = [:c1 => 1.0]

# Unpack model components
@unpack P, E, SE = rn
@parameters sigma, scale, offset

# Define observables
obs_P = PEtabObservable(scale * P + offset, sigma * P, transformation=:lin)
obs_Sum = PEtabObservable(E + SE, 3.0, transformation=:log)
observables = Dict("obs_P" => obs_P, 
                   "obs_Sum" => obs_Sum) 

# Define parameters for estimation
_c3 = PEtabParameter(:c3, scale=:log10)
_se0 = PEtabParameter(:c3, prior=LogNormal(1.0, 0.5), prior_on_linear_scale=true)
_c2 = PEtabParameter(:c2)
_sigma = PEtabParameter(:sigma)
_scale = PEtabParameter(:scale)
_offset = PEtabParameter(:offset)
parameters = [_c2, _c3, _se0, _sigma, _scale, _offset]

# Define simulation conditions
condition_c0 = Dict(:S => 5.0)
condition_c1 = Dict(:S => 2.0)
simulation_conditions = Dict("c0" => condition_c0, 
                             "c1" => condition_c1)
```

Now, let us assume that before gathering data for conditions `c0` and `c1`, 2mM of the substrate `S` was added to the system, which was then allowed to settle into a steady state. You can define this pre-equilibration condition (the conditions under which the system reached steady state) just like any other simulation condition and then add it to `simulation_conditions`.

```julia
condition_c_pre = Dict(:S => 2.0)
simulation_conditions["c0_pre"] = condition_c_pre
```

To ensure that the correct pre-equilibration simulation is performed when simulating the model during parameter estimation, add a new column `pre_eq_id` to the measurement data:

| simulation_id (str) | pre\_eq\_id (str) | obs_id (str) | time (float) | measurement (float) |
|---------------------|-----------------|--------------|--------------|---------------------|
| c0                  | c_pre           | obs_P        | 0.0          | 0.7                 |
| c0                  | c_pre           | obs_Sum      | 10.0         | 0.1                 |
| c1                  | c_pre           | obs_P        | 1.0          | 1.0                 |
| c1                  | c_pre           | obs_Sum      | 20.0         | 1.5                 |

In Julia, it would look like this:

```julia
using DataFrames
measurements = DataFrame(
    simulation_id=["c0", "c0", "c1", "c1"],
    pre_eq_id=["c0_pre", "c0_pre", "c0_pre", "c0_pre"], # Steady-state pre-eq simulations 
    obs_id=["obs_P", "obs_Sum", "obs_P", "obs_Sum"],
    time=[0.0, 10.0, 1.0, 20.0],
    measurement=[0.7, 0.1, 1.0, 1.5]
)
```

With this setup, you can create a `PEtabODEProblem` for model calibration:

```julia
petab_model = PEtabModel(
    rn, simulation_conditions, observables, measurements,
    parameters, state_map=state_map, parameter_map=parameter_map, verbose=true
)
petab_problem = PEtabODEProblem(petab_model)  
```

Note that you can specify multiple pre-equilibration conditions if needed. For information on different `SteadyStateSolver`, see [this](@ref steady_state_conditions) tutorial.