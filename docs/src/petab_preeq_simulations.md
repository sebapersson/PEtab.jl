# [Steady-State Simulations (Pre-Equilibration)](@id define_with_ss)

Sometimes, such as with perturbation experiments, the model should be at a steady state at time zero before performing the simulation that is compared against data. From a modeling perspective, this can be handled by first simulating the model to reach a steady state and then, possibly by changing some control parameters, perform the main simulation. In PEtab.jl, this is handled by defining pre-equilibration simulation conditions.

This tutorial covers how to specify pre-equilibration conditions for a `PEtabModel`. It requires that you are familiar with PEtab simulation conditions, if not; see this [tutorial](@ref petab_sim_cond). As a working example, we use the Michaelis-Menten enzyme kinetics model from the starting [tutorial](@ref tutorial). Even though the code below encodes the model as a `ReactionSystem`, everything works exactly the same if the model is encoded as an `ODESystem`.

```@example 1
using Catalyst, PEtab

t = default_t()
rn = @reaction_network begin
    @parameters S0 c3=1.0
    @species S(t)=S0
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end
speciemap = [:E => 50.0, :SE => 0.0, :P => 0.0]

@unpack E, S, P = rn
@parameters sigma
obs_sum = PEtabObservable(S + E, 3.0)
obs_p = PEtabObservable(P, sigma)
observables = Dict("obs_p" => obs_p, "obs_sum" => obs_sum)

# Unlike the starting tutorial we do not estimate S0 here as it below 
# dictates simulation conditions
p_c1 = PEtabParameter(:c1)
p_c2 = PEtabParameter(:c2)
p_sigma = PEtabParameter(:sigma)
pest = [p_c1, p_c2, p_sigma]
nothing # hide
```

## Specifying Pre-equilibration Conditions

Pre-equilibration conditions are specified in the same way as simulation conditions. For instance, assume we have two simulation conditions for which we have measurement data (`cond1` and `cond2`), where in `cond1`, the parameter value for `S0` is `3.0`, and in `cond2`, the value for `S0` is `5.0`:

```@example 1
cond1 = Dict(:S0 => 3.0)
cond2 = Dict(:S0 => 5.0)
nothing # hide
```

Additionally, assume that before gathering measurements for conditions `cond1` and `cond2`, the system should be at a steady state starting from `S0 = 2.0`. This pre-equilibration condition (the conditions under which the system reaches steady state) is defined just like any other simulation condition:

```@example 1
cond_preeq = Dict(:S0 => 2.0)
nothing # hide
```

As usual, the condition should be collected in a `Dict`:

```@example 1
conds = Dict("cond_preeq" => cond_preeq, "cond1" => cond1, "cond2" => cond2)
```

## Mapping Measurements to Pre-equilibration Conditions

To properly link the measurements to a specific simulation configuration, both the main simulation ID and the pre-equilibration ID must be specified in the measurements `DataFrame`. For our working example, a valid measurement table would look like this (the column names matter, but not the order):

| simulation_id (str) | pre\_eq\_id (str) | obs_id (str) | time (float) | measurement (float) |
|---------------------|-------------------|--------------|--------------|---------------------|
| cond1               | cond_preeq        | obs_p        | 1.0          | 0.7                 |
| cond1               | cond_preeq        | obs_sum      | 10.0         | 0.1                 |
| cond2               | cond_preeq        | obs_p        | 1.0          | 1.0                 |
| cond2               | cond_preeq        | obs_sum      | 20.0         | 1.5                 |

For each measurement, the simulation configuration is interpreted as follows: the model is first simulated to a steady state using the condition specified in the `pre_eq_id` column, and then the model is simulated and compared against the data using the condition in the `simulation_id`. In Julia this measurement table would look like:

```@example 1
using DataFrames
measurements = DataFrame(simulation_id=["cond1", "cond1", "cond2", "cond2"],
                         pre_eq_id=["cond_preeq", "cond_preeq", "cond_preeq", "cond_preeq"],
                         obs_id=["obs_p", "obs_sum", "obs_p", "obs_sum"],
                         time=[1.0, 10.0, 1.0, 20.0],
                         measurement=[0.7, 0.1, 1.0, 1.5])                         
```

## Bringing It All Together

Given a `Dict` with simulation conditions and `measurements` in the correct format, it is straightforward to create a PEtab problem with pre-equilibration conditions by providing the condition `Dict` under the `simulation_conditions` keyword:

```@example 1
model = PEtabModel(rn, observables, measurements, pest;
                   simulation_conditions = conds)
petab_prob = PEtabODEProblem(model)
```

From the printout, we see that the `PEtabODEProblem` now has a `SteadyStateSolver`. The default steady-state solver is generally a good choice, but if you are interested in more details, see to the [API](@ref API) and [this] ADD! example.

## Additional Possible Pre-equilibration Configurations

In the example above, each measurement has the same pre-equilibration condition, and all observations have a pre-equilibration condition. PEtab.jl also allows for more flexibility, and the following configurations are supported:

- **Different pre-equilibration conditions**: If measurements have different pre-equilibration conditions, simply define these as simulation conditions, and specify the corresponding condition in the `pre_eq_id` column of the measurements table.
- **No pre-equilibration for some measurements**: If some measurements do not require pre-equilibration, leave the entry in the `pre_eq_id` column empty for these measurements.
