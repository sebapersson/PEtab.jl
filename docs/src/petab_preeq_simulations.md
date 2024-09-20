# [Pre-Equilibration (Steady-State Simulations)](@id define_with_ss)

Sometimes, like with perturbation experiments, the model should be at a steady state at time zero before the simulation for which the model is compared against data is performed. From a modeling perspective, this can be handled by first simulating the model to a steady state and then, likely by changing some control parameter, perform the main simulation in which the model is compared against data. In PEtab.jl, this is handled by defining pre-equilibration simulation conditions.

This tutorial covers how to specify pre-equilibration conditions for a `PEtabModel`. This tutorial requires that you are familiar with PEtab simulation conditions, if you are not this see this [tutorial](add). As a working example, we use the Michaelis-Menten enzyme kinetics model from the starting [tutorial]. Even though the code below encodes the model as a `ReactionSystem`, everything works exactly the same if the model is encoded as an `ODESystem`.

```@example 1
using Catalyst, PEtab, Plots

t = default_t()
rn = @reaction_network begin
    @parameters begin
        S0
        c3 = 1.0
    end
    @species begin
        SE(t) = S0
    end
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end

@unpack E, S = rn
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

default(left_margin=12.5Plots.Measures.mm, bottom_margin=12.5Plots.Measures.mm) # hide
nothing # hide
```

## Specifying Pre-equilibration Conditions

Pre-equilibration conditions are specified in the same way as simulation conditions. For instance, assume we have two simulation conditions for which we have measurement data (`cond1` and `cond2`), where in `cond1`, the parameter value for `S0`, and by extension the initial value for species `S`, is `3.0`, and in `cond2`, the value for `S0` is `5.0`:

```@example 1
cond1 = Dict(:S0 => 3.0)
cond2 = Dict(:S0 => 5.0)
nothing # hide
```

Additionally, assume that before gathering measurements for conditions `cond1` and `cond2`, the system should be at a steady state starting from `S0 = 2.0`. This pre-equilibration condition (the conditions under which the system reaches steady state) is defined just like any other simulation condition:

```julia
cond_preeq = Dict(:S0 => 2.0)
```

As usual, the condition definitions are bundled together in a `Dict`:

```julia
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

For each measurement, the simulation configuration is interpreted as follows: the model is first simulated to a steady state using the condition specified in the `pre_eq_id`, and then the model is simulated and compared against the data using the condition in the `simulation_id`. In Julia this measurement table would look like:

```julia
using DataFrames
measurements = DataFrame(simulation_id=["cond1", "cond1", "cond2", "cond2"],
                         pre_eq_id=["cond_preeq", "cond_preeq", "cond_preeq", "cond_preeq"],
                         obs_id=["obs_p", "obs_sum", "obs_p", "obs_sum"],
                         time=[1.0, 10.0, 1.0, 20.0],
                         measurement=[0.7, 0.1, 1.0, 1.5])
```

## Bringing It All Together

Given a `Dict` with simulation conditions and `measurements` in the correct format, it is straightforward to create a PEtab problem with pre-equilibration conditions by simply providing the condition `Dict` under the `simulation_conditions` keyword:

```@example 1; ansicolor=false
model = PEtabModel(sys, observables, measurements, pest;
                   simulation_conditions = conds)
petab_prob = PEtabODEProblem(model)
```

From the printout, we can see that the `PEtabODEProblem` now has a `SteadyStateSolver`. Typically, the default solver is a good choice, but if you are interested in more details, see the [API](@ref API) documentation and this [example](@ref steady_state_conditions).

## Additional possible Pre-equilibration Configurations

In the example above each measurment has the same pre-equilibration, and actually every observation has a pre-equilibration. This is not the only configuration, rather PEtab.jl allows for the following when it comes to pre-equilibration:

## Additional Possible Pre-equilibration Configurations

In the example above, each measurement has the same pre-equilibration condition, and all observations have a pre-equilibration condition. However, PEtab.jl allows for more flexibility, and the following configurations are supported:

- **Different pre-equilibration conditions**: If measurements have different pre-equilibration conditions, simply define the appropriate simulation conditions, and specify these in the `pre_eq_id` column of the measurements table.
- **No pre-equilibration for some measurements**: If some measurements do not require pre-equilibration, leave the entry in the `pre_eq_id` column empty for these measurements.
