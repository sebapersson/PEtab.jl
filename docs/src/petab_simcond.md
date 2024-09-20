# [Multiple Simulation Conditions](@id petab_sim_cond)

Sometimes measurements are collected under various experimental conditions, where, for example, the initial concentration of a substrate differs between condition. From a modeling viewpoint, experimental conditions typically correspond to different simulation conditions, where the model is simulated with different initial values and/or different values for a set of control parameters.

This tutorial covers how to specify simulation conditions for a `PEtabModel`. As a working example, we use the Michaelis-Menten enzyme kinetics model from the starting [tutorial]. Even though the code below encodes the model as a `ReactionSystem`, everything works exactly the same if the model is encoded as an `ODESystem`.

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

p_c1 = PEtabParameter(:c1)
p_c2 = PEtabParameter(:c2)
p_s0 = PEtabParameter(:S0)
p_sigma = PEtabParameter(:sigma)
pest = [p_c1, p_c2, p_s0, p_sigma]

default(left_margin=12.5Plots.Measures.mm, bottom_margin=12.5Plots.Measures.mm) # hide
nothing # hide
```

## Specifying Simulation Conditions

Simulation conditions should be encoded as a `Dict`, where for each condition, the parameters and/or initial values that change are specified. For instance, assume we have two simulation conditions (`cond1` and `cond2`), where in `cond1`, the initial value for `E` is `0.0` and the parameter `c3` is `1.0`, whereas in `cond2`, the initial value for `E` is `3.0` and the parameter `c3` is `2.0`. This can be encoded as:

```@example 1
cond1 = Dict(:E => 0.0, :c3 => 1.0)
cond2 = Dict(:E => 3.0, :c3 => 2.0)
conds = Dict("cond1" => cond1, "cond2" => cond2)
```

In more detail, when specifying a model species (e.g., `E` above), its initial value is set to the specified value. Meanwhile, for a parameter (e.g., `c3` above), the parameter is set to the given value.

!!! note
    If a parameter or species is specified for one simulation condition, it must be specified for all simulation conditions. This to prevent ambiguity when simulating the model.

## Mapping Measurements to Simulation Conditions

To properly link the measurements to a specific simulation condition, the condition ID must be specified in the measurement `DataFrame`. For our working example, a valid measurement table would look like this (the column names matter, but not the order):

| simulation_id (str) | obs_id (str) | time (float) | measurement (float) |
|---------------------|--------------|--------------|---------------------|
| cond1               | obs_p        | 1.0          | 0.7                 |
| cond1               | obs_sum      | 10.0         | 0.1                 |
| cond2               | obs_p        | 1.0          | 1.0                 |
| cond2               | obs_sum      | 20.0         | 1.5                 |

In Julia this would look like:

```@example 1; ansicolor=false
using DataFrames
measurements = DataFrame(simulation_id=["cond1", "cond1", "cond2", "cond2"],
                         obs_id=["obs_p", "obs_sum", "obs_p", "obs_sum"],
                         time=[1.0, 10.0, 1.0, 20.0],
                         measurement=[0.7, 0.1, 1.0, 1.5])
```

## Bringing It All Together

Given a `Dict` with simulation conditions and `measurements` in the correct format, it is straightforward to create a PEtab problem with multiple simulation conditions by simply providing the condition `Dict` under the `simulation_conditions` keyword:

```@example 1; ansicolor=false
model = PEtabModel(sys, observables, measurements, pest;
                   simulation_conditions = conds)
petab_prob = PEtabODEProblem(model)
```

From plotting the solution of the ODE model for `cond1` and `cond2`, we can clearly see that both the dynamics and initial value for specie `E` differs:

```@example 1; ansicolor=false
using Plots
x = get_x(petab_prob)
sol_cond1 = get_odesol(x, petab_prob; cid = "cond1")
sol_cond2 = get_odesol(x, petab_prob; cid = "cond2")
p1 = plot(sol_cond1, title = "cond1")
p2 = plot(sol_cond2, title = "cond2")
plot(p1, p2)
```
