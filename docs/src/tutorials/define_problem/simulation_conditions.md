```@meta
CollapsedDocStrings=true
```

# [Simulation conditions](@id petab_sim_cond)

Sometimes measurements are collected under different experimental conditions, which in the
model correspond to simulating with different initial values and/or control parameter
values.

This is handled via **simulation conditions** defined via `PEtabCondition`. This tutorial
shows how to specify simulation conditions. As a running example, we use the
Michaelis-Menten model from the [starting tutorial](@ref tutorial):

:::tabs

== Model as ReactionSystem

```@example 1
using Catalyst, PEtab

sys = @reaction_network begin
    @parameters S0 c3=3.0
    @species begin
        S(t) = S0
        E(t) = 50.0
        SE(t) = 0.0
        P(t) = 0.0
    end
    @observables begin
        obs1 ~ S + E
        obs2 ~ P
    end
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end

@parameters sigma
petab_obs1 = PEtabObservable(:petab_obs1, :obs1, 3.0)
petab_obs2 = PEtabObservable(:petab_obs2, :obs2, sigma)
observables = [petab_obs1, petab_obs2]

p_c1 = PEtabParameter(:c1)
p_c2 = PEtabParameter(:c2)
p_S0 = PEtabParameter(:S0)
p_sigma = PEtabParameter(:sigma)
pest = [p_c1, p_c2, p_S0, p_sigma]
using Plots # hide
default(left_margin=12.5Plots.Measures.mm, bottom_margin=12.5Plots.Measures.mm, size = (600*1.25, 400 * 1.25), palette = ["#CC79A7", "#009E73", "#0072B2", "#D55E00", "#999999", "#E69F00", "#56B4E9", "#F0E442"], linewidth=4.0) # hide
```

== Model as ODESystem

```@example 2
using ModelingToolkit, PEtab
using ModelingToolkit: t_nounits as t, D_nounits as D

@mtkmodel SYS begin
    @parameters begin
        S0
        c1
        c2
        c3 = 1.0
    end
    @variables begin
        S(t) = S0
        E(t) = 50.0
        SE(t) = 0.0
        P(t) = 0.0
        # Observables
        obs1(t)
        obs2(t)
    end
    @equations begin
        # Dynamics
        D(S) ~ -c1 * S * E + c2 * SE
        D(E) ~ -c1 * S * E + c2 * SE + c3 * SE
        D(SE) ~ c1 * S * E - c2 * SE - c3 * SE
        D(P) ~ c3 * SE
        # Observables
        obs1 ~ S + E
        obs2 ~ P
    end
end
@mtkbuild sys = SYS()

@parameters sigma
petab_obs1 = PEtabObservable(:petab_obs1, :obs1, 3.0)
petab_obs2 = PEtabObservable(:petab_obs2, :obs2, sigma)
observables = [petab_obs1, petab_obs2]

p_c1 = PEtabParameter(:c1)
p_c2 = PEtabParameter(:c2)
p_S0 = PEtabParameter(:S0)
p_sigma = PEtabParameter(:sigma)
pest = [p_c1, p_c2, p_S0, p_sigma]
nothing # hide
```

:::

## Defining simulation conditions

Simulation conditions are specified with `PEtabCondition`:

```@docs; canonical=false
PEtabCondition
```

Specifically, each `PEtabCondition` defines a unique condition id (used to link rows in the
measurement table) and one or more assignments of the form `target_id => target_value`. For
example, assume two conditions: in `:cond1`, the initial value of `E` is `40.0` and the
value of parameter `c3` is `0.5`, while in `:cond2`, `E = 100.0` and `c3 = 2.0`:

```@example 1
cond1 = PEtabCondition(:cond1, :E => 40.0, :c3 => 0.5)
cond2 = PEtabCondition(:cond2, :E => 100.0, :c3 => 2.0)
simulation_conditions = [cond1, cond2]
```

All conditions should be collected in a `Vector`. If a model state or parameter is not
assigned in a given condition, its default value from the model is used.

## Mapping measurements to simulation conditions

Given simulation conditions, each measurement must be linked to a simulation condition via a
`simulation_id` column in the measurement `DataFrame` (column names matter, but not order):

| simulation_id (str) | obs_id (str) | time (float) | measurement (float) |
| ------------------- | ------------ | ------------ | ------------------- |
| cond1               | petab_obs2   | 1.0          | 0.7                 |
| cond1               | petab_obs1   | 10.0         | 0.1                 |
| cond2               | petab_obs2   | 1.0          | 1.0                 |
| cond2               | petab_obs1   | 20.0         | 1.5                 |

**Note** that the ids in `simulation_id` must match the ids defined above with
`PEtabCondition`. In Julia:

```@example 1
using DataFrames
measurements = DataFrame(
    simulation_id = ["cond1", "cond1", "cond2", "cond2"],
    obs_id        = ["petab_obs2", "petab_obs1", "petab_obs2", "petab_obs1"],
    time          = [5.0, 10.0, 1.0, 20.0],
    measurement   = [0.7, 0.1, 1.0, 1.5],
)
```

## Bringing it all together

Given a `Vector` of simulation conditions and `measurements` in the format above, a
`PEtabModel` accounting for multiple simulation conditions can be created by passing the
conditions via the `simulation_conditions` keyword:

```@example 1
model = PEtabModel(sys, observables, measurements, pest;
    simulation_conditions = simulation_conditions)
petab_prob = PEtabODEProblem(model)
nothing # hide
```

Plotting the model solution under `:cond1` and `:cond2` shows the effect of the different
condition settings (e.g. the initial value of `E`):

```@example 1
using Plots
x = get_x(petab_prob)
sol_cond1 = get_odesol(x, petab_prob; condition = :cond1)
sol_cond2 = get_odesol(x, petab_prob; condition = :cond2)
p1 = plot(sol_cond1, title = "cond1")
p2 = plot(sol_cond2, title = "cond2")
plot(p1, p2)
plot(p1, p2; size = (800, 400)) # hide
```

## Changing simulation start time

By default, the model is simulated from `t0 = 0.0` for each each simulation condition. This
can be changed via the `t0` keyword in `PEtabCondition`. For example, to simulate the model
from `t0 = 5.0` for `:cond1`:

```@example 1
cond1 = PEtabCondition(:cond1, :E => 0.0, :c3 => 0.5; t0 = 5.0)
```

As seen by plotting, the simulation for `:cond1` now starts at `t = 5.0`:

```@example 1
model = PEtabModel(sys, observables, measurements, pest;
    simulation_conditions = [cond1, cond2])
petab_prob = PEtabODEProblem(model)

x = get_x(petab_prob)
sol_cond1 = get_odesol(x, petab_prob; condition = :cond1)
sol_cond2 = get_odesol(x, petab_prob; condition = :cond2)

p1 = plot(sol_cond1, title = "cond1")
p2 = plot(sol_cond2, title = "cond2")
plot(p1, p2)
plot(p1, p2; size = (800, 400)) # hide
```

!!! tip "Change simulation start time without simulation conditions"
    If a model does not use simulation conditions, simulation start time can be changed by
    creating an empty condition: `cond = PEtabCondition(:cond1; t0 = 5.0)`
