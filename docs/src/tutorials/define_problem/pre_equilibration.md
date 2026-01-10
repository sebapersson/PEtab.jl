# [Pre-equilibration (steady-state initialization)](@id define_with_ss)

Sometime (e.g. perturbation experiments), the model is assumed to be at steady state
at time zero. This can be modeled by first simulating to steady state (pre-equilibration)
and then running the main simulation used for model fitting (optionally with changed control
parameters).

This is handled via **pre-equilibration simulation conditions**. This tutorial shows how to
specify them for a `PEtabModel`, and it assumes familiarity with simulation conditions; see
[Simulation conditions](@ref petab_sim_cond). As a running example, we use the
Michaelis–Menten model from the [starting tutorial](@ref tutorial).

```@example 1
using Catalyst, PEtab

rn = @reaction_network begin
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
nothing # hide
```

## Specifying pre-equilibration conditions

Pre-equilibration conditions are defined with `PEtabCondition`, just like simulation
conditions. For example, assume two simulation conditions with measurements, where `S0`
differs between conditions:

```@example 1
cond1 = PEtabCondition(:cond1, :S0 => 3.0)
cond2 = PEtabCondition(:cond2, :S0 => 5.0)
nothing # hide
```

Further, assume that before simulating `:cond1` and `:cond2`, the system should be simulated
to steady state starting from `S0 = 2.0`. This pre-equilibration condition is defined as:

```@example 1
cond_pre = PEtabCondition(:cond_pre, :S0 => 2.0)
nothing # hide
```

Then, as usual collect all conditions in a `Vector`:

```@example 1
simulation_conditions = [cond_pre, cond1, cond2]
```

## Mapping measurements to pre-equilibration conditions

When using pre-equilibration, each measurement row must specify both the main simulation id
and the pre-equilibration id via `simulation_id` and `pre_eq_id` (column names matter, but
not order):

| simulation_id (str) | pre_eq_id (str) | obs_id (str) | time (float) | measurement (float) |
| ------------------- | --------------- | ------------ | ------------ | ------------------- |
| cond1               | cond_pre        | obs_p        | 1.0          | 2.5                 |
| cond1               | cond_pre        | obs_sum      | 10.0         | 50.0                |
| cond2               | cond_pre        | obs_p        | 1.0          | 2.6                 |
| cond2               | cond_pre        | obs_sum      | 20.0         | 51.0                |

Each row is interpreted as: first simulate to steady state under `pre_eq_id`, then simulate
under `simulation_id` during which the model is compared against measurements. In Julia:

```@example 1
using DataFrames
measurements = DataFrame(
    simulation_id = ["cond1", "cond1", "cond2", "cond2"],
    pre_eq_id     = ["cond_pre", "cond_pre", "cond_pre", "cond_pre"],
    obs_id        = ["obs_p", "obs_sum", "obs_p", "obs_sum"],
    time          = [1.0, 10.0, 1.0, 20.0],
    measurement   = [2.5, 50.0, 2.6, 51.0],
)
```

## Bringing it all together

Given a `Vector` of simulation conditions and `measurements` in the format above, a
`PEtabModel` accounting for pre-equilibration can be created by passing the conditions via
the `simulation_conditions` keyword:

```@example 1
model = PEtabModel(rn, observables, measurements, pest;
    simulation_conditions = simulation_conditions)
petab_prob = PEtabODEProblem(model)
describe(petab_prob)
```

As noted in the printed summary, the `PEtabODEProblem` now includes a `SteadyStateSolver`.
The default solver is typically a good choice, but it can be customized when constructing
`PEtabODEProblem`; see the [API documentation](@ref API).

## Additional pre-equilibration configurations

In the example above, all measurements share the same pre-equilibration condition. PEtab.jl
also supports:

- **Different pre-equilibration conditions**: For this, define multiple pre-equilibration
    conditions with `PEtabCondition` and specify the appropriate id in the `pre_eq_id` of
    the measurement table.
- **No pre-equilibration for subset of measurements**: For this, leave `pre_eq_id` empty in
    the measurement table for measurements without pre-equilibration.
