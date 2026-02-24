# [ML models in observables](@id observable_ml)

Mechanistic models can be misspecified, or the mapping from model states to measurements
may be only partially known. Both scenarios can be addressed by augmenting the observable
formula in `PEtabObservable` with a neural network.

This tutorial shows how to include an ML model in the observable formula. It assumes
familiarity with the [SciML starter tutorial](@ref sciml_starter). As a running example,
the Michaelis-Menten model from the mechanistic [starting tutorial](@ref tutorial) is used:

== Model as ReactionSystem

```@example 1
using Catalyst
t = Catalyst.default_t()
sys = @reaction_network begin
    @parameters S0 c3=3.0
    @species begin
        S(t) = S0
        E(t) = 50.0
        SE(t) = 0.0
        P(t) = 0.0
    end
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end
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
        SE(t) = 0.1
        P(t) = 0.1
    end
    @equations begin
        # Dynamics
        D(S) ~ -c1 * S * E + c2 * SE
        D(E) ~ -c1 * S * E + c2 * SE + c3 * SE
        D(SE) ~ c1 * S * E - c2 * SE - c3 * SE
        D(P) ~ c3 * SE
    end
end
@mtkbuild sys = SYS()
nothing # hide
```

## Defining ML model in observable formulas

An ML model can be embedded in the observable formula of a `PEtabObservable` by (1)
defining a Lux.jl model and (2) wrapping it as an `MLModel` that specifies its inputs and
declares an output variable which later can be referenced in observable formulas. For
example, assume the ML model takes the states `S` and `E` as input:

```@example 1
using Lux, PEtab
lux_model = Lux.Chain(
    Dense(2 => 5, Lux.swish),
    Dense(5 => 1),
)

@variables S(t) E(t)
ml_model = MLModel(
    :net1, lux_model, false; inputs = [S, E], outputs = [:output1]
)
```

Here, `false` indicates that the ML model is not evaluated pre-simulation; instead it is
evaluated when observables are computed. The output variable `output1` can then be used in
observable formulas:

```@example 1
@variables P(t) output1(t)
@parameters sigma
observables = [
    PEtabObservable(:obs_p, P, 3.0),
    PEtabObservable(:obs_sum, output1, sigma),
]
```

Note, when an ML model appears in an observable, the observable formula should be defined in
`PEtabObservable` (rather than in the model system). This allows PEtab.jl to track ML
usage and compute gradients more efficiently. Further, while the inputs are states here,
they can general expressions of model quantities.

Given the `PEtabObservable`s, the rest of the `PEtabODEProblem` is defined as usual:

```@example 1
using DataFrames
pest = [
    PEtabParameter(:c1),
    PEtabParameter(:c2),
    PEtabParameter(:S0),
    PEtabParameter(:sigma),
    PEtabMLParameter(:net1), # ML parameters
]

measurements = DataFrame(
    obs_id = ["obs_p", "obs_sum", "obs_p", "obs_sum"],
    time = [1.0, 10.0, 1.0, 20.0],
    measurement = [0.7, 0.1, 1.0, 1.5],
)

petab_model = PEtabModel(
    sys, observables, measurements, pest; ml_models = ml_model
)
petab_prob = PEtabODEProblem(petab_model)
```

## Simulation-condition input

When an ML model is used in the observable formula, additional informative non-time-series
data (e.g. images or other covariates) may be available per simulation condition. Such
data can be included by giving the ML model multiple inputs: one based on model quantities
(e.g. states) and one provided via `PEtabCondition`. The general approach for multiple
input arguments is described in [Pre-simulation ML models](@ref pre_simulate_ml); this
section focuses on the observable case.

As a concrete example, assume the ML model takes the states `[S, E]`, as well as static
simulation-condition-specific data provided via `input2`. The first step is to define the
`MLModel`:

```@example 1
lux_model = @compact(
    layer1 = Dense(6 => 5, Lux.swish),
    layer2 = Dense(5 => 1),
) do (x1, x2)
    x = cat(x1, x2; dims = 1)
    h = layer1(x)
    out = layer2(h)
    @return out
end

ml_model = MLModel(
    :net1, lux_model, false; inputs = ([S, E], [:input2]), outputs = [:output1]
)
nothing # hide
```

The variable `input2` can then be assigned in `PEtabCondition` (random data are used here
for illustration):

```@example 1
simulation_conditions = [
    PEtabCondition(:cond1, :input2 => rand(4)),
    PEtabCondition(:cond2, :input2 => rand(4))
]
```

The `PEtabODEProblem` can then be built as usual:

```@example 1
measurements = DataFrame(
    simulation_id = ["cond1", "cond1", "cond2", "cond2"],
    obs_id        = ["petab_obs2", "petab_obs1", "petab_obs2", "petab_obs1"],
    time          = [5.0, 10.0, 1.0, 20.0],
    measurement   = [0.7, 0.1, 1.0, 1.5],
)

petab_model = PEtabModel(
    sys, observables, measurements, pest; ml_models = ml_model,
    simulation_conditions = simulation_conditions,
)
petab_prob = PEtabODEProblem(petab_model)
nothing # hide
```
