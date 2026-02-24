# [Observables ML models](@id pre_simulate_ml)

Often a mechanistic model is miss-specifed, or the obsevable mapping mapping model output
to measurements, is incompletely understood. This can be addressed by including a neural
network (universal approximator) in the observable formula in the `PEtabObservable`.

This tutorial shows how to include a ML model in the observable formula. It assumes
familiarity with the [SciML starter tutorial](@ref sciml_starter). As a running example, we
use the Michaelis-Menten model from the PEtab.jl [starting tutorial](@ref tutorial):

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
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end

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

## Define an observable ML model

A pre-simulation ML model will appear in the observable formula in the `PEtabObservable`.
This is obtained by first defining the model structure as a Lux.jl model, followed by defining `MLModel` defining the input expression and an output variable which can then
be inserted into the observable formula. FOr instance, lets say the input is given by the
model states `S` and `E`:

```@example 1
lux_model = Lux.Chain(
    Dense(2 => 5, Lux.swish)
    Dense(5 => 1)
)

@variables S, E
ml_model = MLModel(
    :net1, lux_models, false; inputs = [S, E], outputs = [:output1]
)
```

Note, `false` is provided here to denote the ML-model is evaluated during model simulation
(note pre-simulation case). Here it is interpreted as the input being a Vector, where
everytime it is evaluated it takes the model states `S` and `E` as inputs. Here the inputs
are only the states, but general Julia expressions are allowed. The output variable
`output1` can then be embedded into the observable formulas:

```@example 1
@variables P, output1
@parameters sigma
observables = [
    PEtabObservable(:obs_p, P, 3.0)
    PEtabObservable(:obs_sum, output1, sigma)
]
```

Note, when the ML model appear in the observable formula, the formula should be defined
inside the `PEtabObservable` instead of the model system. This allows PEtab.jl to track
where the ML model is located in the problem, allowing for more efficient gradient
computations.

Given this, the rest of the `PEtabODEProblem` is defined as usual:

```@example 1
pest = [
    PEtabParameter(:c1),
    PEtabParameter(:c2),
    PEtabParameter(:S0),
    PEtabParameter(:sigma),
    PEtabMLParameter(:net1) # note, ML parameters
]

measurements = DataFrame(
    obs_id=["obs_p", "obs_sum", "obs_p", "obs_sum"],
    time=[1.0, 10.0, 1.0, 20.0],
    measurement=[0.7, 0.1, 1.0, 1.5]
)

petab_model = PEtabModel(
    sys, observables, measurements, pest; ml_models = ml_model
)
petab_prob = PEtabODEProblem(petab_model)
```

## Condition specific array input

If the ML model us used to correct model miss-specification, it can if additional data
is available be useful to incorperate additional data. For example, this additional data
might not be directly incoprerate into the model (e.g., being image data, or other
non-time-lapse data), but could be useful to help correct model predictions.

In general, the above scenario can be handled by in the  `inputs` have one input argument the forward function being model entites, while a second input argument being a variable assigned via `PEtabCondition`. How to specify inputs via PEtabCondition, as well as multiple
input argument is covered in the [Pre-simulation ML models](@ref pre_simulate_ml) tutorial.
Here, for the observable case similar holds. So for example, lets say that input argument
is `[S, E]`, and we have additional data which can help inform the model that is condition
specific. This can be handled by first having the additional data assigned to the
PEtab variable `input2`:

```@example 1
lux_model = @compact(
    layer1 = Dense(6 => 5), Lux.swish,
    layer2 = Dense(5 => 1)
) do (x1, x2)
    x = cat(x1, x2; dims = 1)
    embed = layer1(x)
    out = layer2(embed)
    @return out
end

ml_model = MLModel(
    :net1, lux_model, true, inputs = ([S, E], [:input2]), outputs = [:output1]
)
nothing # hide
```

The `input2` can then be assigned in `PEtabCondition`, where it can take the shape of
any array compatiable with the ML model forward function, for example, here using
random data:

```@example 1
conditions = [
    PEtabCondition(:cond1, :input2 => [1.0, 2.0, 3.0, 4.0]),
    PEtabCondition(:cond1, :input2 => [4.0, 3.0, 2.0, 1.0])
]
```

The PEtab problem can then be built as usual:

```@example 1


# Condition specific measurements
measurements = DataFrame(
    simulation_id = ["cond1", "cond1", "cond2", "cond2"],
    obs_id        = ["petab_obs2", "petab_obs1", "petab_obs2", "petab_obs1"],
    time          = [5.0, 10.0, 1.0, 20.0],
    measurement   = [0.7, 0.1, 1.0, 1.5],
)

petab_model = PEtabModel(
    sys, observables, measurements, pest; ml_models = ml_model,
    simulation_conditions = simulation_conditions
)
petab_prob = PEtabODEProblem(petab_model)
nothing # hide
```
