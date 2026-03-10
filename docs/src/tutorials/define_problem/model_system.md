# [Supported model systems](@id model_systems)

The dynamic ODE model in a `PEtabODEProblem` can be provided in three ways: (1) as a
Catalyst `ReactionSystem`, (2) as a ModelingToolkitBase `ODESystem`, and (3) as an
OrdinaryDiffEq `ODEProblem`.This page summarizes options and restrictions for each model
system. As a running example, the Michaelis–Menten model from the [starting tutorial](@ref
tutorial) is used:

```@example 1
using DataFrames, PEtab

pest = [
    PEtabParameter(:c1; value = 1.0),
    PEtabParameter(:c2; value = 2.0),
    PEtabParameter(:S0; value = 5.0),
    PEtabParameter(:sigma),
]

measurements = DataFrame(
    obs_id = ["obs_p", "obs_sum", "obs_p", "obs_sum"],
    time = [1.0, 10.0, 1.0, 20.0],
    measurement = [0.7, 0.1, 1.0, 1.5],
)
nothing # hide
```

## Shared model system options

For all model systems, PEtab.jl reads default parameter/state values defined in the model
system. If a default value is not specified for a state or parameter that is not estimated,
it defaults to zero.

Additionally, default values for a subset of model variables can also be provided via
`speciemap`/`parametermap`, which take precedence over model system defaults. For example,
maps for states/species and parameters can be built as:

```@example 1
using Catalyst
# Option 1: keys as `Num`
t = Catalyst.default_t()
@species S(t) E(t) SE(t) P(t)
@parameters S0 c3
speciemap = [S => S0, E => 1.0, SE => 0.0, P => 0.0]
parametermap = [c3 => 3.0]

# Option 2: keys as `Symbol`
speciemap_sym = [:S => :S0, :E => 1.0, :SE => 0.0, :P => 0.0]
parametermap_sym = [:c3 => 3.0]
nothing # hide
```

### Restrictions

For all model systems vector-valued variables are currently not supported. For example, for
a `ReactionSystem` the following holds:

```julia
# not allowed
@parameters k[1:2]
# allowed
@parameters k1 k2
```

The only exception is ML parameters when using UDE/Neural ODE models.

## Model as a `ReactionSystem`

A [Catalyst.jl](https://github.com/SciML/Catalyst.jl) `ReactionSystem` is a valid dynamic
model which internally is converted to an `ODESystem`. If observables are defined with
`@observables` in the reaction network, they can be referenced directly in
`PEtabObservable`. For example:

```@example 1
using Catalyst

rn = @reaction_network begin
    @parameters S0 c3 = 1.0
    @species begin
        S(t) = S0
        E(t) = 1.0
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
observables = [
    PEtabObservable(:petab_obs1, :obs1, 3.0),
    PEtabObservable(:petab_obs2, :obs2, sigma),
]

model = PEtabModel(rn, observables, measurements, pest)
petab_prob = PEtabODEProblem(model)
nothing # hide
```

To override defaults via maps:

```@example 1
model = PEtabModel(
    rn, observables, measurements, pest; speciemap = speciemap,
    parametermap = parametermap,
)
petab_prob = PEtabODEProblem(model)
```

## Model as an `ODESystem`

A [ModelingToolkitBase.jl](https://github.com/SciML/ModelingToolkit.jl) `ODESystem` is a
valid dynamic model. Observables defined as algebraic equations (e.g. `obs1 ~ ...`) can be
referenced by ID in `PEtabObservable`. For example:

```@example 1
using ModelingToolkitBase
using ModelingToolkitBase: t_nounits as t, D_nounits as D

ps = @parameters S0 c1 c2 c3 = 3.0
sps = @variables S(t) = S0 E(t) = 50.0 SE(t) = 0.0 P(t) = 0.0 obs1(t) obs2(t)
eqs = [
    # Dynamics
    D(S) ~ -c1 * S * E + c2 * SE
    D(E) ~ -c1 * S * E + c2 * SE + c3 * SE
    D(SE) ~  c1 * S * E - c2 * SE - c3 * SE
    D(P) ~  c3 * SE
    # observables
    obs1 ~ S + E
    obs2 ~ P
]

@named sys_model = System(eqs, t, sps, ps)
sys = mtkcompile(sys_model)

@parameters sigma
observables = [
    PEtabObservable(:petab_obs1, :obs1, 3.0),
    PEtabObservable(:petab_obs2, :obs2, sigma),
]

model = PEtabModel(sys, observables, measurements, pest)
petab_prob = PEtabODEProblem(model)
nothing # hide
```

To override defaults via maps:

```@example 1
model = PEtabModel(
    sys, observables, measurements, pest; speciemap = speciemap,
    parametermap = parametermap,
)
petab_prob = PEtabODEProblem(model)
```

## Model as an `ODEProblem`

An `ODEProblem` is a valid dynamic model and is useful when, for example, it is not possible
to define the model as a symbolic system. The main requirement is that PEtab.jl can identify
state and parameter names. Therefore, `u0` and `p` must be provided as named containers
(e.g. `ComponentArray` or `NamedTuple`). Moreover, since observable formulas cannot be
encoded in an `ODEProblem`, they must be defined in `PEtabObservable`. For example:

```@example 1
using ComponentArrays, OrdinaryDiffEqRosenbrock, SimpleUnPack

function ode_f!(du, u, p, t)
    S, E, SE, P = u
    SimpleUnPack.@unpack c1, c2, c3 = p

    du[1] = -c1 * S * E + c2 * SE
    du[2] = -c1 * S * E + c2 * SE + c3 * SE
    du[3] =  c1 * S * E - c2 * SE - c3 * SE
    du[4] =  c3 * SE
end

p = ComponentArray(c1 = 1.0, c2 = 2.0, c3 = 3.0)
u0 = (S = :S0, E = 50.0, SE = 0.0, P = 0.0)
ode_prob = ODEProblem(ode_f!, u0, (0.0, 10.0), p)

@parameters sigma
observables = [
    PEtabObservable(:petab_obs1, "S + E", 3.0),
    PEtabObservable(:petab_obs2, "P", sigma),
]

model = PEtabModel(ode_prob, observables, measurements, pest)
petab_prob = PEtabODEProblem(model)
nothing # hide
```

As we assign a parameter to an initial value for state `S`, initial values need to be
provided as a `NamedTuple`.

::: warning Parameter access in `ODEProblem`

PEtab.jl may reorder parameters internally when processing an `ODEProblem`. Parameter access
should therefore use names (e.g. `p.c3` or `p[:c3]`), or `@unpack`, rather than relying on
positional indexing.

:::

To override defaults via maps:

```@example 1
model = PEtabModel(
    sys, observables, measurements, pest; speciemap = speciemap,
    parametermap = parametermap,
)
petab_prob = PEtabODEProblem(model)
```

## Next steps

More on defining `ReactionSystem` and `ODESystem` models can be found in the
[Catalyst.jl](https://docs.sciml.ai/Catalyst/stable/) and
[ModelingToolkitBase.jl](https://docs.sciml.ai/ModelingToolkit/dev/) documentation
respectively.
