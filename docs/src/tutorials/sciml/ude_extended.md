# [Extended UDE tutorial](@id ude_extended)

This tutorial shows how to set up an Universally Differential Equation (UDE) model as an
`ODEProblem` for more fine-grained control over the problem setup. Defining the dynamics as
an `ODEProblem` instead of via ModelingToolkitNeuralNets gives more control over the UDE
right-hand side (e.g. ML models with multiple input arguments) and is useful for cases not
yet handled by ModelingToolkitNeuralNets.

This tutorial assumes familiarity with the [SciML starter tutorial](@ref sciml_starter). As
a working example, we use the same two-state model as in the [SciML starter tutorial](@ref
sciml_starter):

```math
\begin{align*}
\frac{\mathrm{d}X}{\mathrm{d}t} &= \frac{vY^n}{Y^n + K^n} - dX, \quad X(t_0) = 2.0 \\
\frac{\mathrm{d}Y}{\mathrm{d}t} &= X - dY, \quad Y(t_0) = 0.1
\end{align*}
```

To form a UDE, the production term for $X$ is replaced by a neural network `NN`:

```math
\begin{align*}
\frac{\mathrm{d}X}{\mathrm{d}t} &= \mathrm{NN}_1(Y) - dX \\
\frac{\mathrm{d}Y}{\mathrm{d}t} &= X - dY
\end{align*}
```

Measurements of both $X$ and $Y$ are assumed. The goal is to set up a PEtab problem that
estimates the mechanistic parameter `d` and the parameters of `NN`.

## Creating a PEtab SciML problem (ODEProblem route)

The overall workflow is the same as for a mechanistic model. The main difference is that ML
models are provided by (1) defining one or more Lux.jl neural networks and (2) wrapping them
as `MLModel`s.

### Defining the ML model

The [Lux.jl](https://lux.csail.mit.edu/stable/) model can be defined as a `Lux.Chain`:

```@example 1
using Lux
lux_model1 = Lux.Chain(
    Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 1, Lux.softplus, use_bias = false),
)
nothing # hide
```

For more control over the forward pass, the model can also be defined with `@compact`:

```@example 1
lux_model2 = @compact(
    layer1 = Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
    layer2 = Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
    layer3 = Lux.Dense(3 => 1, Lux.softplus, use_bias = false),
) do x # forward pass
    h = layer1(x)
    h = layer2(h)
    out = layer3(h)
    @return out
end
nothing # hide
```

Regardless of how the Lux model is defined, in the `ODEProblem` case it must be wrapped as
an `MLModel` and given a unique ID:

```@example 1
using PEtab
ml_model = MLModel(:net1, lux_model1, false)
nothing # hide
```

The third argument specifies whether the model is evaluated pre-simulation. As the ML model
is evaluated during simulation here (it enters the ODE dynamics), `false` is used.

### Defining the dynamic UDE model

To define a UDE model as an `ODEProblem`, the UDE right-hand side takes an additional
`ml_models` argument compared to a standard in-place ODE function. It stores the declared ML
models and can be indexed by their IDs:

```@example 1
function ude_f!(du, u, p, t, ml_models)
    X, Y = u
    net1 = ml_models[:net1]
    nn, _ = net1.lux_model([Y], p.net1, net1.st)

    du[1] = nn[1][1] - p.d * X
    du[2] = X - p.d * Y
    return nothing
end
nothing # hide
```

The parameter vector `p` must be a `ComponentVector`, so mechanistic parameters are accessed
as `p.<id>` (e.g. `p.d`) and ML parameters as `p.<ml_id>` (e.g. `p.net1`). Given the UDE
right-hand side, the `ODEProblem` can be constructed using the [`UDEProblem`](@ref) helper
function:

```@example 1
using ComponentVectors
p_mechanistic = ComponentVector(d = 1.0)
u0 = ComponentVector(X = 2.0, Y = 0.1)
ude_prob = UDEProblem(ude_f!, u0, (0.0, 10.0), p_mechanistic, ml_model)
nothing # hide
```

More on defining the model as an `ODEProblem`/`UDEProblem`, can be found in [Supported model
systems](@ref model_systems).

### Defining parameters to estimate

When an ML model is provided as an `MLModel`, its parameters are included for estimation by
adding a `PEtabMLParameter` that refers to the model by its ID:

```@example 1
pest = [
    PEtabMLParameter(:net1),
    PEtabParameter(:d; scale = :log10)
]
nothing # hide
```

### Measurements and observables

Measurements and observables are defined as for a mechanistic PEtab problem. For
illustration, synthetic data are generated from the mechanistic model and stored in a
`DataFrame`:

```@example 1
using OrdinaryDiffEqTsit5, DataFrames, Random
import Random; Random.seed!(123) # hide

function f_true!(du, u, p, t)
    X, Y = u
    v, K, n, d = p
    du[1] = (v * Y^n) / (Y^n + K^n) - d * X
    du[2] = X - d * Y
end

u0_true = [2.0, 0.1]
p_true = [1.0, 2.0, 3.0, 0.5]
tend = 44.0
ode_true = ODEProblem(f_true!, u0_true, (0.0, tend), p_true)
sol = solve(ode_true, Tsit5(); abstol = 1e-8, reltol = 1e-8, saveat = 0:2:tend)

data_X = sol[1, :] .+ randn(length(sol.t)) .* 0.5
data_Y = sol[2, :] .+ randn(length(sol.t)) .* 0.7
df1 = DataFrame(obs_id = "obs_X", time = sol.t, measurement = data_X)
df2 = DataFrame(obs_id = "obs_Y", time = sol.t, measurement = data_Y)
measurements = vcat(df1, df2)
nothing # hide
```

```@example 1
observables = [
    PEtabObservable(:obs_X, :X, 0.5),
    PEtabObservable(:obs_Y, :Y, 0.7),
]
nothing # hide
```

### Bringing it all together

A `PEtabModel` is created as usual, with the UDE dynamics (`ude_prob`), observables,
measurements, and parameters to estimate. The ML model is provided via `ml_models`:

```@example 1
model_ude = PEtabModel(
    ude_prob, observables, measurements, pest; ml_models = ml_model
)
petab_prob = PEtabODEProblem(model_ude)
```

The resulting `PEtabODEProblem` can be trained in the same way as in the [SciML starter
tutorial](@ref sciml_starter).
