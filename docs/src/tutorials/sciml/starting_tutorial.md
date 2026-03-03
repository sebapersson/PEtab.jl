# [SciML starter tutorial: Universal Differential Equations (UDEs)](@id sciml_starter)

Hybrid scientific machine learning (SciML) combines mechanistic (ODE) models with machine
learning (ML) models [rackauckas2020universal](@cite). PEtab.jl supports three SciML problem
types, and any combination of them:

1. ML model inside the ODE dynamics. This covers both Universal Differential Equations
   (UDEs) and Neural ODEs.
2. ML model in the observable formula which links ODE output to measurement data.
3. Pre-simulation ML model, where the ML model is evaluated before simulation to map input
   data (e.g. high-dimensional images) to ODE parameters and/or initial conditions.

This tutorial introduces SciML functionality in PEtab.jl using the UDE case. Other tutorials
in this section cover the remaining cases. Familiarity with setting up a PEtab problem for
mechanistic models is assumed; covered in the [PEtab.jl starting tutorial](@ref tutorial).

While this tutorial focuses on a simple use-case, note that SciML support is implemented on
top of the functionality for mechanistic models. Thus, features for mechanistic models (e.g.
simulation conditions and events) are also supported for SciML problems.

## Input problem

As a running example, we use the following two-state ODE model:

```math
\begin{align*}
\frac{\mathrm{d}X}{\mathrm{d}t} &= \frac{vY^n}{Y^n + K^n} - dX \\
\frac{\mathrm{d}Y}{\mathrm{d}t} &= X - dY
\end{align*}
```

with initial conditions

```math
X(t_0) = 2.0, \quad Y(t_0) = 0.1.
```

To turn this into a UDE [rackauckas2020universal](@cite), we assume the production term for
$X$ is unknown and replace it by a neural network:

```math
\begin{align*}
\frac{\mathrm{d}X}{\mathrm{d}t} &= \mathrm{NN}_1(Y) - dX \\
\frac{\mathrm{d}Y}{\mathrm{d}t} &= X - dY
\end{align*}
```

Here, $\mathrm{NN}_1$ is a feed-forward neural network with input $Y$.

To estimate model parameters, measurements of both $X$ and $Y$ are assumed. The goal of this
tutorial is to set up a PEtab parameter estimation problem and then estimate both the
mechanistic parameter `d` and the parameters of `NN1`.

## Creating a PEtab SciML problem

A PEtab SciML parameter estimation problem (`PEtabODEProblem`) is created largely the same
way as for a mechanistic model. The main difference is that ML models are provided by (1)
defining one or more Lux.jl neural networks and (2) wrapping them as `MLModel`s to specify
how they interact with the ODE model.

### Defining the ML model

PEtab.jl only supports [Lux.jl](https://lux.csail.mit.edu/stable/) ML models. To be
compatible, the Lux model must define a set of layers with unique identifiers together with
a forward pass. This is often most easily done using `Lux.Chain`:

```@example 1
using Lux
lux_model1 = Lux.Chain(
    Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 1, Lux.softplus, use_bias = false),
)
```

Here, `softplus` is used to keep the output positive. For more control over the forward
pass, the model can also be defined with `@compact`:

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

Regardless of how the Lux model is defined, it must be wrapped as an `MLModel` and given a
unique ID:

```@example 1
using PEtab
ml_model = MLModel(:net1, lux_model1, false)
```

The third argument specifies whether the model is evaluated pre-simulation. In this
tutorial, the ML model is evaluated during simulation (it enters the ODE dynamics), so
`false` is used. Each Lux model must be wrapped as an `MLModel` so PEtab.jl can keep track
of how the ML model interacts with the ODE model.

### Defining the dynamic (UDE) model

A UDE is created by embedding one or more `MLModel`s into the ODE dynamics. Currently, UDE
models must be provided as an `ODEProblem` (integration with ModelingToolkitNeuralNets is in
progress). The first step is to define the ODE right-hand side. Compared to a standard
in-place ODE function, an extra `ml_models` argument is provided. It stores the declared ML
models and can be indexed by their IDs:

```@example 1
function ude_f!(du, u, p, t, ml_models)
    X, Y = u
    net1 = ml_models[:net1]
    nn, _ = net1.lux_model([Y], p.net1, net1.st)
    du[1] = nn[1][1] - p.d * X
    du[2] = X - p.d * Y
end
nothing # hide
```

The parameter vector `p` is expected to be a `ComponentVector`, so mechanistic parameters
are accessed as `p.id` (e.g. `p.d`) and ML parameters as `p.ml_id` (e.g. `p.net1`).

Given the right-hand side, the `ODEProblem` can be constructed using the
[`UDEProblem`](@ref) helper function:

```@example 1
using ComponentArrays
p_mechanistic = ComponentArray(d = 1.0)
u0 = ComponentArray(X = 2.0, Y = 0.1)
ude_prob = UDEProblem(ude_f!, u0, (0.0, 10.0), p_mechanistic, ml_model)
nothing # hide
```

When the dynamics are provided as an `ODEProblem`/`UDEProblem`, mechanistic parameters
(`p_mechanistic`) and initial values (`u0`) must be `ComponentArray`s so PEtab.jl can track
parameter and state IDs.

### Defining ML parameters to estimate

A `PEtabMLParameter` must be declared to specify whether its parameters are estimated for
each `MLModel`. Thereby, when specifying the parameters-to-estimate vector for a SciML
problem, both mechanistic parameters (`PEtabParameter`) and ML parameters
(`PEtabMLParameter`) must be included in the parameter vector:

```@example 1
pest = [
    PEtabMLParameter(:net1),
    PEtabParameter(:d; scale = :log10)
]
```

A `PEtabMLParameter` can optionally be given an initial value, priors, and be fixed
settings. Note parameter `d` is estimated on `log10` scale to enforce positivity.

### Measurements and observables

Measurements and observables are defined as for a mechanistic `PEtabODEProblem`. For the
running example, we here use simulated data:

```@example 1
using OrdinaryDiffEqTsit5, DataFrames, Random
Random.seed!(123) # hide

function f_true!(du, u, p, t)
    X, Y = u
    v, K, n, d = p
    du[1] = (v * Y^n) / (Y^n + K^n) - d * X
    du[2] = X - d * Y
end

u0_true = [2.0, 0.1]
p_true = [1.1, 2.0, 3.0, 0.5]
tend = 44.0
ode_true = ODEProblem(f_true!, u0_true, (0.0, tend), p_true)
sol = solve(
    ode_true, Tsit5(); abstol = 1e-8, reltol = 1e-8, saveat = 0:2:tend
)

data_X = sol[1, :] .+ randn(length(sol.t)) .* 0.5
data_Y = sol[2, :] .+ randn(length(sol.t)) .* 0.7
df1 = DataFrame(obs_id = "obs_X", time = sol.t, measurement = data_X)
df2 = DataFrame(obs_id = "obs_Y", time = sol.t, measurement = data_Y)
measurements = vcat(df1, df2)
nothing # hide
```

Since both `X` and `Y` are assumed measured, the observables are:

```@example 1
observables = [
    PEtabObservable(:obs_X, :X, 0.5),
    PEtabObservable(:obs_Y, :Y, 0.7),
]
```

### Bringing it all together

Given the dynamic model (`ude_prob`), ML models, measurements, observables, and parameters
to estimate, a `PEtabModel` is created largely the same way as for a mechanistic problem.
The difference is that the ML models must be provided:

```@example 1
model_ude = PEtabModel(
    ude_prob, observables, measurements, pest; ml_models = ml_model
)
nothing # hide
```

Given a `PEtabModel`, a `PEtabODEProblem` can then be constructed:

```@example 1
petab_prob = PEtabODEProblem(model_ude; odesolver = ODESolver(Tsit5()))
describe(petab_prob)
```

As seen from the problem summary, the problem includes both mechanistic and ML parameters to
estimate. It is further seen the non-stiff ODE solver `Tsit5()` is used and gradients are
computed with ForwardDiff. These options can be changed, and the same `PEtabODEProblem`
options are supported as for mechanistic models.

## Parameter estimation (model training)

SciML problems can be trained using the same approaches as mechanistic models (e.g.
multi-start local optimization with a BFGS-based method via [`calibrate`](@ref)). In
practice, SciML training often benefits from optimizers commonly used for ML
[philipps2025current](@cite), such as the `Adam` optimizer [kingma2014adam](@cite).

Here, we will set up a training loop with `Adam` from
[Optimisers.jl](https://fluxml.ai/Optimisers.jl/stable/). Like most optimizers, `Adam`
requires a start guess, which can be generated with:

```@example 1
using StableRNGs
rng = StableRNG(1) # for reproducibility
x0 = get_startguesses(rng, petab_prob, 1)
```

The start guess includes both mechanistic parameters (e.g. `x0.d`) and ML parameters (e.g.
`x0.net1`). By default, `get_startguesses` initializes ML parameters with the initializers
in the Lux model. While not needed here, SciML training often improve by initializing ML
parameters to smaller values [kidger2022neural](@cite).

With a start guess available, a training loop can be written leveraging that the objective
function and its gradient can be computed with `petab_prob.nllh(x)` and `petab_prob.grad(x)`
respectively. For example, to train for 5000 epochs with `Adam`:

```@example 1
using OrdinaryDiffEqBDF # hide
using Optimisers
global x # hide
global state # hide
n_epochs = 5000
x = deepcopy(x0)

petab_prob = PEtabODEProblem(model_ude; odesolver = ODESolver(QNDF())) # hide

learning_rate = 1e-3
state = Optimisers.setup(Adam(learning_rate), x)
for epoch in 1:n_epochs
    global x # hide
    global state # hide
    g = petab_prob.grad(x)
    state, x = Optimisers.update(state, x, g)

    # Stop if the objective cannot be evaluated (e.g. simulation failure)
    if !isfinite(petab_prob.nllh(x))
        break
    end
end
nothing # hide
```

From plotting the fitted model, we see the model fits the data well:

```@example 1
using Plots
default(left_margin=12.5Plots.Measures.mm, bottom_margin=12.5Plots.Measures.mm, size = (600*1.25, 400 * 1.25), palette = ["#CC79A7", "#009E73", "#0072B2", "#D55E00", "#999999", "#E69F00", "#56B4E9", "#F0E442"], linewidth=4.0) # hide
plot(x, petab_prob)
```

The training loop above is intentionally minimal. In practice, Optimisers.jl and related
packages can be used to build training loops with learning-rate schedules, gradient
clipping, early stopping, logging, etc.

Plain Adam is often inefficient for SciML problems, and it is prudent to run parameter
estimation from multiple start guesses to reduce sensitivity to local minima. More efficient
strategies include curriculum training, multiple shooting, or combinations thereof, all of
which are supported by PEtab.jl.

## Next steps

This tutorial showed how to define a `PEtabODEProblem` for a SciML problem. For all
available options when building SciML problems (e.g. `PEtabMLParameter` settings), see the
[API](@ref API) documentation. For additional features, including SciML problem types beyond
UDEs, see the following tutorials:

- [ML models in observables](@ref observable_ml): define an ML model in the observable
  formula of a `PEtabObservable` (e.g. to correct model misspecification).
- [Pre-simulation ML models](@ref pre_simulate_ml): define ML models that map input data
  (e.g. high-dimensional images) to ODE parameters or initial conditions prior to model
  simulation.
- [Importing PEtab SciML](@ref import_petab_scimlproblem): import problems in the
  PEtab-SciML standard format.

In addition, this tutorial showed how to train a UDE via a simple training loop. PEtab.jl
supports several more efficient SciML training strategies (e.g. curriculum learning and
multiple shooting). For more on improving model training see ADD!

Lastly, as for mechanistic models, `PEtabODEProblem` has many configurable options for SciML
problems. A discussion of defaults and recommendations is available in [Default
PEtabODEProblem options](@ref default_options).

## References

```@bibliography
Pages = ["starting_tutorial.md"]
Canonical = false
```
