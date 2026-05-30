# [SciML starter tutorial: Universal Differential Equations (UDEs)](@id sciml_starter)

Hybrid scientific machine learning (SciML) combines mechanistic (ODE) models with machine
learning (ML) models [rackauckas2020universal](@cite). PEtab.jl supports three SciML problem
types, and any combination of them:

1. ML model inside the ODE dynamics. This covers both Universal Differential Equations
   (UDEs), also called grey-box and hybrid neural ODEs
   [brucker2022neural, zou_hybrid2_2024](@cite), as well as neural ODEs (NODEs)
   [chen2018neural](@cite).
2. ML model in the observable formula which links ODE output to measurement data.
3. Pre-simulation ML model, where the ML model is evaluated before simulation to map input
   data (e.g. high-dimensional images) to ODE parameters and/or initial conditions.

This tutorial introduces SciML functionality in PEtab.jl using the UDE case. Other tutorials
in this section cover the remaining cases. Familiarity with setting up a PEtab problem for
mechanistic models is assumed; see the [PEtab.jl starting tutorial](@ref tutorial).

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

Where $\mathrm{NN}$ is a feed-forward neural network with input $Y$.

To estimate model parameters, measurements of both $X$ and $Y$ are assumed. The goal of this
tutorial is to set up a PEtab parameter estimation problem and then estimate both the
mechanistic parameter `d` and the parameters of `NN`.

## Creating a PEtab SciML problem

A PEtab SciML parameter estimation problem (`PEtabODEProblem`) is created largely the same
way as for a mechanistic model. The difference is that one or more Lux.jl neural networks
are defined and embedded into the problem.

### Defining the ML model

PEtab.jl only supports [Lux.jl](https://lux.csail.mit.edu/stable/) ML models. To be
compatible, the Lux model must define a set of layers with unique identifiers together with
a forward pass. This is most easily done using `Lux.Chain`:

```@example 1
using Lux
lux_model = Lux.Chain(
    Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 1, Lux.softplus, use_bias = false),
)
```

Here, `softplus` activation function is used to keep the output positive.

### Defining the dynamic UDE model

An UDE dynamic model is most easily created using ModelingToolkitNeuralNets. The first step
is to create a symbolic neural network from the Lux model:

```@example 1
using ModelingToolkitNeuralNets
@SymbolicNeuralNetwork NN, theta = lux_model
nothing # hide
```

Here `theta` is the network parameter vector, and `NN` is the neural network which can be
embedded into a ModelingToolkit model:

```@example 1
using ModelingToolkitBase
using ModelingToolkitBase: t_nounits as t, D_nounits as D

@parameters d
@variables X(t)=2.0 Y(t)=0.1
eqs = [
    D(X) ~ NN([Y], theta)[1] - d * X
    D(Y) ~ X - d * Y
]
@mtkcompile sys_ude = System(eqs, t)
nothing # hide
```

Alternatively, the UDE can be formulated as a Catalyst `ReactionSystem`:

```@example 1
using Catalyst
t = Catalyst.default_t()

# Scalar NN rate depending on Y.
A(z) = NN([z], theta)
rn_ude = @reaction_network begin
    @species begin
        X(t) = 2.0
        Y(t) = 0.1
    end
    @parameters d
    # Reactions
    $A(Y)[1], 0 --> X
    d, X --> 0
    1.0, X --> X + Y
    d, Y --> 0
end
nothing # hide
```

When embedding a neural network into a `ReactionSystem`, it must be introduced via a
symbolic function (here `A(z)`) and can only be used in the parameter position of a
reaction.

For more on symbolic UDE creation, see the
[ModelingToolkitNeuralNets documentation](https://docs.sciml.ai/ModelingToolkitNeuralNets/stable/).

### Defining ML parameters to estimate

For each ML-model, a `PEtabMLParameter` must be declared to specify whether its parameters
are estimated. Thereby, when specifying the parameters-to-estimate vector for a SciML
problem, both mechanistic parameters (`PEtabParameter`) and ML parameters
(`PEtabMLParameter`) must be included in the parameter vector:

```@example 1
using PEtab
pest = [
    PEtabMLParameter(:theta),
    PEtabParameter(:d; scale = :log10)
]
```

A `PEtabMLParameter` can optionally be given an initial value, priors, and be fixed to
specific value. Note, parameter `d` is estimated on `log10` scale to enforce positivity.

### Measurements and observables

Measurements and observables are defined as for a mechanistic `PEtabODEProblem`. For the
running example, we here use simulated data:

```@example 1
using OrdinaryDiffEqTsit5, DataFrames, StableRNGs
rng = StableRNGs.StableRNG(2) # for reproducibility

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

data_X = sol[1, :] .+ randn(rng, length(sol.t)) .* 0.5
data_Y = sol[2, :] .+ randn(rng, length(sol.t)) .* 0.7
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

Given the dynamic model (`sys_ude` or `rn_ude`), measurements, observables, and parameters
to estimate, a `PEtabModel` is created the same way as for a mechanistic problem:

```@example 1
# Via ReactionSystem
model_ude = PEtabModel(rn_ude, observables, measurements, pest)
# Via ODESystem
model_ude = PEtabModel(sys_ude, observables, measurements, pest)
nothing # hide
```

Given a `PEtabModel`, a `PEtabODEProblem` can then be constructed:

```@example 1
petab_prob = PEtabODEProblem(model_ude)
describe(petab_prob)
```

As seen from the problem summary, the problem includes both mechanistic and ML parameters to
estimate. It is further seen the non-stiff ODE solver `Tsit5()` is used and gradients are
computed with ForwardDiff. These options can be changed, and the same `PEtabODEProblem`
options are supported as for mechanistic models.

## [Parameter estimation (model training)](@id UDE_training)

SciML problems can be trained using the same approaches as mechanistic models (e.g.
multi-start local optimization with a BFGS-based method via [`calibrate`](@ref)). In
practice, SciML training often benefits from optimizers commonly used for ML
[philipps2025current](@cite), such as the `Adam` optimizer [kingma2014adam](@cite). `Adam`,
and other common ML optimizers are available in
[Optimisers.jl](https://fluxml.ai/Optimisers.jl/stable/).

Like most optimizers, `Adam` requires a start guess which can be generated with:

```@example 1
using StableRNGs
rng = StableRNGs.StableRNG(42) # for reproducibility
x0 = get_startguesses(rng, petab_prob, 1)
```

The start guess includes both mechanistic parameters (e.g. `x0.d`) and ML parameters (e.g.
`x0.theta`). By default, `get_startguesses` initializes ML parameters using the initializers
in the Lux model. While not needed here, training often improves by initializing ML
parameters to smaller values [kidger2022neural](@cite).

With a start guess available, `calibrate` can train with any optimizer from Optimisers.jl.
For example, to train for 7500 epochs/iterations with `Adam`:

```@example 1
using Optimisers
learning_rate = 1e-3
res = calibrate(
    petab_prob, x0, Optimisers.Adam(learning_rate);
    options = OptimisersOptions(iterations = 7500),
)
```

Both [`calibrate`](@ref) and [`calibrate_multistart`](@ref) run the provided update rule for
the number of iterations specified in `OptimisersOptions`. For more fine-grained control
(e.g. learning rate schedules and early stopping), a custom training loop must be
implemented.

Such custom loop can be written leveraging that the objective and its gradient can be
computed with `petab_prob.nllh(x)` and `petab_prob.grad(x)`, respectively. For example, to
train for 7500 epochs/iterations with `Adam`:

```@example 1
using Optimisers
global x # hide
global state # hide
n_epochs = 7500
x = deepcopy(x0)
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

As one of many possible downstream analyses, we can compare the learned neural network to
the true function it aims to approximate. In this example, the approximation is quite good:

```@example 1
# True function to learn
true_func(y) = 1.1 * (y^3) / (2^3 + y^3)

# Extract the fitted neural network and parameters from the fitted ODEProblem
ode_problem_fitted, _ = get_odeproblem(x, petab_prob)
fitted_NN = ode_problem_fitted.ps[NN]
fitted_theta = ode_problem_fitted.ps[theta]
fitted_func(y) = fitted_NN([y], fitted_theta)[1]

# Plot true vs fitted
plot(true_func, 0.0, 5.0; label = "True function")
plot!(fitted_func, 0.0, 5.0; label = "Fitted function", linestyle = :dash)
```

The above training loop above is intentionally minimal. In practice, Optimisers.jl and
related packages can be used to add learning-rate schedules, gradient clipping, early
stopping, and logging. It is also worth noting that plain Adam is often inefficient for
SciML problems, and to address this PEtab.jl supports more effective strategies such as
curriculum training, multiple shooting, and combinations thereof (see below).

## Next steps

This tutorial showed how to define a `PEtabODEProblem` for a SciML problem. For all
available options when building SciML problems (e.g. `PEtabMLParameter` settings), see the
[API](@ref API) documentation. For additional features, including SciML problem types beyond
UDEs, see the following tutorials:

- [Extended UDE tutorial](@ref ude_extended): Define the UDE model as an `ODEProblem` for
  more control over the problem setup.
- [ML models in observables](@ref observable_ml): define an ML model in the observable
  formula of a `PEtabObservable` (e.g. to correct model misspecification).
- [Pre-simulation ML models](@ref pre_simulate_ml): define ML models that map input data
  (e.g. high-dimensional images) to ODE parameters or initial conditions prior to model
  simulation.
- [Importing PEtab SciML](@ref import_petab_scimlproblem): import problems in the
  PEtab-SciML standard format.

In addition, this tutorial showed how to train a UDE via a simple Adam training loop. More
efficient training strategies are supported via
[PEtabTraining.jl](https://github.com/sebapersson/PEtabTraining.jl) (e.g. curriculum
learning and multiple shooting); see [SciML training strategies](@ref sciml_training).

Lastly, as for mechanistic models, `PEtabODEProblem` has many configurable options for SciML
problems. A discussion of defaults and recommendations is available in [Default
PEtabODEProblem options](@ref default_options).

## Copy pasteable example

```@example 2
using Catalyst, ComponentArrays, Lux, ModelingToolkitBase,
    ModelingToolkitNeuralNets, PEtab
using ModelingToolkitBase: t_nounits as t, D_nounits as D

# MLModel
lux_model = Lux.Chain(
    Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 1, Lux.softplus, use_bias = false),
)

# UDE-system
@SymbolicNeuralNetwork NN, theta = lux_model
@parameters d
@variables X(t)=2.0 Y(t)=0.1
eqs = [
    # Equations
    D(X) ~ NN([Y], theta)[1] - d * X
    D(Y) ~ X - d * Y
]
@mtkcompile sys_ude = System(eqs, t)

# UDE-system via ReactionSystem
using Catalyst

# Scalar NN rate depending on Y.
A(z) = NN([z], theta)
rn_ude = @reaction_network begin
    @species begin
        X(t) = 2.0
        Y(t) = 0.1
    end
    @parameters d

    $A(Y)[1], 0 --> X
    d, X --> 0
    1.0, X --> X + Y
    d, Y --> 0
end

# Parameters to estimate
pest = [
    PEtabMLParameter(:theta),
    PEtabParameter(:d; scale = :log10)
]

# Observables linking model output to data
observables = [
    PEtabObservable(:obs_X, :X, 0.5),
    PEtabObservable(:obs_Y, :Y, 0.7),
]

# Simulate data
using OrdinaryDiffEqTsit5, DataFrames, StableRNGs
rng = StableRNGs.StableRNG(2)
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
data_X = sol[1, :] .+ randn(rng, length(sol.t)) .* 0.5
data_Y = sol[2, :] .+ randn(rng, length(sol.t)) .* 0.7
df1 = DataFrame(obs_id = "obs_X", time = sol.t, measurement = data_X)
df2 = DataFrame(obs_id = "obs_Y", time = sol.t, measurement = data_Y)
measurements = vcat(df1, df2)

# Create parameter estimation problem
model_ude = PEtabModel(sys_ude, observables, measurements, pest)
petab_prob = PEtabODEProblem(model_ude)

# Get random start-guess for model training
rng = StableRNG(42) # for reproducibility
x0 = get_startguesses(rng, petab_prob, 1)

# Simple Adam training loop
using Optimisers, Plots
global x # hide
global state # hide
n_epochs = 7500
x = deepcopy(x0)
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
plot(x, petab_prob)

# True function to learn
true_func(y) = 1.1 * (y^3) / (2^3 + y^3)
# Fitted neural network
ode_problem_fitted, _ = get_odeproblem(x, petab_prob)
fitted_NN = ode_problem_fitted.ps[NN]
fitted_theta = ode_problem_fitted.ps[theta]
fitted_func(y) = fitted_NN(y, fitted_theta)[1]
# Plots the true and fitted functions
plot(true_func, 0.0, 5.0; lw=8, label="True function")
plot!(fitted_func, 0.0, 5.0; lw=6, label="Fitted function", linestyle=:dash)
nothing # hide
```

## References

```@bibliography
Pages = ["starting_tutorial.md"]
Canonical = false
```
