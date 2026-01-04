# [Tutorial](@id tutorial)

This starting tutorial shows how to define a parameter-estimation problem in PEtab.jl
(`PEtabODEProblem`) and estimate its parameters.

## Input problem

As a running example, we use the Michaelis-Menten enzyme kinetics model:

```math
\begin{align*}
  S + E &\xrightarrow{c_1} SE \\
  SE &\xrightarrow{c_2} S + E \\
  SE &\xrightarrow{c_3} P + E,
\end{align*}
```

Using mass-action kinetics, this yields the ODE system:

```math
\begin{align*}
    \frac{\mathrm{d}E}{\mathrm{d}t} &= (c_2 + c_3)SE - c_1 S E \\
    \frac{\mathrm{d}P}{\mathrm{d}t} &= c_3 SE \\
    \frac{\mathrm{d}S}{\mathrm{d}t} &= c_2 SE - c_1 S E \\
    \frac{\mathrm{d}SE}{\mathrm{d}t} &= -(c_2 + c_3)SE + c_1 S E
\end{align*}
```

We assume initial conditions for `[S, E, SE, P]`:

```math
S(t_0) = S_0,\quad E(t_0) = 50.0,\quad SE(t_0) = 0.0,\quad P(t_0) = 0.0.
```

and that we have measurements of two observables `S + E` and `P`:

```math
\begin{align*}
    obs_1 &= S + E \\
    obs_2 &= P
\end{align*}
```

We estimate `[c1, c2, S0]` and assume `c3 = 1.0` is known. The tutorial builds the
corresponding `PEtabODEProblem` and estimates these parameters.

## Creating a parameter estimation problem

A PEtab parameter estimation problem (`PEtabODEProblem`) is defined by:

1. **Dynamic model**: A Catalyst `ReactionSystem` or ModelingToolkit `ODESystem`.
2. **Observables**: `PEtabObservable`s that map model states/parameters to measured
   quantities, including noise models (formula + distribution).
3. **Parameters**: `PEtabParameter`s specifying which parameters are estimated (and optional
   priors, bounds, and scale).
4. **Measurements**: Measurement data as a `DataFrame` in the format described below.
5. **Simulation conditions (optional)**: `PEtabCondition`s for measurements collected under
   different experimental conditions, where simulations use different control parameter
   values (see [Simulation conditions tutorial](@ref petab_sim_cond)).

### Defining the dynamic model

The dynamic model can be provided as either a Catalyst `ReactionSystem` or a ModelingToolkit
`ODESystem`. It is recommended to define default parameter values, initial conditions, and
(when possible) observables directly in the model system.

For the Michaelis–Menten problem above, the `ReactionSystem` representation is:

```@example 1
using Catalyst
rn = @reaction_network begin
    @parameters begin
      S0
      c3 = 1.0
    end
    @species begin
      S(t) = S0
      E(t) = 50.0
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
nothing # hide
```

Constant parameters (e.g. `c3`) and parameters used in initial conditions (e.g. `S0`) should
be declared in `@parameters`. Parameters that will be estimated (here `c1`, `c2`, `S0`) do
not need values in the system. Any species or parameters without specified values default to
`0.0`.

Using a `ODESystem`, the model is defined as:

```@example 1
using ModelingToolkit
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
nothing # hide
```

For an `ODESystem`, all parameters and states must be declared in the model definition. Any
parameter/state left without a value (and not estimated) defaults to `0.0`.

### Defining observables

To link model states/parameters to measurement data, each measured quantity needs an
**observable formula** and a **noise/scale formula**. This is specified with
`PEtabObservable`.

Since the model system already defines observables (`obs1`, `obs2` above), we can reference
them by name. For example, assume we measure `obs1 = S + E` with known noise `σ = 3.0`:

```@example 1
petab_obs1 = PEtabObservable(:petab_obs1, :obs1, 3.0)
```

The first argument is the observable id used to link measurement rows in the measurement
table (see below) to this `PEtabObservable`. The second argument (`:obs1`) references the
observable defined in the model system. By default, Normal noise is assumed; other
distributions can be selected via the `distribution` keyword.

The noise level can also be estimated. Assume we measure `obs2 = P` with unknown noise
parameter `sigma`:

```@example 1
@parameters sigma
petab_obs2 = PEtabObservable(:petab_obs2, :obs2, sigma)
```

Finally, all observables should be collected into a `Vector`:

```@example 1
observables = [petab_obs1, petab_obs2]
nothing # hide
```

### Defining parameters to estimate

Parameters to estimate are specified with `PEtabParameter`. For example, to estimate `c1`:

```@example 1
p_c1 = PEtabParameter(:c1)
```

By default, parameters are assigned bounds `[1e-3, 1e3]` and estimated on a `log10` scale.
These defaults typically improve parameter estimation performance performance
[frohlich2022fides](@cite), [raue2013lessons](@cite), [hass2019benchmark](@cite)). Bounds
and scale can be changed via keyword arguments.

Parameters can also be assigned priors. For example, to assign a `LogNormal(1.0, 0.3)` prior
to `c2`:

```@example 1
using Distributions
p_c2 = PEtabParameter(:c2; prior = LogNormal(1.0, 0.3))
```

All parameters to estimate should be defined similarly, and finally collected into a
`Vector`:

```@example 1
p_s0 = PEtabParameter(:S0)
p_sigma = PEtabParameter(:sigma)
pest = [p_c1, p_c2, p_s0, p_sigma]
```

### Measurement data format

Measurement data is provided as a `DataFrame` with the following columns (order does not
matter):

| obs_id (str) | time (float) | measurement (float) |
| ------------ | ------------ | ------------------- |
| id           | val          | val                 |
| ...          | ...          | ...                 |

- `obs_id`: Observable id identifying which `PEtabObservable` the row belongs to.
- `time`: Measurement time point.
- `measurement`: Measured value.

For our working example, using simulated data, a valid measurement table could be:

```@example 1
using OrdinaryDiffEqRosenbrock, DataFrames
# Simulate with 'true' parameters
ps = [:c1 => 1.0, :c2 => 10.0, :c3 => 1.0, :S0 => 100.0]
u0 = [:S => 100.0, :E => 50.0, :SE => 0.0, :P => 0.0]
tspan = (0.0, 10.0)
oprob = ODEProblem(rn, u0, tspan, ps)
sol = solve(oprob, Rodas5P(); saveat = 0:0.5:10.0)

obs1 = (sol[:S] + sol[:E]) .+ randn(length(sol[:E]))
obs2   = sol[:P] .+ randn(length(sol[:P]))

df1 = DataFrame(obs_id = "petab_obs1", time = sol.t, measurement = obs1)
df2   = DataFrame(obs_id = "petab_obs2", time = sol.t, measurement = obs2)
measurements = vcat(df1, df2)
first(measurements, 5) # hide
```

Note that the measurement table follows a tidy/long format [wickham2014tidy](@cite), where
each row is **one** measurement. For repeats at the same time point, add one row per
replicate.

### Bringing it all together

Given a model, observables, parameters to estimate, and measurement data, we can build a
`PEtabODEProblem` for parameter estimation. This is a two-step process: first create a
`PEtabModel`, then a `PEtabODEProblem`.

Regardless of whether the model is a `ReactionSystem` or an `ODESystem`, a `PEtabModel` is
created with:

```@example 1
model_rn = PEtabModel(rn, observables, measurements, pest)
model_sys = PEtabModel(sys, observables, measurements, pest)
nothing # hide
```

From a `PEtabModel`, a `PEtabODEProblem` is created as:

```@example 1
petab_prob = PEtabODEProblem(model_rn)
```

The printed summary shows information such as the number of parameters to estimate, the
default ODE solver, and default gradients/Hessians computation methods computed. These
defaults are tuned for typical biological models (see the [Defaults](@ref default_options)
page) and can be customized when constructing `PEtabODEProblem`.

Next, we estimate the unknown parameters given a `PEtabODEProblem`.

## Parameter estimation

A `PEtabODEProblem` contains everything needed to run parameter estimation with a numerical
optimizer. For convenience, PEtab.jl provides built-in wrappers for several optimizers,
including Optim.jl, Ipopt, Optimization.jl, and Fides.jl. This section shows how to estimate
parameters with Fides.jl from a starting guess `x0`, and how to use multistart optimization
to mitigate local minima.

### Single-start parameter estimation

Numerical optimizers require a starting point `x0` in the parameter order expected by the
`PEtabODEProblem`. A simple option is to use the nominal values from `PEtabParameter` via
`get_x`:

```@example 1
x0 = get_x(petab_prob)
```

`x0` is a `ComponentArray`, so parameters can be accessed by name. Parameters estimated on a
log scale have a prefix such as `log10_`, and values must be set on that scale, e.g. to set
`c1 = 10.0`:

```@example 1
x0.log10_c1 = log10(10.0)
nothing # hide
```

To reduce bias in parameter estimation, it is often best to randomly sample `x0` within the
parameter bounds:

```@example 1
using Random, StableRNGs # hide
rng = StableRNG(123) # hide
x0 = get_startguesses(petab_prob, 1)
x0 = get_startguesses(rng, petab_prob, 1) # hide
nothing # hide
```

Given `x0`, we can estimate parameters. For this small example (4 parameters), we use the
Newton-trust region method from Fides.jl with the Hessian computed by `petab_prob`
(`CustomHessian`):

```@example 1
using Fides
res = calibrate(petab_prob, x0, Fides.CustomHessian())
```

The printout shows parameter estimation statistics, such as the final objective value `fmin`
(which, since PEtab works with likelihoods, corresponds to the negative log-likelihood). The
estimated parameters are available as:

```@example 1
res.xmin
```

Lastly, to evaluate the parameter estimation, it is useful to plot how well the model fits
the data:

```@example 1
using Plots
default(left_margin=12.5Plots.Measures.mm, bottom_margin=12.5Plots.Measures.mm, size = (600*1.25, 400 * 1.25), palette = ["#CC79A7", "#009E73", "#0072B2", "#D55E00", "#999999", "#E69F00", "#56B4E9", "#F0E442"], linewidth=4.0) # hide
plot(res, petab_prob; linewidth = 2.0)
```

ODE parameter estimation problems often have multiple local minima [raue2013lessons](@cite),
so a single start may not find the best solution. Next, we cover how to mitigate this with
multistart optimization.

### Multi-start parameter estimation

In multi-start estimation, the optimizer is run from multiple random starting points to
reduce the risk of converging to a local minimum. This simple global optimization strategy
has proved effective for ODE models [raue2013lessons, villaverde2019benchmarking](@cite).

PEtab.jl provides `calibrate_multistart`, which generates starting points (by default using
Latin hypercube sampling within bounds) and runs the optimizer. For example, to run `n = 50`
starts with Fides (here using a `BFGS` Hessian approximation):

```@example 1
using StableRNGs
rng = StableRNG(42)
ms_res = calibrate_multistart(rng, petab_prob, Fides.BFGS(), 50)
```

Passing an `rng` ensures reproducible starting points. The printed summary reports the best
objective value `fmin` across runs. A useful diagnostic is the waterfall plot, which shows
the final objective value for each start (sorted best to worst):

```@example 1
plot(ms_res; plot_type = :waterfall)
```

The plateaus correspond to different local optima; the lowest plateau indicates the best
solution found across multiple runs. Finally, the best-fit simulation against the data can
be plotted as:

```@example 1
plot(ms_res, petab_prob; linewidth = 2.0)
```

!!! tip "Parallelize multi-start runs" `calibrate_multistart` can often be sped up by
running starts in parallel via the `nprocs` keyword (see this tutorial [Multistart
estimation](@ref multistart_est)).

## Next steps

This tutorial introduced how to define a `PEtabODEProblem`. For available options when
building a `PEtabODEProblem` (e.g. for `PEtabParameter`), see the [API](@ref API). For
common use cases, see the following extended tutorials:

- **Simulation conditions**: Measurements collected under different experimental conditions
  (e.g. simulations use different initial values). See [Simulation conditions](@ref
  petab_sim_cond).
- **Steady-state initialization**: Enforce a steady state before the model is matched
  against data (pre-equilibration). See [Pre-equilibration](@ref define_with_ss).
- **Condition-specific parameters**: Estimate parameters whose values differ between
  simulation conditions, while other parameters are shared across conditions. See
  [Simulation condition-specific parameters](@ref define_conditions).
- **Observable and noise parameters**: Observable/noise parameters in `PEtabObservable`
  formulas that are not part of the model system (e.g. scale/offset), optionally
  time-point-specific. See [Observable and noise parameters](@ref petab_observable_options).
- **Events**: Time- or state-triggered events/callbacks. See [Events/callbacks](@ref
  define_events).
- **Import PEtab models**: Load problems provided in the PEtab standard table format. See
  [Import PEtab standard format](@ref import_petab_problem).
- **Model definition**: More on defining `ReactionSystem` and `ODESystem` models can be
  found in the [Catalyst.jl](https://docs.sciml.ai/Catalyst/stable/) and
  [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/dev/) documentation
  respectively.

In addition to defining a parameter estimation problem, this tutorial showed how to fit
parameters using Fides.jl. For more on parameter estimation, see:

- **Extended estimation tutorial**: Extended tutorial on estimation functionality (e.g.
  multi-start, Optimization.jl integration). See [Parameter estimation extended
  tutorial](@ref pest_methods).
- **Available optimization algorithms**: Supported algorithms and recommended defaults. See
  [Available optimization algorithms](@ref options_optimizers).
- **Plotting results**: Plotting options for parameter estimation output. See [Plotting
  parameter estimation results](@ref pest_plotting).
- **Bayesian inference**: Sampling-based inference (e.g. NUTS and AdaptiveMCMC; see the
  Bayesian inference page).

Lastly, `PEtabODEProblem` has many configurable options. The defaults are based on
benchmarks for dynamic models in biology (see [Defaults](@ref default_options)). For models
outside biology, see [Non-stiff models](@ref nonstiff_models). For gradient and Hessian
options, see [Derivative methods](@ref gradient_support).

## Copy pasteable example

```@example 1
using Catalyst, ModelingToolkit, PEtab
using ModelingToolkit: t_nounits as t, D_nounits as D

rn = @reaction_network begin
    @parameters begin
      S0
      c3 = 1.0
    end
    @species begin
      S(t) = S0
      E(t) = 50.0
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

# Observables
@parameters sigma
petab_obs1 = PEtabObservable(:petab_obs1, :obs1, 3.0)
petab_obs2 = PEtabObservable(:petab_obs2, :obs2, sigma)
observables = [petab_obs1, petab_obs2]

# Parameters to estimate
using Distributions
p_c1 = PEtabParameter(:c1)
p_c2 = PEtabParameter(:c2; prior = LogNormal(1.0, 0.3))
p_s0 = PEtabParameter(:S0)
p_sigma = PEtabParameter(:sigma)
pest = [p_c1, p_c2, p_s0, p_sigma]

# Measurements; simulate with 'true' parameters
using OrdinaryDiffEqRosenbrock, DataFrames
ps = [:c1 => 1.0, :c2 => 10.0, :c3 => 1.0, :S0 => 100.0]
u0 = [:S => 100.0, :E => 50.0, :SE => 0.0, :P => 0.0]
tspan = (0.0, 10.0)
oprob = ODEProblem(rn, u0, tspan, ps)
sol = solve(oprob, Rodas5P(); saveat = 0:0.5:10.0)
obs1 = (sol[:S] + sol[:E]) .+ randn(length(sol[:E]))
obs2   = sol[:P] .+ randn(length(sol[:P]))
df1 = DataFrame(obs_id = "petab_obs1", time = sol.t, measurement = obs1)
df2   = DataFrame(obs_id = "petab_obs2", time = sol.t, measurement = obs2)
measurements = vcat(df1, df2)

# Create the PEtabODEProblem
model_rn = PEtabModel(rn, observables, measurements, pest)
model_sys = PEtabModel(sys, observables, measurements, pest)
petab_prob = PEtabODEProblem(model_rn)

# Parameter estimation, single start
using Fides, Plots
x0 = get_startguesses(petab_prob, 1)
res = calibrate(petab_prob, x0, Fides.CustomHessian())
plot(res, petab_prob; linewidth = 2.0)

# Multistart parameter estimation
using StableRNGs
rng = StableRNG(42)
ms_res = calibrate_multistart(rng, petab_prob, Fides.BFGS(), 50)
plot(ms_res; plot_type = :waterfall)
plot(ms_res, petab_prob; linewidth = 2.0)
nothing # hide
```

## References

```@bibliography
Pages = ["tutorial.md"]
Canonical = false
```
