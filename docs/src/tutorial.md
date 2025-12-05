# [Tutorial](@id tutorial)

This overarching tutorial of PEtab.jl covers how to create a parameter estimation problem in
Julia (a `PEtabODEProblem`) and how to estimate the unknown parameters for the created
problem.

## Input Problem

As a working example, this tutorial considers the Michaelis-Menten enzyme kinetics chemical
reaction model:

```math
S + E \xrightarrow{c_1} SE \\
SE \xrightarrow{c_2} S + E \\
SE \xrightarrow{c_3} P + E,
```

Which, via the [law of mass action](https://en.wikipedia.org/wiki/Law_of_mass_action), can
be converted to a system of Ordinary Differential Equations (ODEs):

```math
\begin{align*}
    \frac{\mathrm{d}E}{\mathrm{d}t} &= c_2 SE + c_3 SE - c_1 S \cdot E  \\
    \frac{\mathrm{d}P}{\mathrm{d}t} &= c_3 SE \\
    \frac{\mathrm{d}S}{\mathrm{d}t} &= c_2 SE - c_1 S \cdot E \\
    \frac{\mathrm{d}SE}{\mathrm{d}t} &=  -c_2 SE - c_3 SE + c_1 S \cdot E
\end{align*}
```

For the working example, we assume that the initial values for the species `[S, E, SE, P]`
are:

```math
S(t_0) = S_0, \quad E(t_0) = 50.0, \quad SE(t_0) = 0.0, \quad P(t_0) = 0.0
```

And that the observables for which we have time-lapse measurement data are the sum of
`S + E` as well as `P`:

```math
\begin{align*}
    obs_1 &= S + E \\
    obs_2 &= P
\end{align*}
```

For the parameter estimation, we aim to estimate the parameters `[c1, c2]` and the initial
value `S(t_0) = S0` (a total of three parameters), while assuming `c3 = 1.0` is known. This
tutorial demonstrates how to set up this parameter estimation problem (create a
`PEtabODEProblem`) and estimate parameters using
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).

## Creating the Parameter Estimation Problem

To define a parameter estimation problem we need four components:

1. **Dynamic Model**: The dynamic model can be provided as either a
   [Catalyst.jl](https://petab.readthedocs.io/en/latest/) `ReactionSystem` or a
   [ModellingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) `ODESystem`.
2. **Observable Formulas**: To link the model to the measurement data, we need observable
   formulas. Since real-world data often comes with measurement noise, PEtab also requires
   that noise formulas and noise distributions are provided for each observable. All of this
   is specified with the `PEtabObservable`.
3. **Parameters to Estimate**: A parameter estimation problem needs parameters to be
   estimated. Since often only a subset of the dynamic model parameters is estimated, PEtab
   explicitly requires that the parameters to be estimated are specified as a
   `PEtabParameter`. It is also possible to set priors on these parameters.
4. **Measurement Data**: To estimate parameters, measurement data is required. This data
   should be provided as a `DataFrame` in the format explained below.
5. **Simulation Conditions (Optional)**: Measurements are often collected under various
   experimental conditions, which correspond to different simulation conditions. Details on
   how to handle such conditions are provided in [this](@ref petab_sim_cond) tutorial.

### Defining the Dynamic Model

The dynamic model can be either a [Catalyst.jl](https://github.com/SciML/Catalyst.jl)
`ReactionSystem` or a [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl)
`ODESystem`. For the Michaelis-Menten model above, the Catalyst representation is given by:

```@example 1
using Catalyst
rn = @reaction_network begin
    @parameters S0 c3=1.0
    @species S(t)=S0
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end
```

Parameters that are constant (`c3`) and those that set initial values (`S0`) should be
defined in the `parameters` block. Values for parameters that are to be estimated (here
`[c1, c2, S0]`) do not need to be specified. Similarly, for species, only those with a
parameter-dependent initial value need to be defined in the `species` block, while species
with a constant initial value can be defined directly in the system (similar to `c3` above)
or as a specie map:

```@example 1
speciemap = [:E => 50.0, :SE => 0.0, :P => 0.0]
```

Any species or parameters with undeclared initial values default to 0. For additional
details on how to create a `ReactionSystem`, see the excellent Catalyst
[documentation](https://docs.sciml.ai/Catalyst/stable/).

Using a ModelingToolkit `ODESystem`, the model is defined as:

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
    end
    @equations begin
        D(S) ~ -c1 * S * E + c2 * SE
        D(E) ~ -c1 * S * E + c2 * SE + c3 * SE
        D(SE) ~ c1 * S * E - c2 * SE - c3 * SE
        D(P) ~ c3 * SE
    end
end
@mtkbuild sys = SYS()
```

For an `ODESystem`, all parameters and species must be declared in the `@mtkmodel` block. If
the value of a parameter or species is left empty (e.g., `c2` above) and the parameter is
not set to be estimated, it defaults to 0. For additional details on how to create an
`ODESystem` model, see the ModelingToolkit
[documentation](https://docs.sciml.ai/ModelingToolkit/dev/).

### Defining the Observables

To connect the model with measurement data, we need an observable formula. Additionally,
since measurement data is typically noisy, PEtab requires a measurement noise formula.

For example, let us assume we have observed the sum `E + S` ($obs_1$ above) with a known
normally distributed measurement error (`σ = 3.0`). This in encoded as:

```@example 1
using PEtab
@unpack E, S = rn
obs_sum = PEtabObservable(S + E, 3.0)
```

Note that `@unpack` is a
[Julia macro](https://docs.julialang.org/en/v1/manual/metaprogramming/) that can
conveniently be used to extract any species from a Catalyst `ReactionSystem` (more details
can be found
[here](https://docs.sciml.ai/Catalyst/stable/model_creation/dsl_advanced/#dsl_advanced_options_symbolics_and_DSL_unpack)).
In `PEtabObservable`, the first argument is the observed formula, and the second argument is
the formula for the measurement error. In this case, we assumed a known measurement error
(`σ = 3.0`), but often the measurement error is unknown and needs to be estimated. For
example, let us assume we have observed `P` ($obs_2$) with an unknown measurement error
`sigma`. This in encoded as:

```@example 1
@unpack P = rn
@parameters sigma
obs_p = PEtabObservable(P, sigma)
```

By defining `sigma` as a `PEtabParameter` (explained below), it is estimated along with the
other parameters. To complete the definition of the observables, we need to group all
`PEtabObservable`s together into a `Dict` and assign an appropriate name for each
observable:

```@example 1
observables = Dict("obs_p" => obs_p, "obs_sum" => obs_sum)
```

More formally, a `PEtabObservable` defines a likelihood function for an observable. By
default, a normally distributed error and corresponding likelihood is assumed, but
log-normal distribution is also supported. For more details, see the [API](@ref API).

### Defining Parameters to Estimate

To set up a parameter estimation problem, we need to specify the parameters to estimate via
`PEtabParameter`. To set `c1` to be estimated, use:

```@example 1
p_c1 = PEtabParameter(:c1)
```

From the printout, we see that by default `c1` is assigned bounds `[1e-3, 1e3]`. This is
because benchmarks have shown that using bounds is advantageous, as it prevents simulation
failures during parameter estimation[frohlich2022fides](@cite). Furthermore, we see that by
default `c1` is estimated on a `log10` scale. Benchmarks have demonstrated that estimating
parameters on a `log10` scale improves performance
[raue2013lessons, hass2019benchmark](@cite). Naturally, it is possible to change the bounds
and/or scale; see the [API](@ref API) for details.

When specifying a `PEtabParameter` we can also provide prior information. For example,
assume we know that `c2` should have a value around 10. To account for this we can provide a
[prior](https://en.wikipedia.org/wiki/Prior_probability) for the parameter using any
continuous distribution from
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl). For example, to assign a
`Normal(10.0, 0.3)` prior to `c2`, do:

```@example 1
using Distributions
p_c2 = PEtabParameter(:c2; prior = Normal(10.0, 0.3))
```

By default, the prior is on a linear scale (not the default `log10` scale), but this can be
changed if needed. For more details, see the [API](@ref API).

To complete the definition of the parameters to estimate, we need to assign a
`PEtabParameter` for each unknown and group them into a `Vector`:

```@example 1
p_s0 = PEtabParameter(:S0)
p_sigma = PEtabParameter(:sigma)
pest = [p_c1, p_c2, p_s0, p_sigma]
```

### Measurement Data Format

The measurement data should be provided in a `DataFrame` with the following format (the
column names matter, but not the order):

| obs_id (str) | time (float) | measurement (float) |
| ------------ | ------------ | ------------------- |
| id           | val          | val                 |
| ...          | ...          | ...                 |

Where the columns correspond to:

- `obs_id`: The observable to which the measurement corresponds. It must match one of the
  `keys` in the `PEtabObservable` `Dict`.
- `time`: The time point at which the measurement was collected.
- `measurement`: The measurement value.

For our working example, using simulated data, a valid measurement table would look like:

```@example 1
using OrdinaryDiffEq, DataFrames
# Simulate with 'true' parameters
ps = [:c1 => 1.0, :c2 => 10.0, :c3 => 1.0, :S0 => 100.0]
u0 = [:S => 100.0, :E => 50.0, :SE => 0.0, :P => 0.0]
tspan = (0.0, 10.0)
oprob = ODEProblem(rn, u0, tspan, ps)
sol = solve(oprob, Rodas5P(); saveat = 0:0.5:10.0)
obs_sum = (sol[:S] + sol[:E]) .+ randn(length(sol[:E]))
obs_p = sol[:P] .+ randn(length(sol[:P]))
df_sum = DataFrame(obs_id = "obs_sum", time = sol.t, measurement = obs_sum)
df_p = DataFrame(obs_id = "obs_p", time = sol.t, measurement = obs_p)
measurements = vcat(df_sum, df_p)
first(measurements, 5) # hide
```

It is important to note that the measurement table follows a
[tidy](https://r4ds.hadley.nz/data-tidy) format [wickham2014tidy](@cite), where each row
corresponds to **one** measurement. Therefore, for repeated measurements at a single time
point, one row should be added for each repeat.

### Bringing It All Together

Given a model, observables, parameters to estimate, and measurement data, it is possible to
create a `PEtabODEProblem`, which contains all the information needed for parameter
estimation. This is done in a two-step process, where the first step is to create a
`PEtabModel`. For our `ReactionSystem`, this is done as:

```@example 1
model_rn = PEtabModel(rn, observables, measurements, pest; speciemap = speciemap)
nothing # hide
```

For an `ODESystem` the syntax is the same:

```@example 1
model_sys = PEtabModel(sys, observables, measurements, pest)
nothing # hide
```

Note that any potential `speciemap` or `parametermap` must be provided as a keyword. Given a
`PEtabModel`, it is straightforward to create a `PEtabODEProblem`:

```@example 1
petab_prob = PEtabODEProblem(model_rn)
```

The printout shows relevant statistics for the `PEtabODEProblem`. First, we see that there
are 4 parameters to estimate. Additionally, we see that the ODE solver used for simulating
the model is the stiff `Rodas5P` solver, and that both the gradient and Hessian are computed
via forward-mode automatic differentiation using
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl). These [defaults](@ref
default_options) are based on extensive benchmarks and typically do not need to be changed
for models in biology. For models outside of biology, a discussion of the options can be
found [here](@ref default_options). While the defaults generally perform well, they are not
always perfect. Therefore, when creating a `PEtabODEProblem`, anything from the `ODESolver`
to the gradient methods can be customized. For details, see the [API](@ref).

Overall, the `PEtabODEProblem` contains all the information needed for performing parameter
estimation. Next, this tutorial covers how to estimate unknown model parameters given a
`PEtabODEProblem`.

## Parameter estimation

A `PEtabODEProblem` (which we defined above) contains all the information needed to wrap a
numerical optimization library to perform parameter estimation, and details on how to do
this can be found [here](@ref wrap_est). However, wrapping existing optimization libraries
is cumbersome, therefore PEtab.jl provides wrappers for
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl),
[Ipopt](https://coin-or.github.io/Ipopt/),
[Optimization.jl](https://github.com/SciML/Optimization.jl), and
[Fides.jl](https://fides-dev.github.io/Fides.jl/stable/).

This section of the tutorial covers how to use
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) to estimate parameters given a
starting guess `x0`. Moreover, since the objective function to minimize for ODE models often
contains multiple local minima, the tutorial also covers how to perform global optimization
using multistart parameter estimation.

### Single-Start Parameter Estimation

To perform parameter estimation with a numerical optimization algorithm, we typically need a
starting point `x0`, where it is important that `x0` follows the parameter order expected by
the `PEtabODEProblem`. One way to obtain such a vector is by retrieving the
`PEtabODEProblem`'s vector of nominal values, which correspond to the optional parameter
values specified in `PEtabParameter` (if unspecified, these values default to the mean of
the lower and upper bounds). This vector can be retrieved with `get_x`:

```@example 1
x0 = get_x(petab_prob)
```

From the printout we see that `x0` is a `ComponentArray`, so in addition to the parameter
values, it also holds the parameter names. Additionally, we see that parameters like
`log10_c1` have a `log10` prefix. This is because the parameter (by default) is estimated on
the `log10` scale, which, as mentioned above, often improves parameter estimation
performance [hass2019benchmark](@cite). Consequently, when changing the value for this
parameter, the new value should be provided on the `log10` scale. For example, to change
`c1` to `10.0` do:

```@example 1
x0.log10_c1 = log10(10.0)
nothing # hide
```

For more details on how to interact with a `ComponentArray`, see the ComponentArrays.jl
[documentation](https://github.com/jonniedie/ComponentArrays.jl). `get_x` is not the only
way, and generally not the recommended way to retrieve a starting point. To avoid biasing
the parameter estimation, it is recommended to use a random starting guess within the
parameter bounds. This can be generated with `get_startguesses`:

```@example 1
using Random # hide
Random.seed!(123) # hide
x0 = get_startguesses(petab_prob, 1)
nothing # hide
```

Given a starting point `x0`, we can now perform the parameter estimation. As this is a small
problem with only 4 parameters to estimate, we use the Interior-point Newton method from
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) (for algorithm recommendations, see
[this](@ref options_optimizers) page):

```@example 1
using Optim
res = calibrate(petab_prob, x0, IPNewton())
```

The printout shows parameter estimation statistics, such as the final objective value `fmin`
(which, since PEtab works with likelihoods, corresponds to the negative log-likelihood). We
can further obtain the minimizing parameter vector:

```@example 1
res.xmin
```

This vector is close to the true parameters used to simulate the data above. For information
on additional statistics stored in `res`, see the [API](@ref) on `PEtabOptimisationResult`.

Lastly, to evaluate the parameter estimation, it is useful to plot how well the model fits
the data. Using the built-in plotting functionality in PEtab, this is straightforward:

```@example 1
using Plots
default(left_margin=12.5Plots.Measures.mm, bottom_margin=12.5Plots.Measures.mm, size = (600*1.25, 400 * 1.25), palette = ["#CC79A7", "#009E73", "#0072B2", "#D55E00", "#999999", "#E69F00", "#56B4E9", "#F0E442"], linewidth=4.0) # hide
plot(res, petab_prob; linewidth = 2.0)
plot(res, petab_prob; linewidth = 2.0) # hide
```

Even though the plot looks good, it is important to remember that ODE models often have
multiple local minima [raue2013lessons](@cite). To ensure the global optimum is found, a
global optimization approach is required. One effective method is multi-start parameter
estimation, which we cover next.

### Multi-Start Parameter Estimation

In multi-start parameter estimation, `n` parameter estimation runs are initiated from `n`
random starting points. The rationale is that a subset of these runs should converge to the
global optimum, and even though this is a simple global optimization approach, benchmarks
have shown that it performs well for ODE models in biology
[raue2013lessons, villaverde2019benchmarking](@cite).

The first step in multi-start parameter estimation is to generate `n` starting points.
Simple uniform sampling is not preferred, as randomly generated points tend to cluster.
Instead, a [Quasi-Monte Carlo](https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method)
method, such as
[Latin hypercube sampling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling), is
better suited to generate well-spread starting points. In `get_startguesses`,
`LatinHypercubeSample` is the default method used. Therefore, `n = 50` Latin
hypercube-sampled starting points can be generated with:

```@example 1
Random.seed!(123) # hide
x0s = get_startguesses(petab_prob, 50)
nothing # hide
```

Besides `LatinHypercubeSample`, `get_startguesses` also supports other sampling methods; for
details, see the [API](@ref API). Given our starting points, we can perform multi-start
parameter estimation:

```@example 1
res = Any[]
for x0 in x0s
    push!(res, calibrate(petab_prob, x0, IPNewton()))
end
nothing # hide
```

As manually generating start guesses and calling `calibrate` can be cumbersome, PEtab.jl
provides a convenience function, `calibrate_multistart`. For example, to run `n = 50`
multistarts, do:

```@example 1
Random.seed!(123) # hide
ms_res = calibrate_multistart(petab_prob, IPNewton(), 50)
```

The printout shows parameter estimation statistics, such as the best objective value `fmin`
across all runs. For further details on what is stored in `ms_res` see the [API](@ref API)
documentation for `PEtabMultistartResult`.

!!! tip "Parallelize Parameter Estimation" Runtime of `calibrate_multistart` can often be
reduced by performing parameter estimation runs in parallel. To do this, set the `nprocs`
keyword argument, more details can be found [here](@ref multistart_est).

Following multi-start parameter estimation, it is important to evaluate the results. One
common evaluation approach is plotting, and a frequently used evaluation plot is the
waterfall plot, which in a sorted manner shows the final objective values for each run:

```@example 1
plot(ms_res; plot_type=:waterfall)
```

In the waterfall plot, each plateau corresponds to different local optima. Since many runs
(dots) are found on the plateau with the smallest objective value, we can be confident that
the global optimum has been found. Further, we can check how well the best run fits the
data:

```@example 1
plot(ms_res, petab_prob; linewidth = 2.0)
```

## Next Steps

This overarching tutorial provides an overview of how to create a parameter estimation
problem in Julia. As an introduction, it showcases only a subset of the features supported
by PEtab.jl for creating parameter estimation problems. In the extended tutorials, you will
find how to handle:

- **Simulation conditions**: Sometimes data is gathered under various experimental
  conditions, where, for example, the initial concentration of a substrate differs between
  condition. To learn how to setup a problem with such simulation conditions see [this](@ref
  petab_sim_cond) tutorial.
- **Steady-State Initialization**: Sometimes the model should be at a steady state at time
  zero, before it is simulated and compared against data. To learn how to set up a problem
  with such pre-equilibration criteria, see [this](@ref define_with_ss) tutorial.
- **Events**: Sometimes a model may incorporate events like substrate addition at specific
  time points or parameter changes when a state/species reaches a certain value. To learn
  how to add model events see [this](@ref define_events) tutorial.
- **Condition-Specific System/Model Parameters**: Sometimes a subset of model parameters to
  estimate, such as protein synthesis rates, varies between simulation conditions, while
  other parameters remain constant across all conditions. To learn how to handle
  condition-specific parameters, see [this](@ref define_conditions) tutorial.
- **Timepoint Specific Parameters**: Sometimes one observable is measured with different
  assays. This can be handled by introducing different observable parameters (e.g., scale
  and offset) and noise parameters for different measurements. To learn how to add
  timepoint-specific measurement and noise parameters, see [this](@ref
  time_point_parameters) tutorial.
- **Import PEtab Models**: PEtab is a standard table-based format for parameter estimation.
  If a problem is provided in this standard format, PEtab.jl can import it directly. To
  learn how to import models in the standard format, see [this](@ref import_petab_problem)
  tutorial.

Besides creating a parameter estimation problem, this overarching tutorial demonstrated how
to perform parameter estimation using
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl). In addition, PEtab.jl also supports
using [Ipopt](https://coin-or.github.io/Ipopt/),
[Optimization.jl](https://github.com/SciML/Optimization.jl), and
[Fides.jl](https://fides-dev.github.io/Fides.jl/stable/). More information on available
algorithms for parameter estimation can be found on [this](@ref pest_methods) page. Besides
frequentist parameter estimation, PEtab.jl also supports Bayesian inference with
state-of-the-art samplers such as [NUTS](https://github.com/TuringLang/Turing.jl) (the same
sampler used in [Turing.jl](https://github.com/TuringLang/Turing.jl)) and
[AdaptiveMCMC.jl](https://github.com/mvihola/AdaptiveMCMC.jl). For more information, see the
Bayesian inference [page].

Lastly, when creating a `PEtabODEProblem` there are many configurable options (see the
[API](@ref API)). The default options are based on extensive benchmarks for dynamic models
in biology, see [this](@ref default_options) page. For how to configure models outside of
biology, see [this](@ref nonstiff_models) page. Additionally, for a discussion on available
gradient and Hessian methods, see [this](@ref gradient_support) page.

## Copy Pasteable Example

```@example 1
using Catalyst, PEtab
# Create the dynamic model(s)
rn = @reaction_network begin
    @parameters S0 c3=1.0
    @species S(t)=S0
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end
speciemap = [:E => 50.0, :SE => 0.0, :P => 0.0]

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
    end
    @equations begin
        D(S) ~ -c1 * S * E + c2 * SE
        D(E) ~ -c1 * S * E + c2 * SE + c3 * SE
        D(SE) ~ c1 * S * E - c2 * SE - c3 * SE
        D(P) ~ c3 * SE
    end
end
@mtkbuild sys = SYS()

# Observables
@unpack E, S = rn
obs_sum = PEtabObservable(S + E, 3.0)
@unpack P = rn
@parameters sigma
obs_p = PEtabObservable(P, sigma)
observables = Dict("obs_p" => obs_p, "obs_sum" => obs_sum)

# Parameters to estimate
using Distributions
p_c1 = PEtabParameter(:c1)
p_c2 = PEtabParameter(:c2; prior = Normal(10.0, 0.3))
p_s0 = PEtabParameter(:S0)
p_sigma = PEtabParameter(:sigma)
pest = [p_c1, p_c2, p_s0, p_sigma]

# Simulate measurement data with 'true' parameters
using OrdinaryDiffEq, DataFrames
ps = [:c1 => 1.0, :c2 => 10.0, :c3 => 1.0, :S0 => 100.0]
u0 = [:S => 100.0, :E => 50.0, :SE => 0.0, :P => 0.0]
tspan = (0.0, 10.0)
oprob = ODEProblem(rn, u0, tspan, ps)
sol = solve(oprob, Rodas5P(); saveat = 0:0.5:10.0)
obs_sum = (sol[:S] + sol[:E]) .+ randn(length(sol[:E]))
obs_p = sol[:P] + .+ randn(length(sol[:P]))
df_sum = DataFrame(obs_id = "obs_sum", time = sol.t, measurement = obs_sum)
df_p = DataFrame(obs_id = "obs_p", time = sol.t, measurement = obs_p)
measurements = vcat(df_sum, df_p)

model_sys = PEtabModel(sys, observables, measurements, pest)
model_rn = PEtabModel(rn, observables, measurements, pest; speciemap = speciemap)
petab_prob = PEtabODEProblem(model_rn)

# Parameter estimation
using Optim, Plots
x0 = get_startguesses(petab_prob, 1)
res = calibrate(petab_prob, x0, IPNewton())
plot(res, petab_prob; linewidth = 2.0)

ms_res = calibrate_multistart(petab_prob, IPNewton(), 50)
plot(ms_res; plot_type=:waterfall)
plot(ms_res, petab_prob; linewidth = 2.0)
nothing # hide
```

## References

```@bibliography
Pages = ["tutorial.md"]
Canonical = false
```
