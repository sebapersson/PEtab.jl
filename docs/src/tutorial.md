# Tutorial

This overarching tutorial of PEtab covers how to create a parameter estimation problem in Julia (a `PEtabODEProblem`) and how to estimate parameters for the created problem.

## Input problem

As a working example, this tutorial considers the Michaelis-Menten enzyme kinetics model:

```math
S + E \xrightarrow{c_1} SE \\
SE \xrightarrow{c_2} S + E \\
SE \xrightarrow{c_3} S + P,
```

Using the [law of mass action](https://en.wikipedia.org/wiki/Law_of_mass_action), this `ReactionSystem` can be converted to an `ODESystem`:

```math
\begin{align}
    \frac{\mathrm{d}S}{\mathrm{d}t} &= c_1 S \cdot E - c_2 SE \\
    \frac{\mathrm{d}E}{\mathrm{d}t} &= c_1 S \cdot E - c_2 SE \\
    \frac{\mathrm{d}SE}{\mathrm{d}t} &= -c_1S*E - c_2SE - c_3SE \\
    \frac{\mathrm{d}P}{\mathrm{d}t} &= c_3SE
\end{align}
```

We further assume that the initial values for the species `[S, E, P]` are:

```math
S(t_0) = S0, \quad E(t_0) = 0.0, \quad SE(t_0) = 0.0, \quad P(t_0) = 0.0
```

and that as observables, that we have measurements on the sum of `S + E` as well as on `P`:

```math
\begin{align}
    obs_1 &= S + E \\
    obs_2 &= P
\end{align}
```

For the parameter estimation, we aim to estimate the parameters `[c1, c2]` and the initial value `[S0]` (a total of three parameters), while assuming that `c3 = 1.0` is known. This tutorial demonstrates how to set up a parameter estimation problem (`PEtabODEProblem`) and how to estimate parameters using [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).

## Creating the Parameter Estimation Problem

To define a parameter estimation problem we need four components:

1. **Dynamic Model**: The dynamic model can be either a [Catalyst](https://petab.readthedocs.io/en/latest/) or a [ModellingToolkit](https://github.com/SciML/ModelingToolkit.jl)`ODESystem`.
2. **Observable Formula**: To link the model to the measurement data, we need an observable formula. Since real-world data often comes with measurement noise, PEtab also requires that a noise formula and noise distribution is provided. This is done via the `PEtabObservable`.
3. **Parameters to Estimate**: A parameter estimation problem needs parameters to estimate. As often it not relevant to estimate all model parameter PEtab explicitly requires that all parameters to estimate are specieed as a `PEtabParameter`. Moreover, here it is also possible to assign potential priors to the parameters.
4. **Measurement Data**: To parameter estimate the model, measurement data is needed. Measurement data should be provided as a `DataFrame`, and the format is explained below.
5. **Simulation Conditions (optional)**: Measurements are often collected under various experimental conditoin, that corresponds to different simulation conditions. More details on how to handle this scenario see this [tutorial](ADD).

### Defining the Dynamic Model

The dynamic model can be either a [Catalyst](https://github.com/SciML/Catalyst.jl) `ReactionSystem` or a [ModelingToolkit](https://github.com/SciML/ModelingToolkit.jl) `ODESystem`. For the Michaelis-Menten model above, the Catalyst model is given by:

```@example 1
using Catalyst
t = default_t()
rn = @reaction_network begin
    @parameters begin
        S0
        c3 = 1.0
    end
    @species begin
        SE(t) = S0
    end
    c1, S + E --> SE
    c2, SE --> S + E
    c3, SE --> P + E
end
```

Parameters that are constant (`c3`) and those that set initial values (`S0`) should be defined in the `parameters` block. The other parameters which are to be estimated (here `[c1, c2]`) only need to be defined in the reactions. Similarly, for species, only those with a declared initial value need to be defined in the `species` block, while species with undeclared initial values default to 0. For additional details on how to create a `ReactionSystem` model, see the excellent Catalyst [documentation](https://docs.sciml.ai/Catalyst/stable/).

Using a ModelingToolkit `ODESystem`, the model is defined as:

```@example 1
using ModelingToolkit
t = default_t()
D = default_time_deriv()
@mtkmodel SYS begin
    @parameters begin
        S0
        c1
        c2
        c3 = 1.0
    end
    @variables begin
        S(t) = S0
        E(t) = 0.0
        SE(t) = 0.0
        P(t) = 0.0
    end
    @equations begin
        D(S) ~ -c1 * S * E + c2 * SE,
        D(E) ~ -c1 * S * E + c2 * SE + c3 * SE,
        D(SE) ~ c1 * S * E - c2 * SE - c3 * SE,
        D(P) ~ c3 * SE
    end
end
@mtkbuild sys = SYS()
```

For an `ODESystem`, all parameters and species must be declared. If the value of a parameter or species is left empty (e.g., `c2` above) and the parameter is not set to be estimated, its value defaults to 0. For additional details on how to create an `ODESystem` model, see the ModelingToolkit [documentation](https://docs.sciml.ai/ModelingToolkit/dev/).

### Defining the Observables

To connect the model with measurement data, we need an observable formula. Additionally, since measurement data is typically noisy, PEtab requires a measurement noise formula.

For example, let us assume we have observed the sum `E + S` with a known normally distributed measurement error (`σ = 3.0`). This can be encoded as:

```@example 1
using PEtab
@unpack E, S = rn
obs_sum = PEtabObservable(S + E, 3.0)
```

In `PEtabObservable`, the first argument is the observed formula, and the second argument is the formula for the measurement error. In this case, we assumed a known measurement error (`σ = 3.0`), but often the measurement error is unknown and needs to be estimated. Now, let's assume we have also observed `P` with an unknown measurement error `sigma`. This can be coded as:

```@example 1
@unpack P = rn
@parameters sigma
obs_p = PEtabObservable(P, sigma)
```

By defining `sigma` as a `PEtabParameter` (explained below), it can be estimated along with the other parameters. To complete the definition of the observables, we need to group together into a `Dict` with appropriate names:

```@example 1
observables = Dict("obs_p" => obs_p, "obs_sum" => obs_sum)
```

More formally, the `PEtabObservable` defines a likelihood function for an observable. By default, a normally distributed error is assumed, but log-normal is also supported. For more details, see the [API](@ref API).

### Defining Parameters to Estimate

To set up a parameter estimation problem, we need to specify the parameters to estimate using. To set the parameter `c1` to be estimated, do:

```@example 1
p_c1 = PEtabParameter(:c1)
```

From the printout, we can see that by default `c1` is assigned bounds `[1e-3, 1e3]`. This is because several benchmark studies have shown that using bounds is advantageous, as it prevents simulation failures during model fitting [ADD]. Additionally, by default `c1` is estimated on a `log10` scale. This is again because numerous benchmark studies have demonstrated that estimating parameters on a `log10` scale is beneficial [ADD]. Naturally, it is possible to change the bounds and/or scale, for details see the [API](@ref API).

With `PEtabParameter`, we can also incorporate prior information. For example, assume we know that `c2` should have a value around 10. To account for this we can provide a [prior](https://en.wikipedia.org/wiki/Prior_probability) for the parameter using any continuous distribution from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl). For example, to assign a `Normal(10.0, 0.3)` prior to `c2`, do:

```julia
using Distributions
p_c2 = PEtabParameter(:c2; prior = Normal(10.0, 0.3))
```

By default, the prior is on a linear scale (not the default `log10` scale), but this can be changed if needed. For more details, see the [API](@ref API).

To complete the definition of the parameters, we need to assign a `PEtabParameter` for each unknown and group them into a `Vector`:

```@example 1
p_s0 = PEtabParameter(:S0)
p_sigma = PEtabParameter(:sigma)
pest = [p_c1, p_c2, p_s0, p_sigma]
```

### Measurement Data Format

The measurement data should be provided in a `DataFrame` with the following format (the column names matter, but not the order):

| obs_id (str) | time (float) | measurement (float) |
|--------------|--------------|---------------------|
| id           | val          | val                 |
| ...          | ...          | ...                 |

For each measurement, it is necessary to specify:

- `obs_id`: The observable the measurement corresponds to. Must correspond to one of the `PEtabObservable` ids above.
- `time`: The time point at which the measurement was collected.
- `measurement`: The measurement value.

For our working example, simulating some data the measurement table would look like:

```@example 1
using OrdinaryDiffEq, DataFrames
# Simulate with 'true' parameters
ps = [c1 => 1.0, c2 => 10.0, c3 => 1.0]
u0 = [S => 100.0, E => 0.0, SE => 0.0, P => 0.0]
tspan = (0.0, 10.0)
oprob = ODEProblem(sys, u0, tspan, p)
sol = solve(oprob, Rodas5P(); saveat = 0:0.1:10.0)
obs_sum = sol[:S] + sol[:E]
obs_p = sol[:P]
df_sum = DataFrame(obs_id = "obs_sum", time = sol.t, measurement = obs_sum)
df_p = DataFrame(obs_id = "obs_p", time = sol.t, measurement = obs_p)
measurements = vcat(df_sum, df_p)
```

It is important to note that the measurement table follows a [tidy](https://r4ds.hadley.nz/data-tidy) format, where each row corresponds to **one** measurement. Therefore, for repeated measurements at a single time point one row should be added for each repeat.

### Bringing It All Together

Given a model, observables, parameters to estimate, and measurement data, a `PEtabODEProblem` can be created. The `PEtabODEProblem` contains all the information needed for parameter estimation and is created in a two-step process. The first step is to create a `PEtabModel`, which can be done using either our `ReactionSystem` model or `ODESystem` model as follows:

```@example 1
model_sys = PEtabModel(sys, observables, measurements, pest)
model_rn = PEtabModel(rn, observables, measurements, pest)
nothing # hide
```

Given a `PEtabModel`, it is straightforward to create a `PEtabODEProblem`:

```@example 1
petab_prob = PEtabODEProblem(model_rn)
```

The printout highlights relevant statistics for the `PEtabODEProblem`. First, we see that there are 4 parameters to estimate. We also see that the ODE solver used for simulating the model is the stiff `Rodas5P` solver, and that the gradient and Hessian are computed via forward-mode automatic differentiation using [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl). These defaults are based on extensive benchmarks and often do not need to be changed. However, defaults are not always perfect, and when constructing the `PEtabODEProblem`, anything from the `ODESolver` to gradient methods can be specified. For more details, see the [API](@ref), and for information on the defaults, see ADD.

Overall, the `PEtabODEProblem` contains all the information needed for performing parameter estimation. Next, this tutorial covers how to estimate unknown model parameters using multistart parameter estimation.

## Parameter estimation

A `PEtabODEProblem` (which we defined above) contains all the information needed to wrap a numerical optimization library to perform parameter estimation. Details on how to do this can be found in this [tutorial](). However, as wrapping existing optimization libraries can be cumbersome, PEtab.jl provides wrappers for [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [Ipopt](https://coin-or.github.io/Ipopt/), [Optimization.jl](https://github.com/SciML/Optimization.jl), and [Fides.py](https://github.com/fides-dev/fides).

This tutorial shows how to use Optim.jl to estimate parameters given a starting guess `x0`. Moreover, since the objective function to minimize for ODE models often contains multiple local minima, the tutorial then covers how to perform global optimization using multistart parameter estimation.

### Single-Start Parameter Estimation

To perform numerical optimization (in other words, parameter estimation), we need a starting point `x0`. Importantly, the parameters in `x0` must be in the order expected by the `PEtabODEProblem`. Such a starting point can be obtained in several ways. First, the `PEtabODEProblem` holds a vector of nominal values, which correspond to the optional values specified in `PEtabParameter` and default to the mean of the lower and upper bounds. To access this vector, use `get_x`:

```@example 1
x0 = get_x(petab_prob)
```

From the printout, we can see that `x0` is a `ComponentArray`, so in addition to the values, it also holds the parameter names. Additionally, we see that parameters like `log10_c1` have a `log10` prefix. This is because the parameter (by default) is estimated on the `log10` scale, which, as mentioned above, often improves parameter estimation performance. Consequently, when changing the value for this parameter, the new value should be provided on the `log10` scale. Furthermore, as `x0` is a `ComponentArray`, it is easy to change any value. For example, to change `c1` to `10.0` do:

```@example 1
x0.log10_c1 = log10(10.0)
```

For more details on how to interact with a `ComponentArray`, see the package's [documentation](https://github.com/jonniedie/ComponentArrays.jl). To avoid bias in the parameter estimation, it is recommended to generate a random starting guess. This can be done `get_startguesses`:

```@example 1
using Random # hide
Random.seed!(123) # hide
x0 = get_startguesses(petab_prob, 1)
```

Again `x0` is a `ComponentArray`. With this starting guess, we can now do parameter estimation Since this is a small problem with only 4 parameters to estimate, we use the Interior-point Newton method from Optim.jl. For recommendations on optimization algorithms, see this [page](ADD!).

```@example 1
using Optim
res = calibrate(petab_prob, x0, IPNewton())
```

The printout provides general statistics, such the final objective value `fmin` (since PEtab uses likelihoods, this value represents the negative likelihood). To view the minimizing parameter vector, use:

```@example 1
res.xmin
```

For information on additional statistics stored in `res` see the [API](@ref ) on `PEtabOptimisationResult`. 

To evaluate the results of parameter estimation, it is useful to plot how well the model fits the data. Using the built-in plotting functionality in PEtab, this is straightforward:

```@example 1
using Plots
plot(res, petab_prob)
```

Even if the plot looks good, it is important to remember that ODE models often have multiple local minima. To ensure the global optimum is found, global optimization is needed. One effective global optimization approach is multi-start parameter estimation, which we will explore next.

### Multi-Start Parameter Estimation

Multi-start parameter estimation is an approach where `n` parameter estimation runs are initiated from `n` random starting points. The rationale is that a subset of these runs should converge to the global optimum. Even though this is a simple global optimization approach, empirical benchmark studies have shown that this method performs well for ODE models in biology.

The first step in multi-start parameter estimation is to generate `n` starting points. Simple random uniform sampling is not ideal, as randomly generated points tend to cluster. Instead, it is better to use a [Quasi-Monte Carlo](https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method) method, such as [Latin hypercube sampling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling), which is the default method used by `get_startguesses`. Therefore, to generate `n = 100` Latin hypercube sampled parameters, do:

```@example 1
Random.seed!(123) # hide
x0s = get_startguesses(petab_prob, 100)
```

Under the hood, `get_startguesses` uses the `LatinHypercubeSample` method from [QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl). For information on how to change the sampling algorithm, see the [API](@ref API). Given our starting points, we can now perform multi-start parameter estimation:

```@example 1
res = Any[]
for x0 in x0s
    push!(res, calibrate(petab_prob, x0, IPNewton()))
end
nothing # hide
```

Even though feasible, manually generating start guesses and calling `calibrate` is cumbersome. Therefore, PEtab.jl provides a convenience function for multi-start parameter estimation. To run `n = 100` multistarts, simply use:

```@example 1
ms_res = calibrate_multistart(petab_prob, 100, IPNewton())
```

From the printout, we can see that the best minimum found is ADD. Besides `fmin`, more information is available in `ms_res` and for further details see the [API](@ref API) documentation for `PEtabMultistartResult`. 

A common approach to evaluate the result of multi-start parameter estimation is through plotting. One commonly used evaluation plot is the waterfall plot, which shows the final objective values for each run:

```@example 1
plot(ms_res; plot_type=:waterfall)
```

In the waterfall plot, each plateau corresponds to different local optima (represented by different colors). Since many runs (dots) are found on the plateau with the smallest objective value, we can be confident that the global optimum has been found. Furthermore, we can check how well the best run fits the data:

```@example 1
plot(res_ms, petab_prob)
```

## Next Steps

This overarching tutorial provides an overview of how to set up a PEtab parameter estimation problem in Julia. As an introduction, it only showcases a subset of the features supported by PEtab.jl for parameter estimation problems. In the in-depth tutorials, you will find how to handle:

- **Steady-State Initialization**: In some cases, the model should be at a steady state at time zero, before it is simulated and compared against data. To learn how to set up a problem with such pre-equilibration criteria, see [this](@ref define_with_ss) tutorial.
- **Time-Point Specific Parameters**: Sometimes the same observable is measured with different assays. This can be handled by introducing different observable parameters (e.g., scale and offset) and noise parameters for different measurements. To learn how to add time-point-specific measurement and noise parameters, see [this](@ref time_point_parameters) tutorial.
- **Condition-Specific System/Model Parameters**: Sometimes a subset of model parameters, such as protein synthesis rates, varies between simulation conditions, while other parameters remain constant across all conditions. To learn how to handle condition-specific parameters, see [this](@ref define_conditions) tutorial.
- **Events**: Sometimes a model may incorporate events like substrate addition at specific time points or parameter changes when a state/species reaches certain values. To learn how to add model events see [this](@ref define_events) tutorial.
- **Import PEtab Models**: PEtab is a standard table-based format for parameter estimation. If a problem is provided in this standard format, PEtab.jl can import it directly. To learn how to import models in the standard format, see [this] tutorial.

This tutorial also demonstrated how to perform parameter estimation using [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl). Additionally, PEtab.jl supports [Ipopt](https://coin-or.github.io/Ipopt/), [Optimization.jl](https://github.com/SciML/Optimization.jl), and [Fides.py](https://github.com/fides-dev/fides). For information on how to use these packages, see the parameter estimation [page]. Furthermore, besides frequentist parameter estimation, PEtab.jl also supports Bayesian inference with state-of-the-art methods such as [NUTS](https://github.com/TuringLang/Turing.jl) (the same sampler used in [Turing.jl](https://github.com/TuringLang/Turing.jl)) and [AdaptiveMCMC.jl](https://github.com/mvihola/AdaptiveMCMC.jl). For more information, see the Bayesian inference [page].

Lastly, for further details on each function showcased in the tutorials, see the [API](@ref API). If you are interested in performance and how to configure `PEtabODEProblem`, see the [Choosing the Best Options for a PEtab Problem](@ref best_options) section and the [Supported Gradient and Hessian Methods](@ref gradient_support). Moreover, the example pages provide more in-depth guidance on how to configure the `PEtabODEProblem` for various problem types.
