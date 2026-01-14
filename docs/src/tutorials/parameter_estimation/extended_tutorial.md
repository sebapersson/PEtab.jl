```@meta
CollapsedDocStrings=true
```

# [Parameter estimation tutorial](@id pest_methods)

A `PEtabODEProblem` provides runtime-efficient objective, gradient, and (optionally) Hessian
functions for estimating unknown model parameters with numerical optimization algorithms.
Specifically, the parameter-estimation problem targeted is

```math
\min_{\mathbf{x} \in \mathbb{R}^N} \; -\ell(\mathbf{x})
\quad \text{subject to} \quad
\mathbf{lb} \le \mathbf{x} \le \mathbf{ub},
```

where `-ℓ(x)` is a negative log-likelihood (see [`PEtabObservable`](@ref) for mathematical
definition) and `lb`/`ub` are parameter bounds.

This extended tutorial covers PEtab.jl’s single-start and multi-start parameter estimation
workflows. It also shows how to construct an `OptimizationProblem` to solve the problem
using the Optimization.jl interface. As a running example, we use the Michaelis–Menten model
from the [starting tutorial](@ref tutorial).

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

# Observables
@parameters sigma
petab_obs1 = PEtabObservable(:petab_obs1, :obs1, 3.0)
petab_obs2 = PEtabObservable(:petab_obs2, :obs2, sigma)
observables = [petab_obs1, petab_obs2]

# Parameters to estimate
p_c1 = PEtabParameter(:c1)
p_c2 = PEtabParameter(:c2)
p_S0 = PEtabParameter(:S0)
p_sigma = PEtabParameter(:sigma)
pest = [p_c1, p_c2, p_S0, p_sigma]

# Measurements; simulate with 'true' parameters
using DataFrames, OrdinaryDiffEqRosenbrock
ps = [:c1 => 1.0, :c2 => 10.0, :c3 => 1.0, :S0 => 100.0]
u0 = [:S => 100.0, :E => 50.0, :SE => 0.0, :P => 0.0]
tspan = (0.0, 10.0)
oprob = ODEProblem(rn, u0, tspan, ps)
sol = solve(oprob, Rodas5P(); saveat = 0:0.5:10.0)
obs1 = (sol[:S] + sol[:E]) .+ randn(length(sol[:E]))
obs2   = sol[:P] .+ randn(length(sol[:P]))
df1 = DataFrame(obs_id = "petab_obs1", time = sol.t, measurement = obs1)
df2 = DataFrame(obs_id = "petab_obs2", time = sol.t, measurement = obs2)
measurements = vcat(df1, df2)

# Create the PEtabODEProblem
petab_model = PEtabModel(rn, observables, measurements, pest)
petab_prob = PEtabODEProblem(petab_model)
nothing # hide
```

## Single-start parameter estimation

Single-start parameter estimation runs a local optimizer from an initial parameter vector
`x0`. In PEtab.jl, parameters must be provided in the internal order expected by the
objective. The easiest way to construct a correctly ordered start vector is [`get_x`](@ref):

```@example 1
x0 = get_x(petab_prob)
```

`x0` is a `ComponentArray`, so parameters can be accessed by name. Names are also prefixed
by parameter scale, and by default parameters are estimated on a `log10` scale as it often
improves estimation performance [raue2013lessons, hass2019benchmark](@cite). Values should be modified on the estimation scale:

```@example 1
x0.log10_c1 = log10(10.0)
x0[:log10_c1] = log10(10.0) # alternatively
nothing # hide
```

Given a starting point, single-start parameter estimation can be performed with
[`calibrate`](@ref):

```@docs; canonical=false
calibrate
```

Algorithm recommendations are summarized on [Optimizer options](@ref options_optimizers).
Following those recommendations, we use Optim.jl’s interior-point Newton method:

```@example 1
import Optim
res = calibrate(petab_prob, x0, Optim.IPNewton())
```

`calibrate` returns a [`PEtabOptimisationResult`](@ref) containing the optimized parameters
and common diagnostics. To visualize the fit and other diagnostics, see [Plotting optimization results](@ref pest_plotting). For example, the model fit can be plotted with:

```@example 1
using Plots
default(left_margin=12.5Plots.Measures.mm, bottom_margin=12.5Plots.Measures.mm, size = (600*1.25, 400 * 1.25), palette = ["#CC79A7", "#009E73", "#0072B2", "#D55E00", "#999999", "#E69F00", "#56B4E9", "#F0E442"], linewidth=4.0) # hide
plot(res, petab_prob)
```

A major drawback of single-start estimation is that it may converge to a local minimum. A
better strategy for finding the global minimum is multi-start parameter estimation.

## [Multi-start parameter estimation](@id multistart_est)

Multi-start parameter estimation runs a local optimizer from multiple randomly sampled
starting points. This approach empirically performs well for ODE models
[raue2013lessons, persson2025petab](@cite).

The first step is to generate `n` starting points within the parameter bounds. Pure random
sampling tends to cluster, so PEtab.jl supports quasi-Monte Carlo sampling (Latin hypercube
by default), which typically gives more space-filling points that improves estimation performance [raue2013lessons](@cite):

```@example 1
using Distributions, QuasiMonteCarlo, Plots
import Random # hide
Random.seed!(123) # hide
lb, ub = [-1.0, -1.0], [1.0, 1.0]
s_uniform = QuasiMonteCarlo.sample(100, lb, ub, Uniform())
s_lhs     = QuasiMonteCarlo.sample(100, lb, ub, LatinHypercubeSample())
p1 = plot(s_uniform[1, :], s_uniform[2, :]; title="Uniform sampling", seriestype=:scatter, label=false)
p2 = plot(s_lhs[1, :], s_lhs[2, :]; title="Latin hypercube sampling", seriestype=:scatter, label=false)
plot(p1, p2)
plot(p1, p2; size=(800, 400)) # hide
```

For a `PEtabODEProblem`, hypercube start guesses (respecting bounds and parameter scales)
can be generated with [`get_startguesses`](@ref):

```@example 1
using StableRNGs
rng = StableRNG(42)
x0s = get_startguesses(rng, petab_prob, 50)
nothing # hide
```

Given `x0s`, multi-start estimation can be performed by calling [`calibrate`](@ref)
repeatedly. For convenience, `calibrate_multistart` combines start-guess generation and
estimation:

```@docs; canonical=false
calibrate_multistart
```

Two important keyword arguments to `calibrate_multistart` are `nprocs` and `dirsave`.
`nprocs` controls how many runs are executed in parallel (via `pmap` from Distributed.jl).
Using `nprocs > 1` often reduces wall-clock time for problems taking >5 minutes. `dirsave`
optionally saves results as runs finish and is **strongly recommended** to not loose
intermediate results. For example, to run 50 multi-starts with Optim.jl’s `IPNewton`:

```@example 1
ms_res = calibrate_multistart(petab_prob, Optim.IPNewton(), 50;
                              nprocs = 2,
                              dirsave = "path_to_save_directory")
```

Results are returned as a `PEtabMultistartResult`, which contains per-run statistics and the
best solution. Different ways to visualize results are found in
[Plotting optimization results](@ref pest_plotting). A common diagnostic is a waterfall
plot of the final objective values across runs:

```@example 1
plot(ms_res; plot_type=:waterfall)
```

Plateaus typically indicate distinct local optima (multiple runs converging to similar final
objective values). As another example, the best model fit can be plotted as:

```@example 1
plot(ms_res, petab_prob)
```

## Creating an OptimizationProblem

[Optimization.jl](https://github.com/SciML/Optimization.jl) provides a unified interface to
many optimization libraries (100+ solvers across 25+ packages). Since the Optimization.jl
ecosystem is still evolving, some PEtab.jl convenience features (e.g. plotting of
multi-start estimation results) are not yet available when using Optimization.jl directly.
We therefore recommend the PEtab.jl wrappers as the default workflow, but this is likely to
change as Optimization.jl matures.

A `PEtabODEProblem` can be converted directly into an `OptimizationProblem`:

```@docs; canonical=false
PEtab.OptimizationProblem
```

For our working example:

```@example 1
using Optimization
opt_prob = OptimizationProblem(petab_prob)
```

Given a start guess `x0`, parameters can be estimated with any supported solver. For
example, using Optim.jl’s `ParticleSwarm()` via `OptimizationOptimJL`:

```@example 1
using OptimizationOptimJL
opt_prob.u0 .= x0
res = solve(opt_prob, Optim.ParticleSwarm())
```

`solve` returns an `OptimizationSolution`. For solver options and how to work with the
returned solution object, see the Optimization.jl
[documentation](https://docs.sciml.ai/Optimization/stable/).

## Next steps

This extended tutorial showed how to estimate parameters with PEtab.jl. The parameter
estimation tutorials also covers:

- [Available optimization algorithms](@ref options_optimizers): Supported and recommended
  algorithms.
- [Plotting parameter estimation results](@ref pest_plotting): Plotting options for
  parameter estimation output.
- [Wrapping optimization packages](@ref wrap_est): How to use a `PEtabODEProblem` directly
  with an optimization package such as Optim.jl.
- [Model selection (PEtab-select)](@ref petab_select): Automatic model selection with
  PEtab-Select [pathirana2025petab](@cite).
- Litterature references: Good introductions on parameter estimation for ODE models can
  be found in [raue2013lessons, villaverde2022protocol](@cite).

## References

```@bibliography
Pages = ["extended_tutorial.md"]
Canonical = false
```
