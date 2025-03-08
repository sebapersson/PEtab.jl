```@meta
CollapsedDocStrings=true
```

# [PEtab.jl API](@id API)

## PEtabModel

A `PEtabModel` for parameter estimation/inference can be created by importing a PEtab parameter estimation problem in the [standard format](https://petab.readthedocs.io/en/latest/), or it can be directly defined in Julia. For the latter, observables that link the model to measurement data are provided by `PEtabObservable`, parameters to estimate are defined by `PEtabParameter`, and any potential events (callbacks) are specified as `PEtabEvent`.

```@docs
PEtabObservable
PEtabParameter
PEtabEvent
```

Then, given a dynamic model (as `ReactionSystem` or `ODESystem`), measurement data as a `DataFrame`, and potential simulation conditions as a `Dict` (see [this](@ref petab_sim_cond) tutorial), a `PEtabModel` can be created:

```@docs
PEtabModel
```

## PEtabODEProblem

From a `PEtabModel`, a `PEtabODEProblem` can:

```@docs
PEtabODEProblem
```

A `PEtabODEProblem` has numerous configurable options. Two of the most important options are the `ODESolver` and, for models with steady-state simulations, the `SteadyStateSolver`:

```@docs
ODESolver
SteadyStateSolver
```

PEtab.jl provides several functions for interacting with a `PEtabODEProblem`:

```@docs
get_x
remake(::PEtabODEProblem, ::Dict)
```

And additionally, functions for interacting with the underlying dynamic model (`ODEProblem`) within a `PEtabODEProblem`:

```@docs
get_u0
get_ps
get_system
get_odeproblem
get_odesol
solve_all_conditions
```

## Parameter Estimation

A `PEtabODEProblem` contains all the necessary information for wrapping a suitable numerical optimization library, but for convenience, PEtab.jl provides wrappers for several available optimizers. In particular, single-start parameter estimation is provided via `calibrate`:

```@docs
calibrate
PEtabOptimisationResult
```

Multi-start (recommended method) parameter estimation, is provided via `calibrate_multistart`:

```@docs
calibrate_multistart
get_startguesses
PEtabMultistartResult
```

Lastly, model selection is provided via `petab_select`:

```@docs
petab_select
```

For each case case, PEtab.jl supports the usage of optimization algorithms from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl), and [Fides.py](https://github.com/fides-dev/fides):

```@docs
Fides
IpoptOptimizer
IpoptOptions
```

Parameter estimation results can be visualized using the plot-recipes detailed in [this](@ref optimization_output_plotting) page, and with `get_obs_comparison_plots`:

```@docs
get_obs_comparison_plots
```

As an alternative to the PEtab.jl interface to parameter estimation, a `PEtabODEProblem` can be converted to an `OptimizationProblem` to access the algorithms available via [Optimization.jl](https://github.com/SciML/Optimization.jl):

```@docs
PEtab.OptimizationProblem
```

## Bayesian Inference

PEtab.jl offers wrappers to perform Bayesian inference using state-of-the-art methods such as [NUTS](https://github.com/TuringLang/Turing.jl) (the same sampler used in [Turing.jl](https://github.com/TuringLang/Turing.jl)) or [AdaptiveMCMC.jl](https://github.com/mvihola/AdaptiveMCMC.jl). It should be noted that this part of PEtab.jl is planned to be moved to a separate package, so the syntax will change and be made more user-friendly in the future.

```@docs
PEtabLogDensity
to_prior_scale
to_chains
```
