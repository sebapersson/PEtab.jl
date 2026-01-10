```@meta
CollapsedDocStrings=true
```

# [PEtab.jl API](@id API)

## PEtabModel

A `PEtabModel` for model fitting can be imported from the
[PEtab standard format](https://petab.readthedocs.io/en/latest/), or defined directly in
Julia. In the Julia interface, observables that link model outputs to data are specified
with `PEtabObservable`, estimated parameters with `PEtabParameter`, simulation conditions
with `PEtabConditions`, and events/callbacks with `PEtabEvent`.

```@docs
PEtabObservable
PEtabParameter
PEtabCondition
PEtabEvent
```

Given a dynamic model (as a `ReactionSystem` or `ODESystem`), measurement data as a
`DataFrame`, and optional simulation conditions, a `PEtabModel` can be created with:

```@docs
PEtabModel
```

## PEtabODEProblem

From a `PEtabModel`, a `PEtabODEProblem` can be created with:

```@docs
PEtabODEProblem
```

A detailed overview of problem size and configuration is available via:

```@docs
describe(::PEtabODEProblem)
```

A `PEtabODEProblem` has many configurable options. Two important options are the `ODESolver`
and, for problems with steady-state simulations, the `SteadyStateSolver`:

```@docs
ODESolver
SteadyStateSolver
```

Utility functions for interacting with a `PEtabODEProblem` (including efficient subsetting
via `remake`) are:

```@docs
get_x
remake(::PEtabODEProblem)
```

Utilities for interacting with the underlying dynamic model (`ODEProblem`) are:

```@docs
get_ps
get_u0
get_system
get_odeproblem
get_odesol
solve_all_conditions
```

## Parameter estimation

A `PEtabODEProblem` contains everything needed to use an optimizer directly, but PEtab.jl
also provides convenience wrappers. For single-start parameter estimation:

```@docs
calibrate
PEtabOptimisationResult
```

For multi-start parameter estimation:

```@docs
calibrate_multistart
get_startguesses
PEtabMultistartResult
```

And finally model selection (PEtab-Select interface):

```@docs
petab_select
```

For calibration, optimizers from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl),
[Ipopt.jl](https://github.com/jump-dev/Ipopt.jl), and
[Fides.jl](https://fides-dev.github.io/Fides.jl/stable/). Ipopt-specific configuration is
available via:

```@docs
IpoptOptimizer
IpoptOptions
```

Parameter-estimation results can be visualized using the plotting recipes described on
[Plotting parameter estimation results](@ref pest_plotting), and with:

```@docs
get_obs_comparison_plots
```

As an alternative workflow, a `PEtabODEProblem` can be converted to an `OptimizationProblem`
to access solvers via [Optimization.jl](https://github.com/SciML/Optimization.jl):

```@docs
PEtab.OptimizationProblem
```

Parameter-estimation results for problems in the PEtab standard format can be exported with:

```@docs
export_petab
```

## Bayesian inference

PEtab.jl supports Bayesian inference by exposing a `PEtabLogDensity` compatible with the
`LogDensityProblems.jl` interface, which can be sampled with
[AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) (including NUTS) or
[AdaptiveMCMC.jl](https://github.com/mvihola/AdaptiveMCMC.jl). This functionality is planned
to move into a separate package, so the API will change in the future.

```@docs
PEtabLogDensity
to_prior_scale
to_chains
```
