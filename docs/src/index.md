# PEtab.jl

PEtab is a Julia package for setting up parameter estimation problems for fitting Ordinary Differential Equation (ODE) models to data in Julia. 

## Major highlights

* It supports coding parameter estimation problems directly in Julia, where the dynamic model can be provided as a [Catalyst](https://github.com/SciML/Catalyst.jl) `ReactionSystem`, a [ModelingToolkit](https://github.com/SciML/ModelingToolkit.jl) `ODESystem`, or as an [SBML](https://sbml.org/) file importer through [SBMLImporter](https://github.com/sebapersson/SBMLImporter.jl).
* It can directly import parameter estimation problems encoded in the [PEtab](https://petab.readthedocs.io/en/latest/) standard format and has full support for the PEtab standard.
* It supports a wide range of features for parameter estimation problems, including multiple observables, multiple simulation conditions, models with events, and models with steady-state pre-equilibration simulations.
* It integrates with Julia's [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) ecosystem, which among other things, means it supports any of the state-of-the-art ODE solvers in [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl).
* It supports efficient forward and adjoint gradient methods, suitable for small and large models, respectively.
* It supports exact Hessian calculations for small models and good Hessian approximations for large models.
* It includes wrappers for performing parameter estimation using efficient optimization packages such as [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [Ipopt](https://coin-or.github.io/Ipopt/), [Optimization.jl](https://github.com/SciML/Optimization.jl), and [Fides.py](https://github.com/fides-dev/fides).
* It includes wrappers for performing Bayesian inference using state-of-the-art methods such as [NUTS](https://github.com/TuringLang/Turing.jl) (the same sampler used in [Turing.jl](https://github.com/TuringLang/Turing.jl)) or [AdaptiveMCMC.jl](https://github.com/mvihola/AdaptiveMCMC.jl).

!!! note
    If you find the package useful in your work please consider giving us a star on [GitHub](https://github.com/sebapersson/PEtab.jl). This will help us secure funding in the future to continue maintaining the package.

## Installation

To install PEtab.jl in the Julia REPL enter

```julia
julia> ] add PEtab
```

or alternatively

```julia
julia> using Pkg; Pkg.add("PEtab")
```

PEtab is compatible with Julia version 1.9 and above. For best performance we strongly recommend using the latest Julia version.

## Getting help

If you have any problems using PEtab, here are some helpful tips:

* Read the [FAQ](@ref FAQ) section in the online documentation.
* Post your questions in the `#sciml-sysbio` channel on the [Julia Slack](https://julialang.org/slack/).
* If you believe you have encountered unexpected behavior or a bug, please open an issue on [GitHub](https://github.com/sebapersson/PEtab.jl/issues).
