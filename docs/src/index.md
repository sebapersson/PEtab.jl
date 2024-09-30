# PEtab.jl

PEtab.jl is a Julia package for creating parameter estimation problems for fitting Ordinary Differential Equation (ODE) models to data in Julia.

## Major highlights

* Support for coding parameter estimation problems directly in Julia, where the dynamic model can be provided as a [Catalyst.jl](https://github.com/SciML/Catalyst.jl) `ReactionSystem`, a [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) `ODESystem`, or as an [SBML](https://sbml.org/) file imported via [SBMLImporter.jl](https://github.com/sebapersson/SBMLImporter.jl).
* Direct import and full support for parameter estimation problems in the [PEtab](https://petab.readthedocs.io/en/latest/) standard format
* Support for a wide range of parameter estimation problem features, including multiple observables, multiple simulation conditions, models with events, and models with steady-state pre-equilibration simulations.
* Integration with Julia's [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) ecosystem, which, among other things, means PEtab.jl supports the state-of-the-art ODE solvers in [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl). Consequently, PEtab.jl is suitable for both stiff and non-stiff ODE models.
* Support for efficient forward and adjoint gradient methods, suitable for small and large models, respectively.
* Support for exact Hessian's for small models and good approximations for large models.
* Includes wrappers for performing parameter estimation with optimization packages [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [Ipopt](https://coin-or.github.io/Ipopt/), [Optimization.jl](https://github.com/SciML/Optimization.jl), and [Fides.py](https://github.com/fides-dev/fides).
* Includes wrappers for performing Bayesian inference with the state-of-the-art NUTS sampler (the same sampler used in [Turing.jl](https://github.com/TuringLang/Turing.jl)) or with [AdaptiveMCMC.jl](https://github.com/mvihola/AdaptiveMCMC.jl).

!!! note "Star us on GitHub!"
    If you find the package useful in your work please consider giving us a star on [GitHub](https://github.com/sebapersson/PEtab.jl). This will help us secure funding in the future to continue maintaining the package.

!!! tip "Latest news: PEtab.jl v3.0"
    Version 3.0 is a breaking release that added support for ModelingToolkit v9 and Catalyst v14. Along with updating these packages, PEtab.jl underwent a major update, with new functionality added as well as the renaming of several functions to be more consistent with the naming convention in the SciML ecosystem. See the [HISTORY] file for more details.

## Installation

To install PEtab.jl in the Julia REPL enter

```julia
julia> ] add PEtab
```

or alternatively

```julia
julia> using Pkg; Pkg.add("PEtab")
```

PEtab is compatible with Julia version 1.9 and above. For best performance we strongly recommend using the latest Julia version, which most easily can be installed with [juliaup](https://github.com/JuliaLang/juliaup).

## Getting help

If you have any problems using PEtab, here are some helpful tips:

* Read the [FAQ](@ref FAQ) section in the online documentation.
* Post your questions in the `#sciml-sysbio` channel on the [Julia Slack](https://julialang.org/slack/). While PEtab.jl is not exclusively for systems biology, the `#sciml-sysbio` channel is where the package authors are most active.
* If you have encountered unexpected behavior or a bug, please open an issue on [GitHub](https://github.com/sebapersson/PEtab.jl/issues).
