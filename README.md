# PEtab.jl
*Create parameter estimation problems for dynamic models*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sebapersson.github.io/PEtab.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sebapersson.github.io/PEtab.jl/dev/)
[![Build Status](https://github.com/sebapersson/PEtab.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sebapersson/PEtab.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![codecov](https://codecov.io/gh/sebapersson/PEtab.jl/graph/badge.svg?token=J7PXRF30JG)](https://codecov.io/gh/sebapersson/PEtab.jl)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

PEtab.jl is a Julia package for creating parameter estimation problems for fitting Ordinary Differential Equation (ODE) models to data in Julia. Some major highlights of PEtab.jl are:

* It supports coding parameter estimation problems directly in Julia, where the dynamic model can be provided as a [Catalyst](https://github.com/SciML/Catalyst.jl) `ReactionSystem`, a [ModelingToolkit](https://github.com/SciML/ModelingToolkit.jl) `ODESystem`, or as an [SBML](https://sbml.org/) file imported through [SBMLImporter](https://github.com/sebapersson/SBMLImporter.jl).
* It can import and has full support for parameter estimation problems in the [PEtab](https://petab.readthedocs.io/en/latest/) standard format
* It supports a wide range of features for parameter estimation problems, including multiple observables, multiple simulation conditions, models with events, and models with steady-state pre-equilibration simulations.
* It integrates with Julia's [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) ecosystem, which among other things, means it supports any of the state-of-the-art ODE solvers in [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl).
* It supports efficient forward and adjoint gradient methods, suitable for small and large models, respectively.
* It supports exact Hessian's for small models and good approximations for large models.
* It includes wrappers for performing parameter estimation with optimization packages [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [Ipopt](https://coin-or.github.io/Ipopt/), [Optimization.jl](https://github.com/SciML/Optimization.jl), and [Fides.py](https://github.com/fides-dev/fides).
* It includes wrappers for performing Bayesian inference using state-of-the-art methods such as [NUTS](https://github.com/TuringLang/Turing.jl) (the same sampler used in [Turing.jl](https://github.com/TuringLang/Turing.jl)) or [AdaptiveMCMC.jl](https://github.com/mvihola/AdaptiveMCMC.jl).

Additional information and tutorials can be found in the [documentation](https://sebapersson.github.io/PEtab.jl/stable/).

## Installation

PEtab.jl is a registered Julia package and can be installed with the Julia package manager using:

```julia
julia> import Pkg; Pkg.add("PEtab")
```

PEtab.jl is compatible with Julia 1.10 and above. For additional installation details, see the [documentation](https://sebapersson.github.io/PEtab.jl/stable/#Installation).

## Citation

We will soon publish a paper you can cite if you found PEtab helpful in your work.
