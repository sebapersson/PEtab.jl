# PEtab.jl
*Create parameter estimation problems for ODE models*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sebapersson.github.io/PEtab.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sebapersson.github.io/PEtab.jl/dev/)
[![Build Status](https://github.com/sebapersson/PEtab.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sebapersson/PEtab.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![codecov](https://codecov.io/gh/sebapersson/PEtab.jl/graph/badge.svg?token=J7PXRF30JG)](https://codecov.io/gh/sebapersson/PEtab.jl)

PEtab.jl is a Julia package to create ODE parameter estimation problems in Julia. Parameter estimation problems can be directly imported if they are specified in the [PEtab](https://petab.readthedocs.io/en/latest/) standard format, alternatively problems can be directly specified in Julia where the dynamic model can be provided as a [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) `ODESystem` or a [Catalyst](https://github.com/SciML/Catalyst.jl) `ReactionSystem`. Once a problem has been parsed PEtab.jl provides wrappers to Optim, Optimization, Ipopt, and Fides to perform efficient multi-start parameter estimation.

The [SBMLImporter.jl](https://github.com/sebapersson/SBMLImporter.jl) package is used to import SBML models when parameter estimation problems are given in the PEtab standard format. If your goal is solely to import and simulate SBML models, SBMLImporter offers extensive support for various SBML features.

For installation instructions and a quick start see below. For additional details, the [documentation](https://sebapersson.github.io/PEtab.jl/stable/) contains a comprehensive list of all options and recommended settings for different model sizes.

## Installation

To install PEtab.jl in Julia in the Julia REPL enter

```julia
julia> ] add PEtab
```

or enter

```julia
julia> using Pkg; Pkg.add("PEtab")
```

PEtab.jl is compatible with Julia version 1.6 and above. However, for best performance, we strongly recommend using Julia version 1.10.

## Quick start

For how to define a parameter estimation problem directly in Julia, see the [documentation](https://sebapersson.github.io/PEtab.jl/stable/Define_in_julia/).

If the parameter estimation problem is provided in the [PEtab standard format](https://petab.readthedocs.io/en/latest/), importing it is straightforward. Consider the Boehm model (the model files can be downloaded from [here](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/tree/master/Benchmark-Models/Boehm_JProteomeRes2014)), given the path to the PEtab YAML file, the problem can be imported via:

```julia
using PEtab
boehm_model = PEtabModel(yaml_path)
```

`PEtabModel` converts the SBML ODE model into a format compatible with [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl), which with other things allows for symbolic Jacobian computations. It also automatically generates functions for computing initial values, observation functions, and events based on the data in the PEtab files.

With a `PEtabModel` a `PEtabODEProblem` can be created. This enables the computation of cost, gradient, and Hessian for the parameter estimation problem. For instance, to simulate the ODE with the [Rodas5](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/) solver, to compute the gradient with [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl), and to approximate the Hessian with the Gauss-Newton method do:

```julia
using OrdinaryDiffEq
petab_problem = PEtabODEProblem(boehm_model,
                                ode_solver=ODESolver(Rodas5()),
                                gradient_method=:ForwardDiff,
                                hessian_method=:GaussNewton)
```

If no ODE-solver, gradient and/or Hessian method are provided, good defaults are used. To compute the cost, gradient, and Hessian, first define the parameter vector `x`, then initialize the gradient and Hessian matrices:

```julia
x = petab_problem.θ_nominalT
∇f = zeros(length(x))
H = zeros(length(x), length(x))

f = petab_problem.compute_cost(x)
petab_problem.compute_gradient!(∇f, x)
petab_problem.compute_hessian!(H, x)
```

Calibrating the model to data with multistart optimization is also straightforward. For example, to use the [Ipopt](https://github.com/jump-dev/Ipopt.jl) optimizer with 10 Latin-Hypercube sampled start guesses and to save the results in `dir_save` do:

```julia
using Ipopt
alg = IpoptOptimiser(false)
res = calibrate_model_multistart(petab_problem, alg, 10, dir_save)
```

For additional examples and a list of available optimizers for parameter estimation, see the [documentation](https://sebapersson.github.io/PEtab.jl/stable/).

## Features

* Full [PEtab](https://github.com/PEtab-dev/PEtab) support, which with other things include:
    * Support for multiple observables.
    * Support for multiple simulation conditions.
    * Suport for pre-equilibration (steady-state simulations).
    * Support for parameter specific to a simulation condition.
* Ability to import ODE systems defined in SBML files.
* Support for models created in Julia, either as a [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) ODE-system or as [Catalyst](https://github.com/SciML/Catalyst.jl) reaction systems.
* Model selection via [PEtab Select](https://github.com/PEtab-dev/petab_select).
* Symbolic model pre-processing using ModelingToolkit.jl.
* Compatibility with all ODE solvers in DifferentialEquations.jl.
* Several options for computing gradients:
    * Forward-mode automatic differentiation with ForwardDiff.jl.
    * Forward sensitivity analysis using ForwardDiff.jl or SciMLSensitivity.jl.
    * Adjoint sensitivity analysis with algorithms from SciMLSensitivity.jl.
    * Automatic differentiation via Zygote.jl.
* Several options for computing Hessians:
    * Exact calculation using Forward-mode automatic differentiation with ForwardDiff.jl.
    * Approximate block approach with ForwardDiff.jl.
    * Gauss-Newton method, which is often more performant than (L)-BFGS.
* Support for models incorporating discrete events and logical operations.

## Development team

This package was originally developed by [Damiano Ognissanti](https://github.com/damianoognissanti), [Rafael Arutjunjan](https://github.com/RafaelArutjunjan), [Sebastian Persson](https://github.com/sebapersson) and [Viktor Hasselgren](https://github.com/CleonII).
