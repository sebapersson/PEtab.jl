# PEtab.jl
*Importer for systems biological models defined in the PEtab format.*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sebapersson.github.io/PEtab.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sebapersson.github.io/PEtab.jl/dev/)
[![Build Status](https://github.com/sebapersson/PEtab.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sebapersson/PEtab.jl/actions/workflows/CI.yml?query=branch%3Amain)

PEtab.jl is a Julia package that imports ODE parameter estimation problems specified in the PEtab format. PEtab.jl uses Julia's [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) package for ODE solvers and [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) for symbolic model processing, which combined enables fast model simulations. Additionally, it supports gradients using both forward and adjoint sensitivity approaches, as well as Hessians via exact and approximate methods, making it suitable for efficient parameter estimation for models of various sizes. In an extensive benchmark, PEtab.jl was frequently 2-4 times faster than the pyPESTO toolbox, which utilizes the AMICI interface to the Sundials suite.

For installation instructions, refer below. For additional details, the documentation contains a comprehensive list of all options and recommended settings for different model sizes.

## Installation

PEtab.jl can be installed via

```julia
julia> ] add PEtab
```

or alternatively via

```julia
julia> using Pkg; Pkg.add("PEtab")
```

## Quick start

Given a PEtab problem the first step is to read it into a PEtab model, which for the so-called Bachmann model can be done via:

```julia
using PEtab
bachmann_model = PEtabModel(yaml_path)
```

`PEtabModel` translates the SBML ODE model into [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) format, which allows for symbolic computations of the Jacobian. Additionally, it generates functions for computing initial values, observation functions, and events from the PEtab-files.

With a `PEtabModel`, you can create a `PEtabODEProblem` to compute the cost, gradient, and Hessian of the model. For instance, you can solve the model ODE using the [QNDF](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/) solver, compute the gradient via [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl), and approximate the Hessian via the Gauss-Newton method with the following code:

```julia
using OrdinaryDiffEq
petab_problem = PEtabODEProblem(bachmann_model, 
                                ode_solver=ODESolver(QNDF()), 
                                gradient_method=:ForwardDiff, 
                                hessian_method=:GaussNewton)
```

After defining the parameter vector `θ`, you can compute the cost, gradient, and Hessian with the following code:

```julia
θ = petab_problem.θ_nominalT
∇f = zeros(length(θ))
H = zeros(length(θ), length(θ))

f = petab_problem.compute_cost(θ)
petab_problem.compute_gradient!(∇f, θ)
petab_problem.compute_hessian!(H, θ)
```

Using this functionality, you can perform parameter estimation using optimization packages such as [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl), and [Fides.py](https://github.com/fides-dev/fides). In the *examples* folder, you can find code for using these packages together with PEtab.jl. The *examples* folder also includes recommended `PEtabODEProblem` settings for small models (*examples/Boehm.jl*), a medium-sized model (*examples/Bachmann.jl*), a small model where most of the parameters are specific to one of the simulation conditions (*examples/Beer.jl*), and a model with pre-equilibration (steady-state simulation) conditions (*examples/Brannmark.jl*).

## Features

* Importing ODE systems specified either by an SBML file or as a Julia file.
* Model selection via [PEtab Select](https://github.com/PEtab-dev/petab_select).
* Symbolic model pre-processing via ModelingToolkit.jl.
* Support for all ODE solvers in DifferentialEquations.jl.
* Gradient calculations using several approaches:
    * Forward-mode automatic differentiation with ForwardDiff.jl.
    * Forward sensitivity analysis with ForwardDiff.jl or SciMLSensitivity.jl.
    * Adjoint sensitivity analysis with any of the algorithms in SciMLSensitivity.jl.
    * Automatic differentiation via Zygote.jl.
* Hessians computed via:
    * Forward-mode automatic differentiation with ForwardDiff.jl (exact).
    * Block approach with ForwardDiff.jl (approximate).
    * Gauss-Newton method (approximate and often more performant than (L)-BFGS).
* Handling pre-equilibration and pre-simulation conditions.
* Support for models with discrete events and logical operations.

## Documentation

Documentation and tutorials are available [here](https://sebapersson.github.io/PEtab.jl)

## Development team

This package was originally developed by [Damiano Ognissanti](https://github.com/damianoognissanti), [Rafael Arutjunjan](https://github.com/RafaelArutjunjan), [Sebastian Persson](https://github.com/sebapersson) and [Viktor Hasselgren](https://github.com/CleonII).
