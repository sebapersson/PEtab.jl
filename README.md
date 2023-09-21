# PEtab.jl
*Create parameter estimation problems for ODE models*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sebapersson.github.io/PEtab.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sebapersson.github.io/PEtab.jl/dev/)
[![Build Status](https://github.com/sebapersson/PEtab.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sebapersson/PEtab.jl/actions/workflows/CI.yml?query=branch%3Amain)

PEtab.jl is a Julia package to setup ODE parameter estimation problems in Julia. Parameter estimation problems can be directly imported if they are specified in the [PEtab](https://petab.readthedocs.io/en/latest/) standard format, alternatively problems can be directly specifed in Julia where the dynamic model can be provided as a [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) ODE-system or a [Catalyst](https://github.com/SciML/Catalyst.jl) reaction system. Once a problem has been parsed PEtab.jl provides wrappers to Optim, Ipopt, and Fides to perform efficient multi-start parameter estimation.

In an extensive benchmark study, PEtab.jl was found to be 2-4 times faster than the [pyPESTO](https://github.com/ICB-DCM/pyPESTO) toolbox that leverages the [AMICI](https://github.com/AMICI-dev/AMICI) interface to the Sundials suite. For installation instructions, refer below. For additional details, the documentation contains a comprehensive list of all options and recommended settings for different model sizes.

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

After defining the parameter vector `θ`, you can compute the cost, gradient, and Hessian with:

```julia
θ = petab_problem.θ_nominalT
∇f = zeros(length(θ))
H = zeros(length(θ), length(θ))

f = petab_problem.compute_cost(θ)
petab_problem.compute_gradient!(∇f, θ)
petab_problem.compute_hessian!(H, θ)
```

With a `PEtabODEProblem` it is also straighforward to calibrate the model to data with multistart optimization. For example, to use the [Ipopt](https://github.com/jump-dev/Ipopt.jl) optimizer with 10 Latin-Hypercube sampled start-guesses, and to save the results in `dir_save`, write:

```julia
alg = IpoptOptimiser(false)
res = calibrate_model_multistart(petab_problem, alg, 10, dir_save)
```

You can explore additional examples and available optimizers for parameter estimation in the documentation and within the *examples* folder. The examples cover a range of scenarios, including recommended `PEtabODEProblem` settings for small models (*examples/Boehm.jl*), medium-sized models (*examples/Bachmann.jl*), models with parameters specific to certain simulation conditions (*examples/Beer.jl*), and models with pre-equilibration (steady-state simulation) conditions (*examples/Brannmark.jl*).

## Features

* Importing ODE systems specified either by an SBML file.
* Support for models defined in Julia as ODE-systems or as [Catalyst](https://github.com/SciML/Catalyst.jl) reaction systems.
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
