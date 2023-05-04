# PEtab.jl
*Importer for systems biological models defined in the PEtab format.*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sebapersson.github.io/PEtab.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sebapersson.github.io/PEtab.jl/dev/)
[![Build Status](https://github.com/sebapersson/PEtab.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sebapersson/PEtab.jl/actions/workflows/CI.yml?query=branch%3Amain)

PEtab.jl imports ODE parameter estimation problem specified in the [PEtab](https://github.com/PEtab-dev/PEtab) format into Julia. By leveraging the ODE solvers in Julia's [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) package, and symbolic model processing via [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) PEtab.jl achieves fast model simulations. This combined with support for gradients via forward- and adjoint-sensitivity approaches, and hessian via both exact and approximate methods means PEtab.jl provides everything needed to perform efficient parameter estimation for both big and small models. In an extensive benchmark study PEtab.jl was typically 3-5 faster than the [PyPesto](https://github.com/ICB-DCM/pyPESTO) toolbox which leverages the [AMICI](https://github.com/AMICI-dev/AMICI) interface to the Sundials suite.

For installation see below. For a starting example see below and the *examples* folder. An extensive list of all options and recommended settings for different model sizes can be found in the documentation.

## Installation

To read SBML files PEtab.jl uses the Python [python-libsbml](https://pypi.org/project/python-libsbml/) library which can be installed via:

```
pip install python-libsbml
```

Given this PEtab.jl can be installed via

```julia
julia> ] add PEtab
```

## Quick start

Given the path to the PEtab YAML-file for the so-called [Bachmann model](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/tree/master/Benchmark-Models/Bachmann_MSB2011) as a string `yamlPath` a PEtab model can be created via;

```julia
using PEtab
bachmannModel = readPEtabModel(yamlPath)
```

`readPEtabModel` translates the SBML ODE model into [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) which, among other things, allows for symbolic computations of the Jacobian. Moreover, in this step PEtab-files are parsed to generate functions for computing initial values, observation functions, and events.

Given a `PEtabModel` the user can create a `PEtabODEProblem` to compute the cost, gradient and hessian of a model. Say we want to solve the model ODE using the [QNDF](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/) solver, compute the gradient via [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) and approximate the hessian via the Gauss-Newton method;

```julia
using OrdinaryDiffEq
solverOptions = ODESolverOptions(QNDF())
petabProblem = setupPEtabODEProblem(bachmannModel, solverOptions, 
                                    gradientMethod=:ForwardDiff, 
                                    hessianMethod=:GaussNewton)
```

Given a parameter vector θ the cost, gradient and hessian can now be computed via
```julia
# We only support in-place Hessian and gradients
∇f = zeros(length(θ))
H = zeros(length(θ), length(θ))

f = petabProblem.computeCost(θ)
petabProblem.computeGradient!(∇f, θ)
petabProblem.computeHessian!(H, θ)
```

With this functionality it is straightforward to perform parameter estimation using optimization packages such as [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl) and [Fides.py](https://github.com/fides-dev/fides). In the *examples* folder we provide code for how to use these packages together with PEtab.jl. We also cover recommended `PEtabODEProblem` settings small models (*examples/Boehm.jl*), a medium-sized model (*examples/Bachmann.jl*), a small model where most of the parameters are specific to one of the simulation conditions (*examples/Beer.jl*), and a model with pre-equilibration (steady-state simulation) conditions (*examples/Brannmark.jl*).

## Features

* Symbolic model pre-processing via [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl)
* Support for all ODE solvers in [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl)
* Gradient via:
    * Forward-mode automatic differentiation using [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
    * Forward sensitivity analysis either via ForwardDiff.jl or [SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl)
    * Adjoint sensitivity via any of the algorithms in SciMLSensitivity.jl
    * Automatic differentiation via [Zygote.jl](https://github.com/FluxML/Zygote.jl)
* Hessians computed
    * "exactly" via forward-mode automatic differentiation using ForwardDiff.jl
    * approximately via a block approach using ForwardDiff.jl
    * approximately via the Gauss-Newton method (often outperform (L)-BFGS)
* Pre-equilibration and pre-simulation conditions
* Support for models with discrete events and logical operations


## Development team

This package was originally developed by [Rafael Arutjunjan](https://github.com/RafaelArutjunjan), [Sebastian Persson](https://github.com/sebapersson), [Damiano Ognissanti](https://github.com/damianoognissanti) and [Viktor Hasselgren](https://github.com/CleonII).
