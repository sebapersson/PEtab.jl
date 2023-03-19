# PEtab.jl
*Importer for systems biological models defined in the PEtab format.*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sebapersson.github.io/PEtab.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sebapersson.github.io/PEtab.jl/dev/)
[![Build Status](https://github.com/sebapersson/PEtab.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sebapersson/PEtab.jl/actions/workflows/CI.yml?query=branch%3Amain)

PEtab.jl imports ODE parameter estimation problems specified in the [PEtab](https://github.com/PEtab-dev/PEtab) format into Julia. The importer provides everything needed to call available optimizers such as Optim.jl. It supports several gradient and hessian options for both big and small model, and is overall written with performance in mind. In an extensive parameter estimation benchmark PEtab.jl typically was 3-5 faster than the PyPesto toolbox.

For an extensive list of all options see the documentation. For a starting example see below.

## Installation

PEtab.jl uses the Python [python-libsbml](https://pypi.org/project/python-libsbml/) to read SBML models which can be installed via:

```
pip install python-libsbml
```

Given this PEtab.jl can be installed via

```julia
julia> ] add PEtab
```

## Quick start

Given the path to PEtab YAML-file for the so-called [Bachmann model](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/tree/master/Benchmark-Models/Bachmann_MSB2011) as a string `yamlPath` a PEtab model can be created via;

```julia
using PEtab
bachmannModel = readPEtabModel("yamlPath")
```

`readPEtabModel` translates the SBML ODE model into [ModelingToolkit](https://github.com/SciML/ModelingToolkit.jl) which with other things allows for symbolic computations of the Jacobian. Moreover, the PEtab-files are processed to create jl-files to efficiently compute the initial values, observation functions and events.

Given a `PEtabModel` the user can create a `PEtabODEProblem` to compute the cost, gradient and hessian of a model. Say we want to solve the model ODE using the [QNDF](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/) solver, and we want to compute the gradient via [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) and approximate the hessian via the Gauss-Newton method. All this is possible via;

```julia
using OrdinaryDiffEq
solverOptions = getODESolverOptions(QNDF())
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

Given functions to efficiently compute the cost, gradient and hessian it is straightforward to perform parameter estimation using optimization packages such as [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl) and [Fides.py](https://github.com/fides-dev/fides). In the *examples* folder we provide code for how to use these packages for parameter estimation. We also cover how to best set up a `PEtabODEProblem` for a small model (*examples/Boehm.jl*), a medium-sized model (*examples/Bachmann.jl*) and a small model where most of the parameters are specific to one of the simulation conditions (*examples/Beer.jl*).

## Gradient and hessian options

PEtab.jl supports several gradient options.

|  gradientMethod | Description | Potential user options  |
|---|---|---|
|  :ForwardDiff | Compute the gradient via [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) | Number of chunks |
|  :ForwardEquations | Compute the gradient via by solving the expanded ODE system to obtain the models sensitives | If the sensitives should be computed directory via [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl), or by solving the expanded ODE system |
|  :Adjoint | Compute the gradient via adjoint sensitivity analysis | Whether to use `InterpolatingAdjoint` or `QuadratureAdjoint` provided by [SciMLSensitivity](https://github.com/SciML/SciMLSensitivity.jl) |
|  :Zygote | Compute the gradient using the [Zygote](https://github.com/FluxML/Zygote.jl) automatic differentiation library  | Any of the sensealg provided by [SciMLSensitivity](https://github.com/SciML/SciMLSensitivity.jl) |

The user can also choose to provide a specific ODE solver for the gradient. PEtab.jl also supports three Hessian options:

|  hessianMethod | Description | Potential user options  |
|---|---|---|
|  :ForwardDiff | Compute the Hessian via [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) | Number of chunks |
|  :BlockForwardDiff | Compute a Hessian block approximation via [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) (for details see)  | Number of chunks |
|  :GaussNewton | Compute a Hessian approximation via the Gauss-Newton method. Often performs better than a (L)-BFGS approximation | none |

More details about the possible options, and when each option is suitable can be found in the documentation.

## Development team

This package was originally developed by [Rafael Arutjunjan](https://github.com/RafaelArutjunjan), [Sebastian Persson](https://github.com/sebapersson) and [Viktor Hasselgren](https://github.com/CleonII).