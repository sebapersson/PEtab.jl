# PEtab.jl

This is the documentation of [**PEtab.jl**](https://github.com/sebapersson/PEtab.jl), a Julia package that imports ODE parameter estimation problem specified in the [PEtab](https://github.com/PEtab-dev/PEtab) format into Julia.

By leveraging the ODE solvers in Julia’s [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) package, and symbolic model processing via [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl), PEtab.jl achieves fast model simulations. This combined with support for gradients via forward- and adjoint-sensitivity approaches, and hessian via both exact and approximate methods means PEtab.jl provides everything needed to perform efficient parameter estimation for both big and small models. In an extensive benchmark study (soon to appear on Archive) PEtab.jl was in many cases 3-4 faster than the [PyPesto](https://github.com/ICB-DCM/pyPESTO) toolbox which leverages the [AMICI](https://github.com/AMICI-dev/AMICI) interface to the Sundial's suite.

In this documentation you can find:

* Getting started (a recommended introduction)
* Tutorials for medium-sized models, small models with several conditions specific parameters, and models with pre-equilibration conditions (steady-state simulations).
* Available hessian and gradient options.
* A discussion on best options for specific model types (small, medium and large models)

## Installation

To read SBML files PEtab.jl uses the Python [python-libsbml](https://pypi.org/project/python-libsbml/) library which can be installed via:

```
pip install python-libsbml
```

Given this PEtab.jl can be installed via

```julia
julia> ] add PEtab
```

## Feature list

* Import ODE systems specified either by an SBML file or a julia file.
* Symbolic model preprocessing via [ModelingToolkit](https://github.com/SciML/ModelingToolkit.jl)
* Support for all ODE solvers in [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl)
* Gradient via:
    * Forward-mode automatic differentiation using [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
    * Forward sensitivity analysis either via ForwardDiff.jl or [SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl)
    * Adjoint sensitivity via any of the algorithms in SciMLSensitivity.jl
    * Automatic differentiation via [Zygote.jl](https://github.com/FluxML/Zygote.jl)
* Hessians computed
    * “exactly" via forward-mode automatic differentiation using ForwardDiff.jl
    * approximately via a block approach using ForwardDiff.jl
    * approximately via the Gauss-Newton method (often outperform (L)-BFGS)
* Pre-equilibration and pre-simulation conditions
* Support for models with discrete events and logical operations

## Citation

We will soon publish a preprint you can cite if you found PEtab.jl helpful in your work.
