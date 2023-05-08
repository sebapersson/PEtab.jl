# PEtab.jl

This is the documentation of [**PEtab.jl**](https://github.com/sebapersson/PEtab.jl), a Julia package designed to import ODE parameter estimation problems specified in the PEtab format into Julia.

PEtab.jl uses Julia's [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) package for ODE solvers and [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) for symbolic model processing, which enables fast model simulations. This, combined with support for gradients via forward- and adjoint-sensitivity approaches, and hessian via both exact and approximate methods, allows for efficient parameter estimation for both small and large models. In an extensive benchmark study, PEtab.jl was found to be 3-4 times faster than the [PyPesto](https://github.com/ICB-DCM/pyPESTO) toolbox that leverages the [AMICI](https://github.com/AMICI-dev/AMICI) interface to the Sundials suite.

This documentation includes:

* A guide for getting started with PEtab.jl
* Tutorials on medium-sized models, small models with several condition-specific parameters, models with pre-equilibration onditions (steady-state simulations), and how to define and import a model written in Julia.
* Details about available hessian and gradient options.
* Discussion of the best options for specific model types, including small, medium, and large models.

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

PEtab.jl provides a range of features to import and analyze ODE parameter estimation problems specified in the PEtab format. These include:

* Importing ODE systems specified either by an SBML file or as a Julia file.
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

## Citation

We will soon publish a preprint you can cite if you found PEtab.jl helpful in your work.
