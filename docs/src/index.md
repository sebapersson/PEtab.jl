# PEtab.jl

PEtab.jl is a Julia package for creating ODE parameter estimation problems in Julia. It uses Julia's [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) package for ODE solvers and [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) for symbolic model processing, which enables fast model simulations. This, combined with support for gradients via forward- and adjoint-sensitivity approaches, and hessian via both exact and approximate methods, allows for efficient parameter estimation for both small and large models.

Parameter estimation problems can be directly imported if they are specified in the [PEtab](https://petab.readthedocs.io/en/latest/) standard format, alternatively problems can be directly specified in Julia where the dynamic model can be provided as a [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) ODE-system or a [Catalyst](https://github.com/SciML/Catalyst.jl) reaction system. Once a problem has been parsed PEtab.jl provides wrappers to Optim, Ipopt, and Fides to perform efficient multi-start parameter estimation.

In this documentation you will find:

* How to import a problem provided in the PEtab standard format, see [here](@ref import_petab_problem).
* How to specify a parameter estimation problem directly in Julia, see [here](@ref define_in_julia), with additional information on how to setup problems with:
    * [Pre-Equilibration](@ref define_with_ss) (Steady-State Simulations)
    * [Noise and Observable Parameters](@ref time_point_parameters)
    * [Condition specific parameters](@ref define_conditions)
    * [Events](@ref define_events) (callbacks, dosages etc...)
* Available options for various specific problem types (e.g. models with steady-state simulations), see Select options for a PEtab problem.
* Recommended ODE-solvers, gradient, and Hessian methods for a parameter estimation problem, see [Choosing the best options for a PEtab problem](@ref best_options).
* How to perform efficient parameter estimation for a PEtab problem, see [Parameter estimation](@ref parameter_estimation).
* Details about available hessian and gradient options, see [Supported gradient and hessian methods](@ref gradient_support).

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

## Citation

We will soon publish a preprint you can cite if you found PEtab.jl helpful in your work.
