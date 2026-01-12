# PEtab.jl

PEtab.jl is a Julia package for creating parameter-estimation problems for fitting ordinary differential equation (ODE) models to data.

## Main features

- Define problems directly in Julia, with models provided as
  [Catalyst.jl](https://github.com/SciML/Catalyst.jl) `ReactionSystem`,
  [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) `ODESystem`, or as
  [SBML](https://sbml.org/) (via
  [SBMLImporter.jl](https://github.com/sebapersson/SBMLImporter.jl)). Problems can be
  defined with a wide range of features, such as multiple observables and/or simulation
  conditions, events, and pre-equilibration (steady-state initialization).
- Import and work with PEtab problems in both v1 and v2 of the
  [PEtab](https://petab.readthedocs.io/en/latest/) standard format.
- Built on the SciML ecosystem, with access to performant stiff and non-stiff ODE
  solvers from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl), and
  efficient gradients via forward-mode automatic differentiation (small models) and adjoint
  sensitivities (large models).
- High performant, often faster than the state-of-the-art toolbox AMICI by ~2× for gradient
  and parameter-estimation workloads [persson2025petab](@cite).
- High-level wrappers for parameter estimation via Optim.jl, Ipopt.jl, Fides.jl, and
  Optimization.jl.
- High-level wrapper for Bayesian inference via
  [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) (including NUTS) and
  AdaptiveMCMC.jl.

!!! note "Star us on GitHub!"
    If you find the package useful in your work please consider giving us a star on
    [GitHub](https://github.com/sebapersson/PEtab.jl). This will help us secure funding in
    the future to continue maintaining the package.

!!! tip "Latest news: PEtab.jl v3.0"
    Version 3.0 is a breaking release that added support for ModelingToolkit v9 and
    Catalyst v14. Along with updating these packages, PEtab.jl underwent a major update,
    with new functionality added as well as the renaming of several functions to be more
    consistent with the naming convention in the SciML ecosystem. See the
    [HISTORY](https://github.com/sebapersson/PEtab.jl/blob/main/HISTORY.md) file for more
    details.

## Installation

PEtab.jl is an officially registered Julia package, tested and supported on Linux, macOS and
Windows. The easiest way to install it is via the Julia package manager. In the Julia REPL,
enter:

```julia
julia> ] add PEtab
```

or alternatively

```julia
julia> using Pkg; Pkg.add("PEtab")
```

PEtab.jl is compatible with Julia **1.10 and above**. For best performance, we strongly
recommend using the latest Julia version, which can be most reliably installed using
[juliaup](https://github.com/JuliaLang/juliaup).

If you encounter installation issues, please consult the [troubleshooting guide](@ref
install_fail).

## Getting help

If you have any problems using PEtab, here are some helpful tips:

- Read the [FAQ](@ref FAQ) section in the online documentation.
- Post your questions in the `#sciml-sysbio` channel on the
  [Julia Slack](https://julialang.org/slack/). While PEtab.jl is not exclusively for systems
  biology, the `#sciml-sysbio` channel is where the package authors are most active.
- If you encounter unexpected behavior or a bug, please see how to file an issue on the
  [Contributing page](@ref contribute).

## Citation

If you use PEtab.jl in work that is published, please cite the paper below:

```bibtex
@article{PEtabBioinformatics2025,
  title={PEtab.jl: advancing the efficiency and utility of dynamic modelling},
  author={Persson, Sebastian and Fr{\"o}hlich, Fabian and Grein, Stephan and Loman, Torkel and Ognissanti, Damiano and Hasselgren, Viktor and Hasenauer, Jan and Cvijovic, Marija},
  journal={Bioinformatics},
  volume={41},
  number={9},
  pages={btaf497},
  year={2025},
  publisher={Oxford University Press}
}
```
