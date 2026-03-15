# PEtab.jl

PEtab.jl is a Julia package for creating parameter estimation problems to fit ordinary
differential equation (ODE) and scientific machine learning (SciML) models to time-series
data.

## Main features

- Define ODE parameter estimation problems directly in Julia, with models provided as
  [Catalyst.jl](https://github.com/SciML/Catalyst.jl) `ReactionSystem`,
  [ModelingToolkitBase.jl](https://github.com/SciML/ModelingToolkit.jl) `ODESystem`, an
  [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) `ODEProblem`, or as
  [SBML](https://sbml.org/) model (imported via
  [SBMLImporter.jl](https://github.com/sebapersson/SBMLImporter.jl)). Problems can be
  defined with a wide range of features, such as multiple observables and/or simulation
  conditions, events, and pre-equilibration (steady-state initialization).
- Support for three types of scientific machine learning (SciML) problems combining
  mechanistic ODE models with machine-learning (ML) components: (1) ML in the ODE dynamics
  (e.g. UDEs/Neural ODEs), (2) ML in the observable/measurement model linking simulations to
  data, and (3) pre-simulation ML mapping high-dimensional inputs to ODE parameters.
- Import and work with PEtab problems in both v1 and v2 of the
  [PEtab](https://petab.readthedocs.io/en/latest/) format, as well as the
  [PEtab-SciML](https://github.com/PEtab-dev/petab_sciml) standard format.
- Built on the SciML ecosystem, with access to performant stiff and non-stiff ODE solvers
  from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl), and efficient
  gradients via forward-mode automatic differentiation (small models) and adjoint
  sensitivity analysis (large models).
- High performant, often faster than the state-of-the-art toolbox AMICI by ~2× for gradient
  and parameter-estimation workloads [persson2025petab](@cite).
- High-level wrappers for parameter estimation via Optim.jl, Ipopt.jl, Fides.jl, and
  Optimization.jl.
- Support for state-of-the-art SciML training strategies, such as curriculum learning and
  multiple shooting, via
  [PEtabTraining.jl](https://github.com/sebapersson/PEtabTraining.jl).
- High-level wrapper for Bayesian inference via
  [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) (including NUTS) and
  AdaptiveMCMC.jl.

::: tip Star us on GitHub!

If you find the package useful in your work please consider giving us a star on
[GitHub](https://github.com/sebapersson/PEtab.jl). This will help us secure funding in the
future to continue maintaining the package.

:::

::: tip Latest news: PEtab.jl v5.0!

PEtab.jl v5.0 is a breaking release adding support for scientific machine learning (SciML)
problems combining ODE and ML components. In addition, PEtab.jl has been updated to support
Catalyst v16 and ModelingToolkitBase v1. For a full overview of changes, see the
[HISTORY](https://github.com/sebapersson/PEtab.jl/blob/main/HISTORY.md) file.

:::

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
