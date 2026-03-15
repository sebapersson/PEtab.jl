# PEtab.jl

_Create parameter estimation problems for dynamic models_

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sebapersson.github.io/PEtab.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sebapersson.github.io/PEtab.jl/dev/)
[![Build Status](https://github.com/sebapersson/PEtab.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sebapersson/PEtab.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![codecov](https://codecov.io/gh/sebapersson/PEtab.jl/graph/badge.svg?token=J7PXRF30JG)](https://codecov.io/gh/sebapersson/PEtab.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

[Getting Started](https://sebapersson.github.io/PEtab.jl/stable/tutorial) |
[Documentation](https://sebapersson.github.io/PEtab.jl/stable/) |
[Contributing](https://sebapersson.github.io/PEtab.jl/stable/contributing)

PEtab.jl is a Julia package for creating parameter estimation problems to fit ordinary
differential equation (ODE) and scientific machine learning (SciML) models to time-series
data.

Major features are:

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
  and parameter-estimation workloads.
- High-level wrappers for parameter estimation via Optim.jl, Ipopt.jl, Fides.jl, and
  Optimization.jl.
- Support for state-of-the-art SciML training strategies, such as curriculum learning and
  multiple shooting, via
  [PEtabTraining.jl](https://github.com/sebapersson/PEtabTraining.jl).
- Support for state-of-the-art SciML training strategies (e.g. curriculum learning, multiple
  shooting etc...) via [PEtabTraining.jl](https://github.com/sebapersson/PEtabTraining.jl).
- High-level wrapper for Bayesian inference via
  [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) (including NUTS) and
  AdaptiveMCMC.jl.

## Installation

PEtab.jl is a registered Julia package and can be installed with the Julia package manager
using:

```julia
julia> import Pkg; Pkg.add("PEtab")
```

PEtab.jl is compatible with Julia 1.10 and above. For additional installation details, see
the [documentation](https://sebapersson.github.io/PEtab.jl/stable/#Installation).

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
