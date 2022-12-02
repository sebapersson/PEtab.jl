# PEtab.jl
*Importer for systems biological models defined in the PEtab format.*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sebapersson.github.io/PEtab.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sebapersson.github.io/PEtab.jl/dev/)
[![Build Status](https://github.com/sebapersson/PEtab.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sebapersson/PEtab.jl/actions/workflows/CI.yml?query=branch%3Amain)


This package is based on work by [Viktor Hasselgren](https://github.com/CleonII/Master-Thesis) and [Sebastian Persson](https://github.com/sebapersson) and allows for the import of [PEtab](https://github.com/PEtab-dev/PEtab) models into Julia via the python library [`libsbml`](https://github.com/sbmlteam/libsbml).
Therefore, `libsbml` first has to be installed via
```
pip3 install python-libsbml
```
Then, `PEtab.jl` can be installed from the Julia terminal via
```julia
julia> ] dev https://github.com/sebapersson/PEtab.jl
```

For example, given the path to a folder containing the so-called [Bachmann model](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/tree/master/Benchmark-Models/Bachmann_MSB2011) as a string `BachmannPath`, it can be imported via:
```julia
using PEtab
Bachmann = setUpPeTabModel("Bachmann_MSB2011", BachmannPath; write=false)
```
The `write=true` keyword can optionally be used to permanently save the Julia source code required to define the model via `.jl` files in the same folder.

Subsequently, given an appropriate choice of [ODE solver](https://diffeq.sciml.ai/stable/solvers/ode_solve/) and a desired tolerance for integration, the cost function and its derivatives are constructed by
```julia
using OrdinaryDiffEq
peTabOpt = setUpCostGradHess(Bachmann, AutoTsit5(Rosenbrock23()), 1e-12)
peTabOpt.evalF(peTabOpt.paramVecTransformed)
```
