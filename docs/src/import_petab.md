# [Importing PEtab Standard Format](@id import_petab_problem)

[PEtab](https://petab.readthedocs.io/en/latest/), from which PEtab.jl gets its name, is a flexible, table-based standard format for specifying parameter estimation problems [schmiester2021petab](@cite). If a problem is provided in this standard format, PEtab.jl can import it directly. This tutorial covers how to import PEtab problems.

## Input - a Valid PEtab Problem

To import a PEtab problem, a valid PEtab problem is required. A tutorial on creating a PEtab problem can be found in the PEtab [documentation](https://petab.readthedocs.io/en/latest/), and a [linting tool](https://github.com/PEtab-dev/PEtab/tree/main) is available in Python for checking correctness. Additionally, PEtab.jl performs several format checks when importing the problem.

A collection of valid PEtab problems is also available in the PEtab benchmark [repository](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab). In this tutorial, we will use an already published model from the PEtab benchmark repository. Specifically, we will consider the STAT5 signaling model, referred to here as the Boehm model (after the first author) [boehm2014identification](@cite). The PEtab files for this model can be found [here](https://github.com/sebapersson/PEtab.jl/tree/main/docs/src/assets/boehm).

## Importing a PEtabModel

A PEtab problem consists of five files: an [SBML](https://sbml.org/) model file, a table with simulation conditions, a table with observables, a table with measurements, and a table with parameters to estimate. These are tied together by a YAML file, and to import a problem, you only need to provide the YAML file path:

```@example 1
using PEtab
# path_yaml depends on where the model is saved
path_yaml = joinpath(@__DIR__, "assets", "boehm", "Boehm_JProteomeRes2014.yaml")
model = PEtabModel(path_yaml)
nothing # hide
```

Given a `PEtabModel`, it is straightforward to create a `PEtabODEProblem`:

```@example 1
petab_prob = PEtabODEProblem(model)
```

As described in the starting [tutorial](@ref tutorial), this `PEtabODEProblem` can then be used for parameter estimation, or Bayesian inference. For tunable options when importing a PEtab problem, see the [API](@ref API) documentation.

## What Happens During PEtab Import (Deep Dive)

When importing a PEtab model, several things happen:

1. The SBML file is converted into a [Catalyst.jl](https://github.com/SciML/Catalyst.jl) `ReactionSystem` using the [SBMLImporter.jl](https://github.com/sebapersson/SBMLImporter.jl) package. This `ReactionSystem` is then converted into an `ODESystem`. During this step, the model is symbolically pre-processed, which includes computing the ODE Jacobian symbolically. The latter typically improves simulation performance.
2. The observable PEtab table is translated into Julia functions that compute observables (`h`), measurement noise (`σ`), and initial values (`u0`).
3. Any potential model events are translated into Julia [callbacks](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/).

All of these steps happen automatically. By setting `write_to_file=true` when importing the model, the generated model functions can be found in the `dir_yaml/Julia_model_files/` directory.

## References

```@bibliography
Pages = ["import_petab.md"]
Canonical = false
```
