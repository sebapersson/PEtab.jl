# [Importing PEtab standard format](@id import_petab_problem)

[PEtab](https://petab.readthedocs.io/en/latest/) is a standard table-based format for specifying parameter estimation problems. If a problem is provided in this standard format, PEtab.jl can import it directly. This tutorial will cover how to import PEtab problems.

## Input - a valid PEtab problem

To import a PEtab problem, naturally a valid PEtab problem is required. A tutorial on creating a PEtab problem can be found in the PEtab [documentation](https://petab.readthedocs.io/en/latest/), and there is a [linting tool](https://github.com/PEtab-dev/PEtab/tree/main) in Python for checking correctness. Additionally, PEtab.jl also performs several checks to ensure correctness when importing the problem.

A collection of valid PEtab problems is available in the PEtab benchmark [repository](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab). In this tutorial, we will use an already published model from the PEtab benchmark repository. Specifically, we consider the JakStat5 signaling model, referred to here as the Boehm model (after the first author). The PEtab files for this signaling model can be found [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Boehm).

## Importing a PEtabModel

A PEtab problem consists of five files: an [SBML](https://sbml.org/) model file, a table with experimental conditions, a table with observables, a table with measurements, and a table with parameters to estimate. These are tied together with a YAML file. To import the problem, you only need to provide the path to the YAML file:

```@example 1
using PEtab
# path_yaml depends on where the model is saved
path_yaml = joinpath(@__DIR__, "assets", "boehm", "boehm.yaml")
model = PEtabModel(path_yaml)
```

Given a `PEtabModel`, you can easily create a `PEtabODEProblem`:

```@example 1
petab_prob = PEtabODEProblem(model)
```

This `PEtabODEProblem` can then be used to fit the model to data using numerical optimization (see this [guide]) or a Bayesian inference approach (see this [guide]). For additional options when importing a `PEtabModel`, see the [API](@ref API) documentation.

## What Happens During PEtab Import (Deep Dive)

If you are not interested in the inner workings of PEtab.jl, feel free to skip this section. FOr the interested reader, when importing a PEtab model, several things happen:

1. The SBML file is converted into a [Catalyst.jl](https://github.com/SciML/Catalyst.jl) `ReactionSystem` using the [SBMLImporter.jl](https://github.com/sebapersson/SBMLImporter.jl) package. This `ReactionSystem` is then converted into an `ODESystem`. During this step, the model is symbolically pre-processed, which includes computing the ODE Jacobian symbolically. This typically improves simulation performance.
2. The observable PEtab table is translated into Julia functions that compute observables (`h`), measurement noise (`σ`), and initial values (`u0`).
3. Any potential model events are translated into Julia [callbacks](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/).

All of these steps happen automatically. By setting `write_to_file=true` when importing the model, the generated model functions can be found in the *dir_yaml/Julia_model_files/* directory.