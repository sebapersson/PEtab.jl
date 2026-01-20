# [Import PEtab standard format](@id import_petab_problem)

[PEtab](https://petab.readthedocs.io/en/latest/), from which PEtab.jl gets its name, is a
flexible, table-based standard for specifying parameter estimation problems
[schmiester2021petab](@cite). PEtab.jl has full support for both the PEtab v1 and PEtab v2
format.

## Input: a valid PEtab problem

A tutorial on creating valid PEtab problems is available in the PEtab
[documentation](https://petab.readthedocs.io/en/latest/), and there is also
[PEtab GUI](https://github.com/PEtab-dev/PEtab-GUI/) [jost2025petab](@cite) for creating
problems via a graphical interface. Moreover, a large collection of ready-to-run
PEtab problems is provided in the PEtab benchmark
[repository](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab). In this
tutorial, we use the Boehm model [boehm2014identification](@cite), which can be downloaded
from [here](https://github.com/sebapersson/PEtab.jl/tree/main/docs/src/assets/boehm).

## PEtab import

A PEtab problem consists of multiple tables (e.g., measurements, parameters to estimate) and
a YAML file tying them together. To import a problem, the YAML path is needed:

```@example 1
using PEtab
# path_yaml depends on where the model is saved
path_yaml = joinpath("boehm", "Boehm_JProteomeRes2014.yaml")
path_yaml = joinpath(@__DIR__, "..", "..", "assets", "boehm", "Boehm_JProteomeRes2014.yaml") # hide
model = PEtabModel(path_yaml)
nothing # hide
```

Given a `PEtabModel`, a `PEtabODEProblem` can be created:

```@example 1
petab_prob = PEtabODEProblem(model)
```

As described in the starting [tutorial](@ref tutorial), this `PEtabODEProblem` can be used
for downstream tasks such as parameter estimation or Bayesian inference.

## Exporting a PEtab problem

After tasks like parameter estimation, a PEtab problem can be exported in the standard
format using `export_petab`. For example, following multi-start optimization:

```@example 1
using Fides
ms_res = calibrate_multistart(petab_prob, Fides.BFGS(), 10)
```

using the best parameters in `ms_res` a PEtab problem can be exported to a target directory:

```@example 1
dir_export = joinpath(@__DIR__, "petab_result")
export_petab(dir_export, petab_prob, ms_res)
rm(dir_export; recursive = true) # hide
```

!!! note "Export requires PEtab standard format input"
      Currently `export_petab` only supports problems that were imported from the PEtab
      standard format.

## What happens during PEtab import (deep dive)

During import, PEtab.jl performs the following steps:

1. The SBML model is converted into a `ReactionSystem` using
   [SBMLImporter.jl](https://github.com/sebapersson/SBMLImporter.jl), and then into a
   `ODESystem`. The system is symbolically preprocessed (including symbolic Jacobian
   generation), which typically improves simulation performance.
2. The PEtab observable table is translated into Julia functions for the observables (`h`),
   measurement noise/scale (`σ`), and initial conditions (`u0`).
3. Any PEtab events are translated into Julia
   [callbacks](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/).

These steps happen automatically. To inspect the generated code, import with
`write_to_file = true` to have these files written to `dir_yaml/Julia_model_files/`.

## References

```@bibliography
Pages = ["standard_format.md"]
Canonical = false
```
