# [Import PEtab SciML standard format](@id import_petab_scimlproblem)

[PEtab SciML](https://github.com/PEtab-dev/petab_sciml) extends the
[PEtab](https://github.com/PEtab-dev/PEtab) table-based standard for parameter estimation to
support SciML parameter estimation problems [schmiester2021petab](@cite). PEtab.jl has full
support for the PEtab SciML format, and this tutorial shows how to import such problems.

## Input: a valid PEtab SciML problem

Tutorials on creating valid PEtab SciML problems are available in the PEtab SciML
[documentation](https://petab-sciml.readthedocs.io/latest/introduction.html). In this
tutorial, the Lotka–Volterra UDE example from [rackauckas2020universal](@cite) is imported.
The corresponding PEtab SciML files can be downloaded from
[here](https://github.com/sebapersson/PEtab.jl/tree/main/docs/src/assets/lv_ude).

Briefly, the Lotka–Volterra SciML model describes interactions between a `prey` population
and a `predator` population. In this SciML example, the interaction term in the original
model is replaced by a neural network:

```math
\begin{align*}
\frac{\mathrm{d}prey}{\mathrm{d}t} &= \alpha prey + \mathrm{NN}(prey, predator)[1] \\
\frac{\mathrm{d}predator}{\mathrm{d}t} &= -\gamma predator + \mathrm{NN}(prey, predator)[2]
\end{align*}
```

where `NN(prey, predator)` is a feed-forward neural network taking `prey` and `predator`
abundances as input and returning a two-dimensional output.

## PEtab SciML import

A PEtab SciML problem consists of multiple tables (e.g. measurements and parameter
definitions), model files, and a YAML file that ties them together. To import the problem,
only the path to the YAML file is needed. The first step is to load the ML models in the
PEtab SciML problem:

```@example 1
using Lux, PEtab
# `path_yaml` depends on where the problem files are located
path_yaml = joinpath("lv_ude", "problem.yaml")
path_yaml = joinpath(@__DIR__, "..", "..", "assets", "lv_ude", "problem.yaml") # hide
ml_models = MLModels(path_yaml)
```

Then a `PEtabModel` and the corresponding `PEtabODEProblem` can be created:

```@example 1
petab_model = PEtabModel(path_yaml; ml_models = ml_models)
petab_prob = PEtabODEProblem(petab_model)
```

As described in the [SciML starter tutorial](@ref sciml_starter), `petab_prob` can then be
used for downstream tasks such as simulation and model training.

## References

```@bibliography
Pages = ["standard_format.md"]
Canonical = false
```
