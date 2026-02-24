# [Import PEtab SciML standard format](@id import_petab_scimlproblem)

[PEtab SciML](https://github.com/PEtab-dev/petab_sciml) extends the
[PEtab](https://github.com/PEtab-dev/PEtab) standard format for parameter estimation.
PEtab SciML is a flexible, table-based format for specifying SciML parameter estimation
problems, and PEtab.jl supports importing and solving problems defined in this format.

## Input: a valid PEtab SciML problem

Tutorials on creating valid PEtab SciML problems are available in the PEtab SciML
documentation. In this tutorial, we import the Lotka–Volterra UDE example from
[rackauckas2020universal](@cite), whose PEtab files which can be downloaded
from [here](https://github.com/sebapersson/PEtab.jl/tree/main/docs/src/assets/lv_ude).

Briefly, the Lotka–Volterra SciML model describes interactions between a prey population
`prey` and a predator population `predator`. In this SciML example, the interaction term is
replaced by a neural network:

```math
\begin{align*}
\frac{\mathrm{d}prey}{\mathrm{d}t} &= \alpha prey + \mathrm{NN}(prey, predator)[1] \\
\frac{\mathrm{d}predator}{\mathrm{d}t} &= -\gamma predator + \mathrm{NN}(prey, predator)[2]
\end{align*}
```

where `NN(prey, predator)` is a feed-forward neural network taking prey and predator
abundances as input.

## PEtab SciML import

A PEtab SciML problem consists of multiple tables (e.g. measurements and parameter
definitions), model files, and a YAML file that ties them together. To import the problem,
only the path to the YAML file is needed. The first step is to load the ML models in the
PEtab SciML problem:

```@example 1
using Lux, PEtab
# `path_yaml` depends on where the problem files are located
path_yaml = joinpath("lv_ude", "lv.yaml")
path_yaml = joinpath(@__DIR__, "..", "..", "assets", "lv_ude", "lv.yaml") # hide
ml_models = MLModels(path_yaml)
```

Then a `PEtabModel` and the corresponding `PEtabODEProblem` can be created:

```@example 1
petab_model = PEtabModel(path_yaml; ml_models = ml_models)
petab_prob = PEtabODEProblem(petab_model)
```

As described in the [SciML starter tutorial](@ref sciml_starter), `petab_prob` can then be
used for downstream tasks such as simulation and model training.
