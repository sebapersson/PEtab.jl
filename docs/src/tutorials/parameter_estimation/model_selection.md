# [Model selection with PEtab Select](@id petab_select)

Model selection compares competing model structures to identify a structure that
best explains the data. Common strategies include forward, backward, and exhaustive search,
where candidates are ranked using information criteria such as AIC or BIC. Approaches
combining forward and backward search also exist, such as FAMoS [gabel2019famos](@cite). All
these methods are supported by the [PEtab Select](https://github.com/PEtab-dev/petab_select)
package [pathirana2025petab](@cite), for which PEtab.jl provides an interface.

This tutorial showcases PEtab Select and assumes familiarity with importing PEtab problems
in the standard format (see [Importing PEtab problems](@ref import_petab_problem)) and with
multi-start parameter estimation in PEtab.jl (see the [extended estimation tutorial](@ref
pest_methods)).

!!! note "PEtab Select requires the PEtab standard format"
    For PEtab Select, PEtab.jl currently only supports problems in the PEtab standard, not
    problems defined via the Julia interface in PEtab.jl. If Julia-interface support would
    be useful, please open an issue so its implementation can be prioritized.

## Model selection example

PEtab.jl interfaces with PEtab Select via the `petab_select` function:

```@docs; canonical=false
petab_select
```

As an example, for a simple signaling model (files are available
[here](https://github.com/sebapersson/PEtab.jl/tree/main/docs/src/assets/petab_select)),
PEtab Select can be run as:

```@example 1
using Optim, PEtab, PEtabSelect
path_yaml = joinpath(@__DIR__, "assets", "petab_select", "petab_select_problem")
path_yaml = joinpath(@__DIR__, "..", "..", "assets", "petab_select", "petab_select_problem.yaml") # hide
path_res = petab_select(path_yaml, IPNewton(); nmultistarts = 10)
nothing # hide
```

Here, `IPNewton()` specifies the optimizer used for parameter estimation within each model
evaluation, and `nmultistarts = 10` sets the number of multi-starts per model. The YAML file
with model selection results is saved to `path_res`.

## References

```@bibliography
Pages = ["model_selection.md"]
Canonical = false
```
