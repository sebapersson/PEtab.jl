# Model Selection with PEtab Select

Sometimes we have competing hypotheses (model structures) that we want to compare to ultimately select the best model/hypothesis. There are [various approaches](https://en.wikipedia.org/wiki/Stepwise_regression) for model selection, such as forward search, backward search, and exhaustive search, where models are compared based on information criteria like AIC or BIC. Additionally, there are efficient algorithms that combine both backward and forward search, such as Famos [gabel2019famos](@cite). All these model selection methods are supported by the Python package [PEtab Select](https://github.com/PEtab-dev/petab_select), for which PEtab.jl provides an interface.

This advanced documentation page assumes that you know how to import and crate PEtab problems in the standard format (a tutorial can be found [here](@ref import_petab_problem)) as well as the basics of multi-start parameter estimation with PEtab.jl (a tutorial can be found [here](@ref pest_methods)). Additionally, since PEtab Select is a Python package, to run this code you need to have [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) installed, and you must build PyCall with a Python environment that has [petab_select](https://github.com/PEtab-dev/petab_select) installed:

```julia
using PyCall
# Path to Python executable with PEtab Select installed
path_python_exe = "path_python"
ENV["PYTHON"] = path_python_exe
# Build PyCall with the PEtab Select Python environment
import Pkg
Pkg.build("PyCall")
```

!!! note
    Model selection is currently only possible for problem in the [PEtab Select](https://github.com/PEtab-dev/petab_select) standard format. We plan to add a Julia interface.

## Model Selection Example

PEtab.jl provides support for PEtab Select through the `petab_select` function:

```@docs; canonical=false
petab_select
```

As an example, for a simple signaling model (files can be downloaded from [here](https://github.com/sebapersson/PEtab.jl/tree/main/docs/src/assets/petab_select)), you can run PEtab Select with the `IPNewton()` algorithm:

```julia
using Optim, PEtab, PyCall
path_yaml = joinpath(@__DIR__, "assets", "petab_select", "petab_select_problem.yaml")
path_res = petab_select(path_yaml, IPNewton(); nmultistarts=10)
```
```julia
┌ Info: PEtab select problem info
│ Method: brute_force
└ Criterion: AIC
[ Info: Model selection round 1 with 1 candidates - as the code compiles in this round it takes extra long time https://xkcd.com/303/
[ Info: Callibrating model M1_1
[ Info: Saving results for best model at /home/sebpe/.julia/dev/PEtab/docs/build/assets/petab_select/PEtab_select_brute_force_AIC.yaml
```

Where the YAML file storing the model selection results is saved at `path_res`.

```@bibliography
Pages = ["pest_select.md"]
Canonical = false
```
