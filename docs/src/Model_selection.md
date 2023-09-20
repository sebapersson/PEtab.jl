# Model selection with PEtab Select

In many scenarios we have competing hypotheses (model structures) that we want to compare. For model selection, various approaches like forward search, backward search, and exhaustive search using evaluation criteria such as AIC are commonly used. These methods are supported by [PEtab Select](https://github.com/PEtab-dev/petab_select), a tool designed for model selection.

!!! note
    To use the parameter estimation functionality Optim, QuasiMonteCarlo and PyCall must be loaded (see examples below).

## Example

PEtab.jl provides support for PEtab Select through the `run_PEtab_select` function. This function takes two required arguments; the path to the PEtab Select YAML file, and the optimizer for parameter estimation. For the optimizer, you can choose from `optimizer=Fides()` (Fides Newton-trust region), `optimizer=IPNewton()` from Optim.jl, or `optimizer=LBFGS()` from Optim.jl ([see](@ref parameter_estimation)). Additionally, you can pass any keyword arguments accepted by the `calibrate_model` function for parameter estimation and `PEtabODEProblem` function for setting simulation options ([see](@ref gradient_support)).

Since PEtab Select is a Python package, you need to have [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) installed. Before using it, build PyCall with a Python environment that has PEtab select installed. Here's an example of how to do it (note that `pathToPythonExe` depends on your system configuration):

```julia
using PyCall
pathToPythonExe = joinpath("/", "home", "sebpe", "anaconda3", "envs", "PeTab", "bin", "python")
ENV["PYTHON"] = pathToPythonExe
import Pkg; Pkg.build("PyCall")
```

Once you have a correctly encoded PEtab Select problem (see the [guide](https://github.com/PEtab-dev/petab_select) for details), you can run PEtab Select using the `IPNewton()` optimizer with the following code:

```julia
using PEtab 
using OrdinaryDiffEq
using Optim
using QuasiMonteCarlo

path_yaml = joinpath(@__DIR__, "PEtab_select", "0002", "petab_select_problem.yaml")
pathSave = run_PEtab_select(path_yaml, IPNewton(), 
                          nOptimisationStarts=10, 
                          ode_solver=ODESolver(Rodas5P()),
                          gradient_method=:ForwardDiff, 
                          hessian_method=:ForwardDiff)
```
```
┌ Info: PEtab select problem info
│ Method: forward
└ Criterion: AIC
[ Info: Model selection round 1 with 1 candidates - as the code compiles in this round compiled it takes extra long time https://xkcd.com/303/
[ Info: Callibrating model M1_0
[ Info: Model selection round 2 with 3 candidates
[ Info: Callibrating model M1_1
[ Info: Callibrating model M1_2
[ Info: Callibrating model M1_3
[ Info: Model selection round 3 with 2 candidates
[ Info: Callibrating model M1_5
[ Info: Callibrating model M1_6
[ Info: Model selection round 4 with 1 candidates
[ Info: Callibrating model M1_7
[ Info: Saving results for best model at 0002/PEtab_select_forward_AIC.yaml
```

The YAML file storing the model selection results will be saved at `pathSave`.

To run the code, you will need the PEtab files, which you can find [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/0002). You can also find a fully runnable example of this tutorial [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/PEtab_select.jl).