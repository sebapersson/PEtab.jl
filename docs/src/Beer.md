# [Models with many conditions specific parameters](@id Beer_tut)

In this tutorial we show how to create a `PEtabODEproblem` for a small ODE-model ($\leq 20$ states, $\leq 20$ parameters) with many parameters to estimate ($\approx 70$), because most parameter are specific to a subset of simulation conditions. For example, in *cond1* we have τ_cond1 and in *cond2* we have τ_cond2 that maps to the ODE-system parameter τ respectively.

To run the code you need the Beer PEtab files which can be found [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Beer.jl). A fully runnable example of this tutorial can be found [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Beer.jl).

First we read the model and load necessary libraries.

```julia
using PEtab
using OrdinaryDiffEq
using Printf

pathYaml = joinpath(@__DIR__, "Beer", "Beer_MolBioSystems2014.yaml") 
petabModel = readPEtabModel(pathYaml, verbose=true)
```
```
PEtabModel for model Beer. ODE-system has 4 states and 9 parameters.
Generated Julia files are at ...
```

## Handling condition-specific parameters

For a small ODE-system like Beer the most efficient gradient method is `gradientMethod=:ForwardDiff`, and we can further compute the hessian via `hessianMethod=:ForwardDiff`. However, there are several condition-specific parameters, and as we by default compute the gradient and hessian via a single call to ForwardDiff.jl it means that for this model we have to perform as many forward-passes (solve the ODE model) as there are model-parameters. To force one ForwardDiff.jl call per simulation condition we can use the option `splitOverConditions=true`. Thus, for a model like Beer good options are:

1. `odeSolverOptions` - Rodas5P() (works well for smaller models $\leq 15$ states), and we use the default `abstol, reltol .= 1e-8`.
2. `gradientMethod` - For small models like Beer forward mode automatic differentiation (AD) is fastest, thus below we choose `:ForwardDiff`.
3. `hessianMethod` - For small models like Boehm with $\leq 20$ parameters it is computationally feasible to compute the full Hessian via forward-mode AD. Thus, we choose `:ForwardDiff`.
4. `splitOverConditions=true` - Force a call to ForwardDiff.jl per simulation condition. Most efficient for models where a majority of parameters are specific to a subset of simulation conditions.

```julia
odeSolverOptions = getODESolverOptions(Rodas5P(), abstol=1e-8, reltol=1e-8)
petabProblem = setupPEtabODEProblem(petabModel, odeSolverOptions, 
                                    gradientMethod=:ForwardDiff, 
                                    hessianMethod=:ForwardDiff, 
                                    splitOverConditions=true)

p = petabProblem.θ_nominalT
gradient = zeros(length(p))
hessian = zeros(length(p), length(p))
cost = petabProblem.computeCost(p)
petabProblem.computeGradient!(gradient, p)
petabProblem.computeHessian!(hessian, p)
@printf("Cost = %.2f\n", cost)
@printf("First element in the gradient = %.2e\n", gradient[1])
@printf("First element in the hessian = %.2f\n", hessian[1, 1])
```
```
Cost = -58622.91
First element in the gradient = 7.17e-02
First element in the hessian = 755266.33
```
