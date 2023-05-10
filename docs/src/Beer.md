# [Models with many conditions specific parameters](@id Beer_tut)

This tutorial demonstrates how to create a `PEtabODEproblem` for a small ODE-model with many parameters to estimate. Specifically, the ODE-system has $\leq 20$ states and $\leq 20$ parameters, but there are approximately 70 parameters to estimate, since most parameters are specific to a subset of simulation conditions. For example, *cond1* has a parameter τ_cond1, and *cond2* has τ_cond2, which maps to the ODE-system parameter τ, respectively.

To run the code, you need the Beer PEtab files, which you can find [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Beer.jl). You can also find a fully runnable example of this tutorial [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Beer.jl).

First, we load the necessary libraries and read the model.

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

When dealing with small ODE-systems like Beer, the most efficient gradient method is `gradientMethod=:ForwardDiff`. Additionally, we can compute the hessian via `hessianMethod=:ForwardDiff`. However, since there are several condition-specific parameters in Beer, we have to perform as many forward-passes (solve the ODE model) as there are model-parameters when we compute the gradient and hessian via a single call to ForwardDiff.jl. To address this issue, we can use the option `splitOverConditions=true` to force one ForwardDiff.jl call per simulation condition. This is most efficient for models where a majority of parameters are specific to a subset of simulation conditions.

For a model like Beer, the following options are thus recommended:

1. `odeSolverOptions` - Rodas5P() (works well for smaller models with up to 15 states) and we use the default `abstol, reltol .= 1e-8`.
2. `gradientMethod` - For small models like Beer, forward mode automatic differentiation (AD) is the fastest, so we choose `:ForwardDiff`.
3. `hessianMethod` - For small models like Boehm with up to 20 parameters, it is computationally feasible to compute the full Hessian via forward-mode AD. Thus, we choose `:ForwardDiff`.
4. `splitOverConditions=true` - This forces a call to ForwardDiff.jl per simulation condition.

```julia
odeSolverOptions = ODESolverOptions(Rodas5P(), abstol=1e-8, reltol=1e-8)
petabProblem = createPEtabODEProblem(petabModel, odeSolverOptions, 
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
