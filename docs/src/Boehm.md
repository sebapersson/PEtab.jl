# Getting started

In this introductory tutorial we will create a `PEtabODEproblem` for the small Boehm model, while simultaneously covering the main features of PEtab.jl.

To run the code you need the Boehm PEtab files which can be found [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Boehm). A fully runnable example of this tutorial can be found [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Boehm.jl).

## Reading a PEtab model

As a starting point we need to read the PEtab files into Julia. This is made easy by the package PEtab.jl which, given the path to the PEtab yaml-file, reads all PEtab files into a `PEtabModel` struct. Here several things happen under the hood:
    
1. The SBML file is translated into [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) format (e.g. allow symbolic computation of the ODE-model Jacobian).
2. The observable PEtab table is translated into Julia functions for computing the observables ($h$), noise parameter ($\sigma$) and initial values ($u_0$).
3. To be able to compute gradients via adjoint sensitivity analysis and/or forward sensitivity equations the derivatives of $h$ and $\sigma$ are computed symbolically with respect to the ODE-models states ($u$) and parameters.
    
All this happens automatically and you can find the resulting files in *dirYamlFile/Julia_model_files/*. To save time the function `readPEtabModel` has the default `forceBuildJlFiles=false` meaning that the Julia files are not rebuilt in case they already exist.

```julia
using PEtab
using OrdinaryDiffEq
using Printf

pathYaml = joinpath(@__DIR__, "Boehm", "Boehm_JProteomeRes2014.yaml")
petabModel = readPEtabModel(pathYaml, verbose=true)
```
```
PEtabModel for model Boehm. ODE-system has 8 states and 10 parameters.
Generated Julia files are at ...
```

## Creating a PEtabODEProblem

Given a PEtab model we can create a `PEtabODEProblem` (in the future we plan to add surrogate, SDE, etc... problems). The function `setupPEtabODEProblem` accepts several options (full list in API documentation), but some key ones are:

1. `odeSolverOptions` - Which ODE solver to use and the solver tolerances (`abstol` and `reltol`). Below we use the ODE solver `Rodas5P()` which works well for smaller models ($\leq$ 15 states), and we use the default `abstol, reltol .= 1e-8`.
2. `gradientMethod` - For small models like Boehm forward mode automatic differentiation (AD) is fastest, thus we choose `:ForwardDiff`.
3. `hessianMethod` - For small models like Boehm with $\leq 20$ parameters it is computationally feasible to compute the full Hessian via forward-mode AD. Thus, we choose `:ForwardDiff`.

```julia
odeSolverOptions = ODESolverOptions(Rodas5P(), abstol=1e-8, reltol=1e-8)
petabProblem = setupPEtabODEProblem(petabModel, odeSolverOptions, 
                                    gradientMethod=:ForwardDiff, 
                                    hessianMethod=:ForwardDiff)
```
```
PEtabODEProblem for Boehm. ODE-states: 8. Parameters to estimate: 9 where 6 are dynamic.
---------- Problem settings ----------
Gradient method : ForwardDiff
Hessian method : ForwardDiff
--------- ODE-solver settings --------
Cost Rodas5P(). Options (abstol, reltol, maxiters) = (1.0e-08, 1.0e-08, 1.0e+04)
Gradient Rodas5P(). Options (abstol, reltol, maxiters) = (1.0e-08, 1.0e-08, 1.0e+04)
```

## Computing the cost, gradient and hessian

The `PEtabODEProblem` contains everything needed to set up an optimization problem with most available optimizers. The main fields are:

1. `petabODEProblem.computeCost` - Given a parameter vector θ it computes the cost (objective function).
2. `petabODEProblem.computeGradient!`- Given a parameter vector θ it computes the gradient with the chosen method.
3. `petabODEProblem.computeHessian!`- Given a parameter vector θ it computes the hessian with the chosen method.
4. `petabODEProblem.lowerBounds` - A vector with the lower bounds for the parameters as specified by the PEtab parameters file.
5. `petabODEProblem.upperBounds` - A vector with the upper bounds for the parameters as specified by the PEtab parameters file. 
6. `petabODEProblem.θ_estNames` - A vector with the names of the parameters to estimate.

* **Note1** - The parameter vector θ is assumed to be on the PEtab specified parameter-scale. Thus, if parameter $i$ is on the log-scale so should also `θ[i]` be.
* **Note2** - The `computeGradient!` and `computeHessian!` functions are in-place functions. Thus, their first argument is an already pre-allocated gradient and hessian, respectively (see below).

```julia
# Parameters are log-scaled
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
Cost = 138.22
First element in the gradient = 2.20e-02
First element in the hessian = 2199.49
```

## Where to go from here

After this tutorial we recommend looking at [Choosing best options for a PEtab problem](@ref best_options). We also recommend looking at [Supported gradient and hessian methods](@ref gradient_support).
