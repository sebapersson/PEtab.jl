# Getting started

In this starting tutorial, we will use the small Boehm model to demonstrate the key features of PEtab.jl.

To run the code, you will need the Boehm PEtab files, which can be accessed [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Boehm). For a fully functional example of this tutorial, please visit [this link](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Boehm.jl).

## Reading a PEtab model

To get started, we first need to read the PEtab files into Julia. This can be easily done using the PEtab.jl package. Once we provide the path to the PEtab yaml-file, PEtab.jl reads all the PEtab files and creates a `PEtabModel` struct. Here are some of the things that happen behind the scenes:

1. The SBML file is converted into [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) format, which allows for symbolic computation of the ODE-model Jacobian.
2. The observable PEtab table is translated into Julia functions that compute observables ($h$), noise parameter ($\sigma$), and initial values ($u_0$).
3. To compute gradients via adjoint sensitivity analysis or forward sensitivity equations, the derivatives of $h$ and $\sigma$ are calculated symbolically with respect to the ODE-model states ($u$) and parameters.

All of these steps happen automatically, and you can find the resulting files in the *dirYamlFile/Julia_model_files/* directory. By default, the `readPEtabModel` function does not rebuild the Julia files if they already exist, so it saves time.

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

To create a `PEtabODEProblem` from a PEtab model, we use the `createPEtabODEProblem` function. This function allows us to customize various options (see the API documentation for a full list), but the most important ones are:

* `odeSolverOptions`: This option lets us choose an ODE solver and set tolerances for the solver. For example, we can choose the `Rodas5P()` solver and set tolerances of `abstol, reltol = 1e-8`. This solver works well for smaller models with up to 15 states.
* `gradientMethod`: This option lets us choose a gradient computation method. For small models like Boehm, forward mode automatic differentiation (AD) is the fastest method, so we choose `:ForwardDiff`.
* `hessianMethod`: This option lets us choose a Hessian computation method. For small models with up to 20 parameters, it is computationally feasible to compute the full Hessian via forward-mode AD. Thus, we choose `:ForwardDiff`.

```julia
petabProblem = createPEtabODEProblem(petabModel, 
                                     odeSolverOptions=ODESolverOptions(Rodas5P(), abstol=1e-8, reltol=1e-8), 
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

If we don't provide any of these arguments, we automatically select appropriate options based on the size of the problem following the guidelines in [Choosing best options for a PEtab problem](@ref best_options).

```julia
petabProblem = createPEtabODEProblem(petabModel)
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

The `PEtabODEProblem` includes all the necessary information to set up an optimization problem using most available optimizers. Its main fields are:

1. `petabODEProblem.computeCost` - This field computes the cost (i.e., the objective function) for a given parameter vector `θ`.
2. `petabODEProblem.computeGradient!` - This field computes the gradient of the cost with respect to `θ` using the chosen method.
3. `petabODEProblem.computeHessian!` - This field computes the Hessian of the cost with respect to `θ` using the chosen method.
4. `petabODEProblem.lowerBounds` - This field is a vector containing the lower bounds for the parameters, as specified in the PEtab parameters file.
5. `petabODEProblem.upperBounds` - This field is a vector containing the upper bounds for the parameters, as specified in the PEtab parameters file.
6. `petabODEProblem.θ_estNames` - This field is a vector containing the names of the parameters to be estimated.

* **Note1** - The parameter vector `θ` is assumed to be on the scale specified by the PEtab parameters file. For example, if parameter `i` is on the log scale, then `θ[i]` should also be on the log scale.
* **Note2** - The `computeGradient!` and `computeHessian!` functions are in-place functions, meaning that their first argument is a pre-allocated gradient or Hessian, respectively (see below).

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

Next, we suggest you take a look at the [Choosing best options for a PEtab problem](@ref best_options) guide. Additionally, we recommend exploring the [Supported gradient and hessian methods](@ref gradient_support) section. In case you want to provide your model-file as a Julia-file instead of an SBML file take a look at [Providing a model as a Julia file instead of an SBML File](@ref Beer_Julia_import).
