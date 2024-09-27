# [Importing a PEtab problem](@id import_petab_problem)

In this starting tutorial, we will use the small Boehm model to demonstrate how to import a parameter-estimation problem specifed in the PEtab-format.

To run the code, you will need the Boehm PEtab files, which can be accessed [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Boehm). For a fully functional example of this tutorial, see [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Boehm.jl).

## Reading a PEtab model

To get started, we first need to read the PEtab files into Julia. This can be easily done using `PEtabModel(path_yaml)`, which by using the PEtab-yaml file parses the PEtab files into a `PEtabModel` struct. Here several things happen behind the scenes:

1. The SBML file is converted into [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) format, which allows for symbolic computation of the ODE-model Jacobian.
2. The observable PEtab table is translated into Julia functions that compute observables ($h$), noise parameter ($\sigma$), and initial values ($u_0$).
3. To compute gradients via adjoint sensitivity analysis or forward sensitivity equations, the derivatives of $h$ and $\sigma$ are calculated symbolically with respect to the ODE-model states ($u$) and parameters.

All of these steps happen automatically, and you can find the resulting files in the *dirYamlFile/Julia_model_files/* directory assuming (as default) `write_to_file=true`, otherwise no model files are written to disk. By default, the `PEtabModel` constructor does not rebuild the Julia files if they already exist to save time.

```julia
using PEtab
using OrdinaryDiffEq
using Printf

path_yaml = joinpath(@__DIR__, "Boehm", "Boehm_JProteomeRes2014.yaml")
model = PEtabModel(path_yaml, verbose=true)
```
```
PEtabModel for model Boehm. ODE-system has 8 states and 10 parameters.
Generated Julia files are at ...
```

## Creating a PEtabODEProblem

Next step is to create a `PEtabODEProblem` from a PEtab model, for which we use the `PEtabODEProblem` constructor. This constructors allows various options (see [here](@ref API) for a full list), where the most important ones are:

* `ode_solver`: This option lets us choose an ODE solver and set tolerances for the solver. For example, we can choose the `Rodas5P()` solver and set tolerances of `abstol, reltol = 1e-8`. This solver works well for smaller models with up to 15 states.
* `gradient_method`: This option lets us choose a gradient computation method. For small models like Boehm, forward mode automatic differentiation (AD) is the fastest method, so we choose `:ForwardDiff`.
* `hessian_method`: This option lets us choose a Hessian computation method. For small models with up to 20 parameters, it is computationally feasible to compute the full Hessian via forward-mode AD. Thus, we choose `:ForwardDiff`.

```julia
petab_problem = PEtabODEProblem(model, 
                                ode_solver=ODESolver(Rodas5P(), abstol=1e-8, reltol=1e-8), 
                                gradient_method=:ForwardDiff, 
                                hessian_method=:ForwardDiff)
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

If we don not provide any of these arguments, appropriate options are automatically selected based on the size of the problem following the guidelines in [Choosing best options for a PEtab problem](@ref default_options).

```julia
petab_problem = PEtabODEProblem(model)
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

1. `petabODEProblem.nllh` - This field computes the cost (i.e., the objective function) for a given parameter vector `θ`.
2. `petabODEProblem.grad!` - This field computes the gradient of the cost with respect to `θ` using the chosen method.
3. `petabODEProblem.hess!` - This field computes the Hessian of the cost with respect to `θ` using the chosen method.
4. `petabODEProblem.lower_bounds` - This field is a vector containing the lower bounds for the parameters, as specified in the PEtab parameters file.
5. `petabODEProblem.upper_bounds` - This field is a vector containing the upper bounds for the parameters, as specified in the PEtab parameters file.
6. `petabODEProblem.xnames` - This field is a vector containing the names of the parameters to be estimated.

!!! note
    The parameter vector `θ` is assumed to be on the scale specified by the PEtab parameters file. For example, if parameter `i` is on the log scale, then `θ[i]` should also be on the log scale.
!!! note
    The `compute_gradient!` and `compute_hessian!` functions are in-place functions, meaning that their first argument is a pre-allocated gradient or Hessian, respectively (see below).

```julia
# Parameters are log-scaled
p = petab_problem.xnominal_transformed 
gradient = zeros(length(p))
hessian = zeros(length(p), length(p))
cost = petab_problem.nllh(p)
petab_problem.grad!(gradient, p)
petab_problem.hess!(hessian, p)
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

Next, if you want the estimate the model we suggest you take a look at the [Parameter Estimation (Model Calibration)](@ref parameter_estimation). We also recommend [Choosing best options for a PEtab problem](@ref default_options) guide. In case you want to setup the PEtab problem directory in Julia take a look at [Creating a PEtab Parameter Estimation Problem in Julia](@ref define_in_julia)