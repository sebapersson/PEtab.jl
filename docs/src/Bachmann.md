# Medium-sized models (Bachmann model)

In this tutorial, we will guide you through the process of creating a `PEtabODEproblem` for the Bachmann model, a medium-sized ODE model. We'll cover three topics:

1. Computing the gradient via forward-sensitivity equations
2. Computing the gradient via adjoint sensitivity analysis
3. Computing the Gauss-Newton Hessian approximation, which often performs better than the (L)-BFGS Hessian approximation.

To run the code, you'll need the Bachmann PEtab files, which can be found [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Bachmann). You can find a fully runnable example of this tutorial [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Bachmann.jl).

First, we'll read the model and load the necessary libraries.

```julia
using PEtab
using OrdinaryDiffEq
using Sundials # For CVODE_BDF
using Printf
 
pathYaml = joinpath(@__DIR__, "Bachmann", "Bachmann_MSB2011.yaml") 
petabModel = readPEtabModel(pathYaml, verbose=true)
```

```
PEtabModel for model Bachmann. ODE-system has 25 states and 39 parameters.
Generated Julia files are at ...
```

## Adjoint sensitivity analysis

When working with a subset of medium and definitely for large-sized models, the most efficient way to compute gradients is through adjoint sensitivity analysis (`gradientMethod=:Adjoint`). There are several tuneable options that can improve performance, including:

1. `odeSolverGradientOptions`: This determines which ODE solver and solver tolerances (`abstol` and `reltol`) to use when computing the gradient (i.e., when solving the adjoint ODE-system). Currently, the best performing stiff solver for the adjoint problem in Julia is `CVODE_BDF()`.
2. `sensealg`: This determines which adjoint algorithm to use. Currently, `InterpolatingAdjoint` and `QuadratureAdjoint` from SciMLSensitivity are supported. You can find more information in their [documentation](https://github.com/SciML/SciMLSensitivity.jl). You can provide any of the options that these methods are compatible with. For example, if you want to use the `ReverseDiffVJP` algorithm, an acceptable option is `sensealg=InterpolatingAdjoint(autojacvec=ReversDiffVJP())`.

Here are a few things to keep in mind:

* **Note1:** Adjoint sensitivity analysis is not as reliable in Julia as it is in [AMICI](https://github.com/SciML/SciMLSensitivity.jl/issues/795). However, our benchmarks show that SciMLSensitivity has the potential to be faster.
* **Note2:** Compilation times for adjoint sensitivity analysis can be quite substantial.
* **Note3:** In the example below, we use `QNDF()` for the cost, which is often one of the best Julia solvers for larger models.

```julia
using SciMLSensitivity # For adjoint
petabProblem = createPEtabODEProblem(petabModel, 
                                     odeSolverOptions=ODESolverOptions(QNDF(), abstol=1e-8, reltol=1e-8), 
                                     odeSolverGradientOptions=ODESolverOptions(CVODE_BDF(), abstol=1e-8, reltol=1e-8),
                                     gradientMethod=:Adjoint, 
                                     sensealg=InterpolatingAdjoint(autojacvec=EnzymeVJP())) 
p = petabProblem.θ_nominalT 
gradient = zeros(length(p)) 
cost = petabProblem.computeCost(p)
petabProblem.computeGradient!(gradient, p)
@printf("Cost = %.2f\n", cost)
@printf("First element in the gradient = %.2e\n", gradient[1])
```

```
Cost = -418.41
First element in the gradient = -1.70e-03
```

## Forward sensitivity analysis and Gauss-Newton hessian approximation

For medium-sized models, computing the full Hessian via forward-mode automatic differentiation can be too expensive, so we need an approximation. The [Gauss-Newton](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm) (GN) approximation often performs better than the (L)-BFGS approximation. To compute it, we need the forward sensitivities. These sensitivities can also be used to compute the gradient. As some optimizers such as Fides.py compute both the Hessian and gradient at each iteration, we can save the sensitivities between the gradient and Hessian computations.

When choosing `gradientMethod=:ForwardEquations` and `hessianMethod=:GaussNewton`, there are several tunable options, the key ones are:

1. `sensealg` - which sensitivity algorithm to use when computing the sensitivities. We support both `ForwardSensitivity()` and `ForwardDiffSensitivity()` with tunable options as provided by SciMLSensitivity (see their [documentation](https://github.com/SciML/SciMLSensitivity.jl) for more information). The most efficient option is `:ForwardDiff`, where forward-mode automatic differentiation is used to compute the sensitivities.
2. `reuseS::Bool` - whether or not to reuse the sensitivities from the gradient computations when computing the Gauss-Newton Hessian approximation. Whether this option is applicable depends on the optimizer. For example, it works with Fides.py but not with Optim.jl's `IPNewton()`.
   * Note - this approach requires that `sensealg=:ForwardDiff` for the gradient.

```julia
petabProblem = createPEtabODEProblem(petabModel, 
                                     odeSolverOptions=ODESolverOptions(QNDF(), abstol=1e-8, reltol=1e-8),
                                     gradientMethod=:ForwardEquations, 
                                     hessianMethod=:GaussNewton,
                                     sensealg=:ForwardDiff, 
                                     reuseS=true) 
p = petabProblem.θ_nominalT 
gradient = zeros(length(p)) 
hessian = zeros(length(p), length(p)) 
cost = petabProblem.computeCost(p)
petabProblem.computeGradient!(gradient, p)
petabProblem.computeHessian!(hessian, p)
@printf("Cost for Bachmann = %.2f\n", cost)
@printf("First element in the gradient = %.2e\n", gradient[1])
@printf("First element in the Gauss-Newton Hessian = %.2f\n", hessian[1, 1])
```

```
Cost for Bachmann = -418.41
First element in the gradient = -1.85e-03
First element in the Gauss-Newton Hessian = 584.10
```
