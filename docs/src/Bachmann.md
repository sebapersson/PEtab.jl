# Medium-sized models (Bachmann model)

In this tutorial we show how to create a `PEtabODEproblem` for the medium-sized Bachmann model, and explain i) how to compute the gradient via forward-sensitivity equations, ii) how to compute the gradient via adjoint sensitivity analysis, and iii) how to compute the Gauss-Newton hessian approximation. The latter often perform better than the (L)-BFGS Hessian approximation.

To run the code you need the Bachmann PEtab files which can be found [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Bachmann). A fully runnable example of this tutorial can be found [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Bachmann.jl).

First we read the model and load necessary libraries.

```julia
using PEtab
using OrdinaryDiffEq
using Sundials # For CVODE_BDF
using SciMLSensitivity # For adjoint
using Printf
 
pathYaml = joinpath(@__DIR__, "Bachmann", "Bachmann_MSB2011.yaml") 
petabModel = readPEtabModel(pathYaml, verbose=true)
```

```
PEtabModel for model Bachmann. ODE-system has 25 states and 39 parameters.
Generated Julia files are at ...
```

## Adjoint sensitivity analysis

For a subset of medium-sized models, and definitely for big models, gradients are most efficiently computed via adjoint sensitivity analysis (`gradientMethod=:Adjoint`). For adjoint there are several tuneable options that can improve performance, some key ones are:

1. `odeSolverGradientOptions` - Which ODE solver and solver tolerances (`abstol` and `reltol`) to use when computing the gradient (in this case when solving the adjoint ODE-system). Below we use `CVODE_BDF()` which currently is the best performing stiff solver for the adjoint problem in Julia.
2. `sensealg` - which adjoint algorithm to use. Currently, we support `InterpolatingAdjoint` and `QuadratureAdjoint` from SciMLSensitivity (see their [documentation](https://github.com/SciML/SciMLSensitivity.jl) for info). The user can provide any of the options these methods are compatible with, so if you want to use the ReverseDiffVJP an acceptable option is; `sensealg=InterpolatingAdjoint(autojacvec=ReversDiffVJP())`.

* **Note1** - currently adjoint sensitivity analysis is not as reliable in Julia as in AMICI (https://github.com/SciML/SciMLSensitivity.jl/issues/795), but our benchmarks show that SciMLSensitivity has the potential to be faster.
* **Note2** - the compilation times can be quite substantial for adjoint sensitivity analysis.
* **Note3** - below we use `QNDF()` for the cost which often is one of the best Julia solvers for larger models.

```julia
solverOptions = getODESolverOptions(QNDF(), abstol=1e-8, reltol=1e-8) 
solverGradientOptions = getODESolverOptions(CVODE_BDF(), abstol=1e-8, reltol=1e-8) 
petabProblem = setupPEtabODEProblem(petabModel, solverOptions, 
                                    odeSolverGradientOptions=solverGradientOptions,
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

For medium-sized models computing the full Hessian via forward-mode automatic differentiation is often too expensive, thus we need some approximation. The [Gauss-Newton](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm) (GN)approximation often performs better than the (L)-BFGS approximation. To compute it we need the forward sensitivities. These sensitives can also be used to compute the gradient. As some optimizers such as Fides.py compute both the hessian and gradient at each iteration we can be smart here and save the sensitives between the gradient and hessian computations.

When choosing `gradientMethod=:ForwardEquations` and `hessianMethod=:GaussNewton` there are several tuneable options, the key ones are:

1. `sensealg` - which sensitivity algorithm to use when computing for the sensitives. We support both `ForwardSensitivity()` and `ForwardDiffSensitivity()` with tuneable options as provided by SciMLSensitivity (see their [documentation](https://github.com/SciML/SciMLSensitivity.jl) for info). The most efficient option though is `:ForwardDiff` where forward mode automatic differentiation is used to compute the sensitivities.
2. `reuseS::Bool` - whether or not to reuse the sensitives from the gradient computations when computing the Gauss-Newton hessian-approximation. Whether this option is applicable depends on the optimizers, for example it works with Fides.py but not [Optim.jl:s](https://github.com/JuliaNLSolvers/Optim.jl) `IPNewton()`.
   * Note - this approach requires that `sensealg=:ForwardDiff` for the gradient.

```julia
odeSolverOptions = getODESolverOptions(QNDF(), abstol=1e-8, reltol=1e-8) 
petabProblem = setupPEtabODEProblem(petabModel, odeSolverOptions, 
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
