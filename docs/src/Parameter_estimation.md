# [Optimization (Parameter Estimation)](@id parameter_estimation)

PEtab.jl seamlessly integrates with various optimization packages such as [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl), and [Fides.py](https://github.com/fides-dev/fides). Check out our [examples](https://github.com/sebapersson/PEtab.jl/tree/main/examples), or see below, to see how it is done.

For parameter estimation in ODE models used in systems biology, a widely adopted approach is multi-start local optimization. In this method, a local optimizer is run from a large number (around 100-1000) of random start-guesses. These start-guesses are efficiently generated using techniques like Latin-hypercube sampling to explore the parameter space effectively. When it comes to selecting the optimizer, based on extensive benchmarks, here's a useful rule of thumb:

- If you can provide a full Hessian, the Interior-point Newton method in [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) generally outperforms the trust-region method in Fides.py.
- If you can only provide a Gauss-Newton Hessian approximation (not the full Hessian), the Newton trust-region method in Fides.py usually outperforms the interior-point method in [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).

PEtab.jl offers a lightweight interface for performing multi-start parameter estimation with [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) and [Fides.py](https://github.com/fides-dev/fides) (see below).

!!! note
    Keep in mind that each problem is unique, and although the suggested options are generally effective, they may not always be the ideal choice for a particular model.

## Parameter estimation using Optim.jl

For [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) in PEtab.jl, we provide support for three methods:

- IPNewton(): Interior point Newton method.
- LBFGS(): Newton's method with LBFGS Hessian approximation and box-constraints (Fminbox).
- BFGS(): Newton's method with BFGS Hessian approximation and box-constraints (Fminbox).

To perform parameter estimation, you can use the `calibrateModel` function. This function requires a `PEtabODEProblem`, an optimization algorithm, and the following keyword options:

- `nOptimisationStarts::Int`: The number of multi-starts to be performed. The default value is 100.
- `samplingMethod`: The method for generating start guesses. It supports any method from QuasiMonteCarlo.jl, with LatinHypercube being the default.
- `options`: Optimization options. For Optim.jl optimizers, it accepts an `Optim.Options` struct.

Here's an example where we run a 50 multi-start for the Boehm model using the Interior-point method and Latin-Hypercube sampling to generate the start-guesses:


```julia
using Optim
import QuasiMonteCarlo
fvals, xvals = callibrateModel(petabProblem, IPNewton(), 
                               nOptimisationStarts=5, 
                               samplingMethod=QuasiMonteCarlo.LatinHypercubeSample(), 
                               options=Optim.Options(show_trace = false, iterations=200))
@printf("Best found value = %.3f\n", minimum(fvals))
```
```
Best found value = 138.222
```

Note that via `Optim.Options` we set the maximum number of iterations to 200.

## Parameter estimation using Fides.py

Fides.py is a trust-region Newton method that excels when computing the full Hessian is computationally expensive but we can compute the Gauss-Newton Hessian approximation.

Fides (specifically, Newton trust-region with box-constraints) is not available directly in Julia. Therefore, to use it, you need to call a Python library. To set up the necessary environment, make sure you have [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) installed. Next you must build PyCall with a Python environment that has Fides installed (**note:** `pathToPythonExe` depends on your system configuration):

```julia
pathToPythonExe = joinpath("/", "home", "sebpe", "anaconda3", "envs", "PeTab", "bin", "python")
ENV["PYTHON"] = pathToPythonExe
import Pkg; Pkg.build("PyCall")
```

Fides always computes the cost, gradient, and Hessian at each iteration. Therefore, if you use the Gauss-Newton Hessian approximation, you can reuse the forward sensitivities (`reuseS=true`) from the gradient calculation by setting `gradientMethod=:ForwardEquations` in the `createPEtabODEProblem` function:

```julia
petabProblem = createPEtabODEProblem(petabModel, 
                                     odeSolverOptions=ODESolverOptions(Rodas5P()), 
                                     gradientMethod=:ForwardEquations, 
                                     hessianMethod=:GaussNewton, 
                                     sensealg=:ForwardDiff, 
                                     reuseS=true)
```
```
PEtabODEProblem for Boehm. ODE-states: 8. Parameters to estimate: 9 where 6 are dynamic.
---------- Problem settings ----------
Gradient method : ForwardEquations
Hessian method : GaussNewton
--------- ODE-solver settings --------
Cost Rodas5P(). Options (abstol, reltol, maxiters) = (1.0e-08, 1.0e-08, 1.0e+04)
Gradient Rodas5P(). Options (abstol, reltol, maxiters) = (1.0e-08, 1.0e-08, 1.0e+04)
```

Now, you can proceed with the parameter estimation:

```julia
fvals, xvals = calibrateModel(petabProblem, Fides(verbose=false), 
                              nOptimisationStarts=5, 
                              samplingMethod=QuasiMonteCarlo.LatinHypercubeSample(), 
                              options=py"{'maxiter' : 200}"o)
@printf("Best found value = %.3f\n", minimum(fvals))     
```
```
Best found value = 147.544
```

Please note that since Fides is a Python package, when providing options, they must be in the form of a Python dictionary using the `py"..."` string.