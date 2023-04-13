# Models with preequilibration (steady-state simulation)

In this tutorial we show to create `PEtabODEproblem` for the small Branmark model which has a preequilibration condition. (≤ 75 states). This means that before the main simulation where we compare the model against data, the model must first be at a steady state $du = f(u, p, t) \approx 0$ which can be achieved via

1. Simulations
2. Rootfinding

To run the code you need the Brannmark PEtab files which can be found [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Brannmark/). A fully runnable example of this tutorial can be found [here](https://github.com/sebapersson/PEtab.jl/tree/main/examples/Brannmark.jl).

First we read the model and load necessary libraries.

```julia
using PEtab
using OrdinaryDiffEq
using Printf

pathYaml = joinpath(@__DIR__, "Brannmark", "Brannmark_JBC2010.yaml")
petabModel = readPEtabModel(pathYaml, verbose=true)
```
```
PEtabModel for model Brannmark. ODE-system has 9 states and 23 parameters.
Generated Julia files are at ...
```

## Steady-state solver

For models with preequilibration before the main simulation we must solve for the steady state $du = f(u, p, t) ≈ 0$. This can be done via i) `:Rootfinding` where we use any algorithm from [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl) to find the roots of $f$, and by ii) `:Simulate` where from the initial condition we simulate the model until it reaches a steady state. The latter is more stable and often performs best.

When creating a `PEtabODEProblem` we can set steady-state solver options via the function `getSteadyStateSolverOptions`, where the first argument is the method to use; either `:Rootfinding` or `:Simulate` (recommended). For `:Simulate` we can choose how to terminate steady-state simulation via the `howCheckSimulationReachedSteadyState` argument which accepts:

1. **:wrms** : Weighted root-mean square $\sqrt{\sum_{i=1}^n \bigg( \frac{du[i]}{\mathrm{reltol}*u[i] + \mathrm{abstol}} \bigg)  \frac{1}{n}} \leq 1$ where $n$ is the number of ODE:s.
2. **:Newton** : If Newton-step Δu is sufficiently small $\sqrt{\sum_{i=1}^n \bigg( \frac{\Delta u[i]}{\mathrm{reltol}*u[i] + \mathrm{abstol}} \bigg)  \frac{1}{n}} \leq 1$

Newton often perform better but requires an invertible Jacobian. In case it is non-invertible the code switches automatically to `:wrms`. (`abstol`, `reltol`) defaults to ODE solver tolerances divided by 100.

Below we use `:Simulate` with `:wrms` termination:

```julia
odeSolverOptions = getODESolverOptions(Rodas5P(), abstol=1e-8, reltol=1e-8)
ssOptions = getSteadyStateSolverOptions(:Simulate,
                                        howCheckSimulationReachedSteadyState=:wrms)
petabProblem = setupPEtabODEProblem(petabModel, odeSolverOptions, 
                                    ssSolverOptions=ssOptions,
                                    gradientMethod=:ForwardDiff) 
p = petabProblem.θ_nominalT 
gradient = zeros(length(p)) 
cost = petabProblem.computeCost(p)
petabProblem.computeGradient!(gradient, p)
@printf("Cost= %.2f\n", cost)
@printf("First element in the gradient = %.2e\n", gradient[1])
```
```
Cost = 141.89
First element in the gradient = 2.70e-03
```

Some useful notes regarding the steady-state solver are:

* In case a `SteadyStateSolverOption` is not specified the default is `:Simulate` with `:wrms`.
* A separate steady-state solver option can also be set for the gradient via `ssSolverGradientOptions`.
* All gradient and hessian options are compatible with `:Simulate`. `:Rootfinding` is only compatible with approaches using forward-mode automatic differentiation.
