# [Choosing the best options for a PEtab problem](@id best_options)

PEtab.jl supports several gradient and hessian methods. In addition, it is compatible with the ODE solvers in the [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) package, hence, there are many possible choices when creating a `PEtabODEProblem` via `setupPEtabODEProblem`. To help navigate these we here provide recommended settings that often work well for specific problem types. The recommendations are based on an extensive benchmark study.

!!! note
    Every problem is unique, and the recommended choice here will often work well but might not be optimal for a specific model

## Small models ($\leq 20$ parameters and $\leq 15$ ODE:s)

**ODE solver**: For small stiff models the Rosenbrock `Rodas5P()` solver is often one of the fastest and most accurate one. Julia bdf-solvers such as `QNDF()` can also perform well here, but they are often less accurate and reliable (fail more often) than `Rodas5P()`. If the model is "mildly" stiff composite solvers such as `AutoVern7(Rodas5P())` often performs best. Regardless of solver we recommend to use low tolerances (around `abstol, reltol = 1e-8, 1e-8`) to obtain accurate gradients.

**Gradient method**: For small models forward-mode automatic differentiation via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) performs best, and is often twice as fast as the forward-sensitivity equations approach in AMICI. Thus, we recommend `gradientMethod=:ForwardDiff`.

* **Note1** - For `:ForwardDiff` the user can set the [chunk-size](https://juliadiff.org/ForwardDiff.jl/stable/). This can substantially improve performance, and we plan to add automatic tuning of it.
* **Note2** - If the model has many simulation condition-specific parameters (parameters that only appear in a subset of simulation conditions) it can be efficient to set `splitOverConditions=true` (see [this](@ref Beer_tut) tutorial).

**Hessian method**: For small models it is computationally feasible to compute an accurate full hessian via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl). For most models we benchmarked a provided hessian improved convergence. Thus, we recommend `hessianMethod=:ForwardDiff`.

* **Note1** - For models with pre-equilibration (steady-state simulations) our benchmarks suggest it might be better to use the Gauss-Newton hessian approximation.
* **Note2** - For models where it too expensive to compute the full hessian (e.g. due to many simulation conditions) the hessian [block approximation](@ref gradient_support) can be a good option.
* **Note3** - In particular, the interior-point Newton method from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) performs well if provided with a full hessian.

All in all, for a small model a good setup often is:

```julia
odeSolverOptions = ODESolverOptions(Rodas5P(), abstol=1e-8, reltol=1e-8)
petabProblem = setupPEtabODEProblem(petabModel, odeSolverOptions, 
                                    gradientMethod=:ForwardDiff, 
                                    hessianMethod=:ForwardDiff)
```

## Medium-sized models ($\leq 75$ parameters and $\leq 50$ ODE:s)

**ODE solver**: For medium-sized stiff models the bdf-solvers such as `QNDF()` are among the fastest, while being sufficiently accurate. The drawback with Julia bdf-solvers is that for certain models (e.g. models with many events) they frequently fail at low tolerances. If this happens a good plan B is `KenCarp4()`. Another good option is Sundials' `CVODE_BDF()`, however it is not recommended since as it is written in C++ and thus is not compatible with forward-mode automatic differentiation. Regardless of solver we recommend to use low tolerances (around `abstol, reltol = 1e-8, 1e-8`) to obtain accurate gradients.

**Gradient method**: For medium-sized models when the hessian is approximated via the Gauss-Newton method (which often performs best) we recommend computing the gradient via the forward sensitivities (`gradientMethod=:ForwardEquations`) where the sensitives are computed via forward-mode automatic differentiation (`sensealg=:ForwardDiff`). The main benefit is that if the optimizers always computes the gradient before the hessian these sensitivities can be reused when computing the hessian. Otherwise, if, for example, a BFGS hessian-approximation is used `gradientMethod=:ForwardDiff` often performs best.

* **Note1** - For `:ForwardDiff` the user can set the [chunk-size](https://juliadiff.org/ForwardDiff.jl/stable/). This can substantially improve performance, and we plan to add automatic tuning of it.
* **Note2** - If the model has many simulation condition-specific parameters (parameters that only appear in a subset of simulation conditions) it can be efficient to set `splitOverConditions=true` (see [this](@ref Beer_tut) tutorial).

**Hessian method**: For medium-sized models it is typically computationally infeasible to compute an accurate full hessian via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl). Rather we recommend the Gauss-Newton hessian approximation which often performs better than the commonly used (L)-BFGS approximation. Thus, we recommend `hessianMethod=:GaussNewton`.

* **Note1** - In particular trust-region Newton methods such as [Fides.py](https://github.com/fides-dev/fides) performs well if provided with a full hessian. Interior-point methods do not perform as well.

All in all, in case the gradient is always computed before the optimizer hessian a good setup often is:

```julia
odeSolverOptions = ODESolverOptions(QNDF(), abstol=1e-8, reltol=1e-8)
petabProblem = setupPEtabODEProblem(petabModel, odeSolverOptions, 
                                    gradientMethod=:ForwardEquations, 
                                    hessianMethod=:GaussNewton, 
                                    reuseS=true)
```

Otherwise, a good setup is:

```julia
odeSolverOptions = ODESolverOptions(QNDF(), abstol=1e-8, reltol=1e-8)
petabProblem = setupPEtabODEProblem(petabModel, odeSolverOptions, 
                                    gradientMethod=:ForwardDiff, 
                                    hessianMethod=:GaussNewton)
```

## Large models ($\geq 75$ parameters and $\geq 50$ ODE:s)

**ODE solver**: For large models we recommend to first benchmark different ODE solvers suitable for large models such as; `QNDF()`, `FBDF()`, `KenCarp4()`, and `CVODE_BDF()`. Moreover, we recommend to test providing the ODE solver with a sparse Jacobian (`sparseJacobian::Bool=false`), and to test different linear solvers such as `CVODE_BDF(linsolve=:KLU)`. More details on how to efficiently solve a large stiff models can be found [here](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/advanced_ode_example/).

* **Note** - We strongly recommend comparing different ODE solvers as this can lead to a substantial reduction of runtime.

**Gradient method**: For large models the most scalable approach is adjoint sensitivity analysis (`gradientMethod=:Adjoint`). Currently, for `sensealg` we support `InterpolatingAdjoint()` and `QuadratureAdjoint()` from SciMLSensitivity (see their [documentation](https://github.com/SciML/SciMLSensitivity.jl) for info), and, since it is more reliable, we recommend `InterpolatingAdjoint()`.

* **Note1** - For adjoint sensitivity analysis we recommend to manually set the ODE solver gradient options. Currently, `CVODE_BDF()` outperforms all native Julia solvers.
* **Note2** - The user can provide any options that `InterpolatingAdjoint()` and `QuadratureAdjoint()` accept.
* **Note3** - Currently adjoint sensitivity analysis is not as reliable in Julia as in AMICI ([see](https://github.com/SciML/SciMLSensitivity.jl/issues/795)), but our benchmarks show that SciMLSensitivity has the potential to be faster.

**Hessian method**: For large models computing the sensitives (Gauss-Newton) or a full hessian is not computationally feasible. Hence, the best option is often to use some L-(BFGS) approximation. BFGS support is built into most available optimizers such as [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl) and [Fides.py](https://github.com/fides-dev/fides).

All in all, for a large model a good setup often is:

```julia
odeSolverOptions = ODESolverOptions(CVODE_BDF(), abstol=1e-8, reltol=1e-8) 
odeSolverGradientOptions = ODESolverOptions(CVODE_BDF(), abstol=1e-8, reltol=1e-8) 
petabProblem = setupPEtabODEProblem(petabModel, odeSolverOptions, 
                                    odeSolverGradientOptions=odeSolverGradientOptions,
                                    gradientMethod=:Adjoint, 
                                    sensealg=InterpolatingAdjoint()) 
```
