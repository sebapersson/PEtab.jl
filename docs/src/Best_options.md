# [Choosing the best options for a PEtab problem](@id best_options)

PEtab.jl provides several gradient and hessian methods that can be used with the ODE solvers in the DifferentialEquations.jl package. You can choose from a variety of options when creating a `PEtabODEProblem` using the `PEtabODEProblem` function. If you don't specify any of these options, appropriate options will be selected automatically based on an extensive benchmark study. These default options usually work well for specific problem types. In the following section, we will discuss the main findings of the benchmark study.

!!! note
    These recommendations often work well for specific problem types, they may not be optimal for every model, as each problem is unique.

## Small models ($\leq 20$ parameters and $\leq 15$ ODE:s)

**ODE solver**: For small stiff models, the Rosenbrock `Rodas5P()` solver is often the fastest and most accurate option. While Julia bdf-solvers such as `QNDF()` can also perform well, they are often less reliable and less accurate than `Rodas5P()`. If the model is "mildly" stiff, composite solvers such as `AutoVern7(Rodas5P())` often perform best. Regardless of solver, we recommend using low tolerances (around `abstol, reltol = 1e-8, 1e-8`) to obtain accurate gradients.

**Gradient method**: For small models, forward-mode automatic differentiation via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) tends to be the best performing option, and is often twice as fast as the forward-sensitivity equations approach in AMICI. Therefore, we recommend using `gradient_method=:ForwardDiff`.

* **Note1** - For `:ForwardDiff`, the user can set the [chunk-size](https://juliadiff.org/ForwardDiff.jl/stable/), which can substantially improve performance. We plan to add automatic tuning of this in the future.
* **Note2** - If the model has many simulation condition-specific parameters (parameters that only appear in a subset of simulation conditions), it can be efficient to set `split_over_conditions=true` (see [this](@ref Beer_tut) tutorial).

**Hessian method**: For small models, it is computationally feasible to compute an accurate full Hessian via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl). For most models we benchmarked, using a provided Hessian improved convergence. Therefore, we recommend using `hessian_method=:ForwardDiff`.

* **Note1** - For models with pre-equilibration (steady-state simulations), our benchmarks suggest that it might be better to use the Gauss-Newton Hessian approximation.
* **Note2** - For models where it is too expensive to compute the full Hessian (e.g. due to many simulation conditions), the Hessian [block approximation](@ref gradient_support) can be a good option.
* **Note3** - In particular, the interior-point Newton method from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) performs well if provided with a full Hessian.

Overall, for a small model, a good setup often includes:

```julia
petab_problem = PEtabODEProblem(petab_model, 
                                     ode_solver=ODESolver(Rodas5P(), abstol=1e-8, reltol=1e-8), 
                                     gradient_method=:ForwardDiff, 
                                     hessian_method=:ForwardDiff)
```

## Medium-sized models ($\leq 75$ parameters and $\leq 50$ ODE:s)

**ODE solver**: For medium-sized stiff models, bdf-solvers like `QNDF()` are often fast enough and accurate. However, they can fail for certain models with many events when low tolerances are used. In such cases, `KenCarp4()` is a good alternative. Another option is Sundials' `CVODE_BDF()`, but it's written in C++ and not compatible with forward-mode automatic differentiation. To obtain accurate gradients, we recommend using low tolerances (around `abstol, reltol = 1e-8, 1e-8`) regardless of solver.

**Gradient method**: For medium-sized models, when using the Gauss-Newton method to approximate the Hessian, we recommend computing the gradient via the forward sensitivities (`gradient_method=:ForwardEquations`), where the sensitivities are computed via forward-mode automatic differentiation (`sensealg=:ForwardDiff`). This way, the sensitivities can be reused when computing the Hessian if the optimizer always computes the gradient first. Otherwise, if a BFGS Hessian-approximation is used, `gradient_method=:ForwardDiff` often performs best.

* **Note1** - For `:ForwardDiff`, the user can set the [chunk-size](https://juliadiff.org/ForwardDiff.jl/stable/) to improve performance, and we plan to add automatic tuning of it.
* **Note2** - If the model has many simulation condition-specific parameters (parameters that only appear in a subset of simulation conditions), it can be efficient to set `split_over_conditions=true` (see [this](@ref Beer_tut) tutorial).

**Hessian method**: For medium-sized models, it's often computationally infeasible to compute an accurate full Hessian via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl). Instead, we recommend the Gauss-Newton Hessian approximation, which often performs better than the commonly used (L)-BFGS approximation. Thus, we recommend `hessian_method=:GaussNewton`.

* **Note1** - Trust-region Newton methods like [Fides.py](https://github.com/fides-dev/fides) perform well if provided with a full Hessian. Interior-point methods don't perform as well.

Overall, when the gradient is always computed before the Hessian in the optimizer, a good setup is often:

```julia
petab_problem = PEtabODEProblem(petab_model, 
                                     ode_solver=ODESolver(QNDF(), abstol=1e-8, reltol=1e-8),
                                     gradient_method=:ForwardEquations, 
                                     hessian_method=:GaussNewton, 
                                     reuse_sensitivities=true)
```

Otherwise, a good setup is:

```julia
petab_problem = PEtabODEProblem(petab_model, 
                                     ode_solver=ODESolver(QNDF(), abstol=1e-8, reltol=1e-8),
                                     gradient_method=:ForwardDiff, 
                                     hessian_method=:GaussNewton)
```

## Large models ($\geq 75$ parameters and $\geq 50$ ODE:s)

**ODE solver**: To efficiently solve large models, we recommend benchmarking different ODE solvers such as `QNDF()`, `FBDF()`, `KenCarp4()`, and `CVODE_BDF()`. You can also try providing the ODE solver with a sparse Jacobian (`sparse_jacobian::Bool=false`) and testing different linear solvers such as `CVODE_BDF(linsolve=:KLU)`. Check out [this link](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/advanced_ode_example/) for more information on solving large stiff models.

* **Note** - It's important to compare different ODE solvers, as this can significantly reduce runtime.

**Gradient method**: For large models, the most scalable approach is adjoint sensitivity analysis (`gradient_method=:Adjoint`). We support `InterpolatingAdjoint()` and `QuadratureAdjoint()` from SciMLSensitivity (see their [documentation](https://github.com/SciML/SciMLSensitivity.jl) for info), but we recommend `InterpolatingAdjoint()` because it's more reliable.

* **Note1** - When using adjoint sensitivity analysis, we recommend manually setting the ODE solver gradient options. Currently, `CVODE_BDF()` outperforms all native Julia solvers.
* **Note2** - You can provide any options that `InterpolatingAdjoint()` and `QuadratureAdjoint()` accept.
* **Note3** - Adjoint sensitivity analysis is not as reliable in Julia as in AMICI ([see](https://github.com/SciML/SciMLSensitivity.jl/issues/795)), but our benchmarks show that SciMLSensitivity has the potential to be faster.

**Hessian method**: For large models, computing the sensitives (Gauss-Newton) or a full hessian is not computationally feasible. Thus, the best option is often to use an L-(BFGS) approximation. BFGS support is built into most available optimizers such as [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl), and [Fides.py](https://github.com/fides-dev/fides).

All in all, for a large model, a good setup often is:

```julia
petab_problem = PEtabODEProblem(petab_model, 
                                     ode_solver=ODESolver(CVODE_BDF(), abstol=1e-8, reltol=1e-8), 
                                     ode_solver_gradient=ODESolver(CVODE_BDF(), abstol=1e-8, reltol=1e-8),
                                     gradient_method=:Adjoint, 
                                     sensealg=InterpolatingAdjoint()) 
```
