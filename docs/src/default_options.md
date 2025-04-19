# [Default Options](@id default_options)

PEtab.jl supports several gradient and Hessian computation methods, as well as the ODE solvers available in [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl). As a result, there are many possible choices when creating a `PEtabODEProblem`. To simplify usage, PEtab.jl has benchmark derived heuristics to select appropriate default options based on the size of the parameter estimation problem. This page discusses these default options when creating a `PEtabODEProblem`.

The default options are based on model size, which is determined by the number of ODEs and the number of parameters to estimate. This is because there is typically no "one-size-fits-all" solution: ODE solvers that perform well for small models may not perform well for large models, and gradient methods that are effective for small models may not be suitable for larger ones. It should also be noted that the defaults are based on benchmarks for stiff biological models. For information on how to configure the `PEtabODEProblem` for models outside of biology, see [this](@ref nonstiff_models) page.

!!! note
    These defaults often work well, but they may not be optimal for every model as each problem is unique.

## Small Models ($\leq 20$ Parameters and $\leq 15$ ODEs)

**ODE solver**: For small stiff models, the Rosenbrock `Rodas5P()` solver is typically the fastest and most accurate. While Julia's BDF solvers like `QNDF()` can perform well, they tend to be less reliable and accurate compared to `Rodas5P()` in this regime.

**Gradient method**: For small models, forward-mode automatic differentiation via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is usually the fastest method, often being twice as fast as the forward-sensitivity equations approach. For `:ForwardDiff`, it is possible to set the [chunk size](https://juliadiff.org/ForwardDiff.jl/stable/), which can improve performance. However, determining the optimal value can be challenging, and thus we plan to add automatic tuning.

**Hessian method**: For small models, computing the full Hessian via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is often computationally feasible. Benchmarks have shown that using the full Hessian improves convergence.

Overall, for small models, the default configuration is:

```julia
petab_prob = PEtabODEProblem(model; odesolver=ODESolver(Rodas5P()),
                             gradient_method=:ForwardDiff, 
                             hessian_method=:ForwardDiff)
```

!!! note
    If a model has many condition-specific parameters that only appear in a subset of simulation conditions (see [this](@ref define_conditions) tutorial), runtime can be improved by setting `split_over_conditions=true` in the `PEtabODEProblem`. For more details, see [this] example.

## Medium-Sized Models ($\leq 75$ Parameters and $\leq 75$ ODEs)

**ODE solver**: For medium-sized stiff models, multi-step BDF solvers like `QNDF()` are generally fast and accurate [stadter2021benchmarking](@cite). However, they can fail for models with many events when using low tolerances. In such cases, `KenCarp4()` is a reliable alternative.

**Gradient method**: As with small models, the most efficient gradient method for medium-sized models is forward-mode automatic differentiation via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).

**Hessian method**: For medium-sized models, computing the full Hessian via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is often computationally infeasible. Instead, we recommend the Gauss-Newton Hessian approximation, which in behcmarks frequently outperforms the commonly used (L)-BFGS approximation [frohlich2022fides](@cite).

Overall, for medium models, the default configuration is:

```julia
petab_prob = PEtabODEProblem(model; odesolver=ODESolver(QNDF(), abstol=1e-8, reltol=1e-8),
                             gradient_method=:ForwardDiff, 
                             hessian_method=:GaussNewton)
```

!!! note
    If an optimization algorithm computes both the gradient and Hessian simultaneously, and the Hessian is computed using the Gauss-Newton approximation, it is possible to reuse quantities from gradient computations by setting `gradient_method = :ForwardEquations` and `reuse_sensitivities = true`. For more information, see [this](@ref options_optimizers) page on the Fides optimizer.

## Large Models ($\geq 75$ Parameters and $\geq 75$ ODEs)

While PEtab.jl provides default settings for large models, we recommend benchmarking different methods. This is because selecting the best ODE solver and gradient configuration can substantially impact runtime.

**ODE solver**: For efficiently simulating large models, we recommend benchmarking various ODE solvers designed for large problems, such as `QNDF()`, `FBDF()`, `KenCarp4()`, and `CVODE_BDF()`. Further, we recommend trying a sparse Jacobian (`sparse_jacobian = true`) and testing different linear solvers, such as `CVODE_BDF(linsolve=:KLU)`. For more information on solving large stiff models in Julia, see [this](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/advanced_ode_example/) tutorial.

**Gradient method**: For large models, the most efficient gradient method is adjoint sensitivity analysis (`gradient_method=:Adjoint`). PEtab.jl supports the `InterpolatingAdjoint()`, `GaussAdjoint()`, and `QuadratureAdjoint()` algorithms from SciMLSensitivity.jl. The default is `InterpolatingAdjoint(autojacvec = EnzymeVJP())`, but we strongly recommend benchmarking different adjoint methods and different `autojacvec` options. For further details on adjoint options, see the SciMLSensitivity.jl [documentation](https://docs.sciml.ai/SciMLSensitivity/stable/).

**Hessian method**: For large models, computing sensitivities (Gauss-Newton) or a full Hessian is not computationally feasible. Therefore, using an L-(BFGS) approximation is often the best option. BFGS support is built into most available optimizers such as [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl), and [Fides.jl](https://fides-dev.github.io/Fides.jl/stable/).

Overall, for large models, the default configuration is:

```julia
petab_prob = PEtabODEProblem(model, odesolver=ODESolver(CVODE_BDF()),
                             gradient_method=:Adjoint,
                             sensealg=InterpolatingAdjoint(autojacvec=EnzymeVJP()))
```

## References

```@bibliography
Pages = ["default_options.md"]
Canonical = false
```
