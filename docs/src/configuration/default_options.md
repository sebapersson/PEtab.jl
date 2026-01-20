# [Default PEtabODEProblem options](@id default_options)

A `PEtabODEProblem` supports multiple gradient/Hessian computation methods and the ODE
solvers from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl). This leads to
many valid `PEtabODEProblem` configurations, and to simplify usage, default options are
provided based on an extensive benchmark study [persson2025petab](@cite). This page
summarizes these defaults.

In brief, defaults depend primarily by model size (number of ODEs and number of estimated
parameters), since ODE solver performance and especially gradient-computation methods
depend strongly on problem size.

!!! tip "Non-stiff models"
    Defaults are tuned for biological, typically stiff models. For non-stiff models, see
    [Speeding up non-stiff models](@ref nonstiff_models).

## Small models (≤20 parameters and ≤15 ODEs)

The default configuration for small models is:

```julia
petab_prob = PEtabODEProblem(
    model;
    odesolver = ODESolver(Rodas5P()),
    gradient_method = :ForwardDiff,
    hessian_method = :ForwardDiff,
)
```

The rationale is:

- **ODE solver:** For small stiff models, the Rosenbrock solver `Rodas5P()` is typically
  fast, robust (rare with simulation failures), and accurate. Julia’s BDF solvers (e.g.
  `QNDF()`) can also work well, but are often less robust.
- **Gradient method:** Forward-mode automatic differentiation via
  [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is the fastest option for
  small models, often faster than sensitivity-equation approaches implemented in
  tools such as AMICI. Performance can sometimes be improved by tuning the ForwardDiff chunk
  size, but the optimal value is model-dependent.
- **Hessian method:** Computing the full Hessian with
  [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is often feasible for small
  models, and access to the full Hessian typically improves parameter estimation
  performance.

## Medium-sized models (≤75 parameters and ≤75 ODEs)

The default configuration for medium-sized models is:

```julia
petab_prob = PEtabODEProblem(
    model;
    odesolver = ODESolver(QNDF()),
    gradient_method = :ForwardDiff,
    hessian_method = :GaussNewton,
)
```

The rationale is:

- **ODE solver:** For medium-sized stiff models, multi-step BDF solvers such as `QNDF()` are
  often faster than Rosenbrock solvers [stadter2021benchmarking, persson2025petab](@cite).
  Note, for models with many events, BDF solvers are often slow and in such cases
  `KenCarp4()` is a reliable alternative.
- **Gradient method:** Forward-mode automatic differentiation via
  [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is typically the most
  fasest choice in this regime.
- **Hessian method:** Computing the full Hessian with
  [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is often too expensive. A
  Gauss–Newton approximation is usually a good compromise and often outperforms (L)BFGS
  approximations in practice [frohlich2022fides, persson2025petab](@cite).

!!! note "Reusing sensitivities"
    For optimizers that evaluate gradient and Gauss-Newton Hessian together (e.g.
    Fides.jl), setting `gradient_method = :ForwardEquations` with
    `reuse_sensitivities = true` will reduce runtime. See Fides.jl in
    [Optimization algorithms and recommendations](@ref options_optimizers) for details.

## Large models (≥75 parameters or ≥75 ODEs)

Defaults are provided for large models, but benchmarking is strongly recommended since
solver choice and gradient configuration can strongly affect runtime. Still, the default
configuration is:

```julia
petab_prob = PEtabODEProblem(
    model;
    odesolver = ODESolver(CVODE_BDF()),
    gradient_method = :Adjoint,
    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
)
```

The rationale is:

- **ODE solver:** Large stiff models often benefit from solvers and linear algebra tailored
  for scale. Benchmark `QNDF()`, `FBDF()`, `KenCarp4()`, and `CVODE_BDF()`. Also consider
  enabling sparse Jacobians (`sparse_jacobian = true`) and trying alternative linear solvers
  (e.g. `CVODE_BDF(linsolve = :KLU)`). For guidance, see the DifferentialEquations.jl
  documentation.
- **Gradient method:** Adjoint sensitivities (`gradient_method = :Adjoint`) are typically
  the most efficient choice at this scale. PEtab.jl supports `InterpolatingAdjoint`,
  `GaussAdjoint`, and `QuadratureAdjoint` from
  [SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl). The default is
  `InterpolatingAdjoint(autojacvec = ReverseDiffVJP())`, but different adjoint algorithms
  and `autojacvec` options can perform very differently and should be benchmarked.
- **Hessian method:** Computing Gauss–Newton or full Hessians is usually too expensive for
  large models. In practice, (L)BFGS approximations are often the best option and are
  supported by common optimizers (e.g. Optim.jl, Ipopt.jl, and Fides.jl).

## References

```@bibliography
Pages = ["default_options.md"]
Canonical = false
```
