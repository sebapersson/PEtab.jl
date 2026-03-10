# [Default PEtabODEProblem options](@id default_options)

A `PEtabODEProblem` supports multiple gradient/Hessian computation methods and the ODE
solvers from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl). This leads to
many valid `PEtabODEProblem` configurations. To simplify usage, PEtab.jl provides default
options based on an extensive benchmark study [persson2025petab](@cite). This page
summarizes these defaults.

In brief, the defaults depend on problem size (number of ODE states and number of estimated
parameters), since ODE solver performance and especially gradient-computation methods depend
strongly on size. Defaults further depend on problem type: SciML problems (i.e. embedding ML
models) have different characteristics and therefore use different defaults than purely
mechanistic problems. For this reason, the page is split into mechanistic and SciML models.

::: tip Non-stiff models

Defaults are tuned for biological, typically stiff models. For non-stiff models, see
[Speeding up non-stiff models](@ref nonstiff_models).

::::

## Mechanistic problems

This section outlines the default options for mechanistic models. The primary determinant of
the default configuration is model size.

### Small models (≤20 parameters and ≤15 ODEs)

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
  small models, often faster than sensitivity-equation approaches implemented in tools such
  as AMICI. Performance can sometimes be improved by tuning the ForwardDiff chunk size, but
  the optimal value is model-dependent.
- **Hessian method:** Computing the full Hessian with
  [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is often feasible for small
  models, and access to the full Hessian typically improves parameter estimation
  performance.

### Medium-sized models (≤75 parameters and ≤75 ODEs)

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
  [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is typically the most fasest
  choice in this regime.
- **Hessian method:** Computing the full Hessian with
  [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is often too expensive. A
  Gauss–Newton approximation is usually a good compromise and often outperforms (L)BFGS
  approximations in practice [frohlich2022fides, persson2025petab](@cite).

::: tip Reusing sensitivities

For optimizers that evaluate gradient and Gauss-Newton Hessian together (e.g. Fides.jl),
setting `gradient_method = :ForwardEquations` with `reuse_sensitivities = true` will reduce
runtime. See Fides.jl in [Optimization algorithms and recommendations](@ref
options_optimizers) for details.

:::

### Large models (≥75 parameters or ≥75 ODEs)

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

## SciML problems

This section outlines default options for SciML problems, where one or more ML models are
combined with a mechanistic ODE model. Defaults are summarized for each SciML problem type.

### ML model inside ODE equations (UDE or Neural ODE)

The default options for these problems are:

```julia
petab_prob = PEtabODEProblem(
    model;
    odesolver = ODESolver(Tsit5()),
    gradient_method = :ForwardDiff,
)
```

when the number of estimated parameters is ≤ 75. For models with > 75 estimated parameters,
the default is:

```julia
petab_prob = PEtabODEProblem(
    model;
    odesolver = ODESolver(Tsit5()),
    gradient_method = :Adjoint,
    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
)
```

The rationale is:

- **ODE solver:** UDEs and Neural ODEs are often less stiff than mechanistic models, so an
  explicit solver such as `Tsit5()` is a good default as it speeds up computations. If
  solver warnings indicate stiffness, switching to a stiff solver is a good choice.
- **Gradient method:** As explained above for mechanistic models, for smaller problems,
  forward-mode AD (`:ForwardDiff`) is feasible and reliable. For larger problems, adjoint
  sensitivities (`:Adjoint`) are typically more efficient.
- **Hessian method:** As for mechanistic models, computing full or Gauss–Newton Hessians is
  often too expensive for SciML problems due to the large number of estimated parameters.

### ML model in observable

In this scenario, an ML model appears only in the observable formula(s). Defaults are the
same as for mechanistic models, that is determined by the ODE model size
(small/medium/large) as described above.

The rationale is that parameters that appear only in observable formulas do not affect the
ODE dynamics. Gradients with respect to ML parameters can therefore be computed without
differentiating through the ODE solver (the most computationally expensive step). Instead,
given the ODE solution, PEtab.jl differentiates the observable computation directly using
forward- or reverse-mode AD (chosen automatically). As a result, the computational
bottleneck remains gradients with respect to variables that affect the ODE solution, so
defaults are determined by the characteristics of the ODE model.

### Pre-simulation ML model

In this scenario, ML model(s) map input data (e.g. high-dimensional images) to ODE
parameters and/or initial conditions before simulating the ODE model. The default options
are:

```julia
# Options are the same as for mechanistic models as
# described above, except `split_over_conditions`
petab_prob = PEtabODEProblem(
    model;
    split_over_conditions = true,
)
```

The rationale is that when the ML model is evaluated outside the ODE dynamics, ML gradients
can be computed in three steps: (1) compute the Jacobian of the ML model output with respect
to its parameters; (2) compute the gradient of the objective with respect to the ODE
parameters (including those set by the ML model); and (3) obtain the ML-parameter gradient
via a Jacobian-vector product between the Jacobian from (1) and the gradient from (2). This
separates ML-parameter gradients from mechanistic gradients, so defaults are determined by
the characteristics of the ODE model. In addition, setting `split_over_conditions = true`
enables compilation/reuse of the ML model reverse pass when computing the Jacobian, which
reduces gradient runtime.

## References

```@bibliography
Pages = ["default_options.md"]
Canonical = false
```
