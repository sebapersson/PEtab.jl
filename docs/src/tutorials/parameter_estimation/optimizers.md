```@meta
CollapsedDocStrings=true
```

# [Optimization algorithms and recommendations](@id options_optimizers)

For [`calibrate`](@ref) and [`calibrate_multistart`](@ref), PEtab.jl supports optimizers
from three popular packages: [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl),
[Ipopt.jl](https://github.com/jump-dev/Ipopt.jl), and
[Fides.jl](https://fides-dev.github.io/Fides.jl/stable/). This page summarizes the available
algorithms and provides recommendations by model size.

## Recommended algorithm

When choosing an optimization algorithm, the **no free lunch** principle applies:
performance is problem-dependent and no method is universally best. Still, benchmark studies
have identified methods that often work well for ODE models in biology (and likely beyond)
[raue2013lessons, hass2019benchmark, villaverde2019benchmarking](@cite). In brief, the
choice is driven by model size: for small models an accurate Hessian (or good approximation)
is often feasible to compute, while for larger models it typically is not (see
[gradient and Hessian support](@ref gradient_support)). Thereby, we recommend:

- **Small models** (<10 ODEs and <20 estimated parameters): `Optim.IPNewton()` with the
  exact Hessian (`hessian_method = :ForwardDiff` in the `PEtabODEProblem`).
- **Medium-sized models** (≈10-20 ODEs and <75 estimated parameters):
  `Fides.CustomHessian()` with a Gauss-Newton Hessian approximation
  (`hessian_method = :GaussNewton` in the `PEtabODEProblem`). Gauss-Newton often outperforms
  the (L)BFGS approximation in this regime [frohlich2022fides](@cite).
- **Large models** (>20 ODEs or >75 estimated parameters): (L)BFGS methods such as Ipopt or
  `Fides.BFGS`.

## Fides

[Fides.jl](https://github.com/fides-dev/Fides.jl) is a trust-region Newton method for
box-constrained optimization [frohlich2022fides](@cite). It performs particularly well with
a Gauss-Newton Hessian approximation, but it also has built-in support for other
approximations (e.g., `BFGS`, `SR1`; see the Fides
[documentation](https://fides-dev.github.io/Fides.jl/stable/API/)).

Fides evaluates the objective, gradient, and Hessian in each iteration, enabling reuse of
intermediate quantities. Since the Gauss–Newton Hessian is based on forward sensitivities
(which can also be used to compute the gradient), a recommended `PEtabODEProblem`
configuration is:

```julia
using Fides
petab_prob = PEtabODEProblem(
    model;
    gradient_method = :ForwardEquations,
    hessian_method = :GaussNewton,
    reuse_sensitivities = true,
)
res = calibrate(
    petab_prob, x0, Fides.CustomHessian();
    options = FidesOptions(maxiter = 1000)
)
```

Fides solver options are set with `FidesOptions` (see the Fides
[documentation](https://fides-dev.github.io/Fides.jl/stable/API/)). Fides also provides
built-in Hessian approximations such as `BFGS`:

```julia
res = calibrate(petab_prob, x0, Fides.BFGS())
```

## [Optim.jl](@id Optim_alg)

PEtab.jl supports three algorithms from
[Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/): `LBFGS`, `BFGS`, and
`IPNewton` (interior-point Newton).

Solver settings are passed via `options = Optim.Options(...)` (see the Optim.jl
[documenation](https://julianlsolvers.github.io/Optim.jl/stable/user/config/#Solver-options)
page). For example, to run `Optim.LBFGS()` for 10_000 iterations:

```julia
using Optim
res = calibrate(
    petab_prob, x0, Optim.LBFGS();
    options = Optim.Options(iterations = 10_000)
)
```

## Ipopt

[Ipopt](https://coin-or.github.io/Ipopt/) is an interior-point Newton method for nonlinear
optimization [wachter2006implementation](@cite). In PEtab.jl, Ipopt can either use the
Hessian provided by the `PEtabODEProblem` or an L-BFGS Hessian approximation via
`IpoptOptimizer`:

```@docs; canonical=false
IpoptOptimizer
```

Ipopt exposes many solver options. A commonly used subset can be set via `IpoptOptions`:

```@docs; canonical=false
IpoptOptions
```

For example, to run Ipopt for 1000 iterations using the L-BFGS Hessian approximation:

```julia
using Ipopt
res = calibrate(
    petab_prob, x0, IpoptOptimizer(true);
    options = IpoptOptions(max_iter = 1000)
)
```

For details and the full option list, see the Ipopt documentation and the original
publication [wachter2006implementation](@cite).

!!! note
    To use Ipopt, load [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl) with
    `using Ipopt` before running parameter estimation.

## References

```@bibliography
Pages = ["optimizers.md"]
Canonical = false
```
