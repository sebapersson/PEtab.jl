# [Derivative methods (gradients and Hessians)](@id gradient_support)

When constructing a `PEtabODEProblem`, PEtab.jl supports several methods for computing
gradients and Hessians. This page summarizes the available methods and their tunable
options.

## Gradient methods

PEtab.jl supports three gradient methods for `PEtabODEProblem`: forward-mode automatic
differentiation (`:ForwardDiff`), forward-sensitivity equations (`:ForwardEquations`), and
adjoint sensitivity analysis (`:Adjoint`). Introductions to the underlying mathematics and
autodiff can be found in [sapienza2024differentiable, blondel2024elements](@cite). Below is
a brief overview.

- `:ForwardDiff`: Uses [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) to
  compute gradients with forward-mode automatic differentiation [revels2016forward](@cite).
  The main tuning option is `chunksize` (number of directional derivatives per pass). The
  default is usually good, but tuning can yield small speedups; this method is often fastest
  for small models [mester2022differential, persson2025petab](@cite).
- `:ForwardEquations`: Computes gradients via forward sensitivities by solving an expanded
  ODE system. The main option is `sensealg`. The default is `sensealg=:ForwarDiff`, which
  uses ForwardDiff-based sensitivity and is often the fastest. PEtab.jl also supports
  `ForwardSensitivity()` and `ForwardDiffSensitivity()` from SciMLSensitivity.jl for
  `sensealg`; see the SciMLSensitivity
  [documentation](https://github.com/SciML/SciMLSensitivity.jl) for details and tunable
  options.
- `:Adjoint`: Computes gradients via adjoint sensitivity analysis by solving an adjoint
  problem backward in time. Benchmarks often find adjoints most efficient for large models
  [frohlich2017scalable, ma2021comparison](@cite). The main option is `sensealg`, selecting
  a SciMLSensitivity adjoint algorithm (`InterpolatingAdjoint`, `GaussAdjoint`, or
  `QuadratureAdjoint`). See the SciMLSensitivity
  [documentation](https://github.com/SciML/SciMLSensitivity.jl) for tunable options.

!!! note "Using SciMLSensitivity methods"
    To use SciMLSensitivity-based methods (e.g. adjoints), load the package with
    `using SciMLSensitivity` before creating the `PEtabODEProblem`.

## Hessian methods

Three Hessian methods are supported; forward-mode automatic differentiation
(`:ForwardDiff`), a block approximation (`:BlockForwardDiff`), and a Gauss-Newton
approximation (`:GaussNewton`). Below is a brief overview.

- `:ForwardDiff`: Computes the full Hessian via forward-mode automatic differentiation with
  [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl). The main tuning option is
  `chunksize`. This method has quadratic cost in the number of estimated parameters
  ($\mathcal{O}(n^2)$), and is typically only feasible up to around `n ≈ 20` parameters.
  When feasible, access to the full Hessian can improve convergence, especially in
  multi-start estimation [persson2025petab](@cite).
- `:BlockForwardDiff`: Computes a block-diagonal Hessian approximation using forward-mode
  automatic differentiation with
  [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl). In many PEtab problems,
  parameters can be split into ODE parameters $\mathbf{x}_p$ and non-ODE parameters
  $\mathbf{x}_q$. This method computes each block Hessian while setting cross-terms to zero:

```math
\mathbf{H}_{\text{block}} =
\begin{bmatrix}
\mathbf{H}_{p} & \mathbf{0} \\
\mathbf{0} & \mathbf{H}_{q}
\end{bmatrix}
```

- `:GaussNewton`: Approximates the Hessian using the
  [Gauss–Newton](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm) method. It
  often performs better than (L)BFGS [frohlich2017scalable](@cite), but requires forward
  sensitivities (similar to `:ForwardEquations`). For models with many parameters
  (often >75), computing forward sensitivities can be too expensive; in that regime, (L)BFGS
  approximations are often the only practical option. For mathematical details, see
  [raue2015data2dynamics](@cite).

## References

```@bibliography
Pages = ["derivatives.md"]
Canonical = false
```
