# [Gradient and Hessian methods](@id gradient_support)

PEtab.jl supports several gradient and Hessian computation methods to choose between when creating a `PEtabODEProblem`. This section provides a brief overview of each available method and the corresponding tunable parameters.

## Gradient Methods

PEtab.jl supports three gradient computation methods: forward-mode automatic differentiation (`:ForwardDiff`), forward-sensitivity equations (`:ForwardEquations`), and adjoint sensitivity analysis (`:Adjoint`). A good introduction to the math behind these methods can be found in, and a good introduction to automatic differentitation can be found in. Below is a brief description of each option.

- `:ForwardDiff`: This method uses [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) to compute the gradient via forward-mode automatic differentiation. The only tunable option is the `chunksize` (the number of directional derivatives computed in a single forward pass). While the default `chunksize` is typically a good choice, performance can be slightly improved by tuning this parameter, and we plan to add automatic tuning. This method is often the fastest for smaller models, while for medium-sized models `:ForwardEquations` may be more efficient for certain optimization algorithms (for details see ADD!).
- `:ForwardEquations`: This method computes the gradient by solving an expanded ODE system for the forward sensitivities during the forward pass, which are then used to compute the gradient. The tunable option is `sensealg`, where the default option `sensealg=:ForwardDiff` (which computes sensitivities via forward-mode automatic differentiation) is often the fastest. PEtab.jl also supports the `ForwardSensitivity()` and `ForwardDiffSensitivity()` methods from [SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl). For more details and tunable options for these two methods, see the SciMLSensitivity [documentation](https://github.com/SciML/SciMLSensitivity.jl).
- `:Adjoint`: This method computes the gradient via adjoint sensitivity analysis. Several benchmark studies have shown that adjoint methods are the most efficient for larger models. The tunable option is `sensealg`, which specifies the adjoint algorithm from SciMLSensitivity to use. Available methods are `InterpolatingAdjoint`, `GaussAdjoint`, and `QuadratureAdjoint`, and for information on their tunable options, see the SciMLSensitivity [documentation](https://github.com/SciML/SciMLSensitivity.jl).

!!! note
    To use functionality from SciMLSensitivity (e.g., adjoint sensitivity analysis), the package must be loaded with `using SciMLSensitivity` before creating the `PEtabODEProblem`.

## Hessian Methods

PEtab.jl supports three Hessian computation methods: forward-mode automatic differentiation (`:ForwardDiff`), a block Hessian approximation (`:BlockForwardDiff`), and the Gauss-Newton Hessian approximation (`:GaussNewton`). Below is a brief description of each option.

`:ForwardDiff`: Computes the Hessian via forward-mode automatic differentiation using [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl). As with the gradient, the only tunable option is the `chunksize`. This method has quadratic complexity, ``\\mathcal{O}(n^2)``, where ``n`` is the number of parameters, making it feasible only for models with up to ``n = 20`` parameters. However, when computationally feasible, access to the full Hessian can improve the convergence of parameter estimation runs when doing multi-start parameter estimation.

- `:BlockForwardDiff`: Computes a block Hessian approximation using forward-mode automatic differentiation with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl). Specifically, for PEtab problems, there are typically two sets of parameters to estimate: the parameters that are part of the ODE system, ``x_p``, and those that are not, ``x_q``. This block approach computes the Hessian for each block while approximating the cross-terms as zero:

```math
\mathbf{H}_{block} =
\begin{bmatrix}
\mathbf{H}_{p} & \mathbf{0} \\
\mathbf{0} & \mathbf{H}_q
\end{bmatrix}
```

- `:GaussNewton`: Approximates the Hessian using the [Gauss-Newton](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm) method. This method often performs better than a (L)-BFGS approximation but requires access to forward sensitivities (similar to `:ForwardEquations` above), and computing these for models with more than 75 parameters is often not computationally feasible. Therefore, for larger models, a (L)-BFGS approximation is often the only feasible option.
