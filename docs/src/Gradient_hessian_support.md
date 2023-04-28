# [Supported gradient and hessian methods](@id gradient_support)

PEtab.jl supports several gradient and hessian methods when building a `PEtabODEProblem` via `setupPEtabODEProblem`. Here we briefly cover each method and its associated tuneable parameters.

## Gradient methods

* `:ForwardDiff`: Compute the gradient via forward mode automatic differentiation using [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl). Via the argument `chunkSize` the user can set the Chunk-size (see [here](https://juliadiff.org/ForwardDiff.jl/stable/)). This can improve performance, and we plan to add automatic tuning of it.
* `:ForwardEquations`: Compute the gradient via the forward sensitivities. Via the `sensealg` argument the user can choose method for computing sensitivities. We support both `ForwardSensitivity()` and `ForwardDiffSensitivity()` with tuneable options as provided by SciMLSensitivity (see their [documentation](https://github.com/SciML/SciMLSensitivity.jl) for info). The most efficient option though is `sensealg=:ForwardDiff` where forward mode automatic differentiation is used to compute the sensitivities.
* `:Adjoint`: Compute the gradient via adjoint sensitivity analysis. Via the `sensealg` argument the user can choose between the methods `InterpolatingAdjoint` and `QuadratureAdjoint` from SciMLSensitivity (see their [documentation](https://github.com/SciML/SciMLSensitivity.jl) for info). The user can provide any of the options that these methods accept.
* `:Zygote`: Compute the gradient using the [Zygote](https://github.com/FluxML/Zygote.jl) automatic differentiation library. Via the sensealg the user can choose any of the methods provided by [SciMLSensitivity](https://github.com/SciML/SciMLSensitivity.jl).
    * **Note**: As the code relies heavily on for-loops `:Zygote` is by far the slowest option and is not recommended.

## Hessian methods

* `:ForwardDiff`: Compute the hessian via forward mode automatic differentiation using [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl).
* `:BlockForwardDiff`: Compute a Hessian block approximation via forward mode automatic differentiation using [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl). In general for a PEtab model we have two set of parameters to estimate: parameters part of the ODE-system $\theta_p$ and parameter which are not a part of the ODE-system $\theta_q$. This approach computes the hessian for each block and assumes zero-valued cross-terms:

```math
    H_{block} = 
    \begin{bmatrix}
    H_{p} & \mathbf{0} \\
    \mathbf{0} & \mathbf{H}_q
    \end{bmatrix}
```

This approach often works well if the number of non-dynamic parameters in $\theta_q$ are few.

`:GaussNewton`: Compute a Hessian approximation via the [Gauss-Newton](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm) method. This method often performs better than a (L)-BFGS approximation. However, it requires access to sensitives which are only feasible to compute for smaller models ($\leq 75$ parameters).
