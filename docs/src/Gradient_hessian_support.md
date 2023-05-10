# [Supported gradient and hessian methods](@id gradient_support)

PEtab.jl offers various gradient and Hessian methods that can be used to build a `PEtabODEProblem` using `createPEtabODEProblem()`. In this section, we will provide a brief overview of each method and the corresponding adjustable parameters.

## Gradient methods

* `:ForwardDiff`: Uses [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) to compute the gradient via forward mode automatic differentiation. You can set the chunk size using the `chunkSize` argument to improve performance. We plan to add automatic tuning for this in the future.
* `:ForwardEquations`: Computes the gradient via the forward sensitivities. You can choose the method for computing sensitivities using the `sensealg` argument. We support both `ForwardSensitivity()` and `ForwardDiffSensitivity()`, which have adjustable options provided by SciMLSensitivity (see their [documentation](https://github.com/SciML/SciMLSensitivity.jl)). The most efficient option is `sensealg=:ForwardDiff` though, which uses forward mode automatic differentiation to compute sensitivities.
* `:Adjoint`: Computes the gradient via adjoint sensitivity analysis. You can choose between the `InterpolatingAdjoint` and `QuadratureAdjoint` methods from SciMLSensitivity (see their [documentation](https://github.com/SciML/SciMLSensitivity.jl)) using the `sensealg` argument. You can provide any options accepted by these methods.
* `:Zygote`: Computes the gradient using the [Zygote](https://github.com/FluxML/Zygote.jl) automatic differentiation library. You can choose any of the methods provided by [SciMLSensitivity](https://github.com/SciML/SciMLSensitivity.jl) using the `sensealg` argument.
    * **Note**: Because the code uses many for-loops, `:Zygote` is the slowest option and not recommended.

## Hessian methods

* `:ForwardDiff`: This method computes the Hessian via forward mode automatic differentiation using [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl). You can use the `chunkSize` argument to set the chunk size, which can help improve performance. In the future, we plan to add automatic tuning for this parameter.
* `:BlockForwardDiff`: This method computes a Hessian block approximation via forward mode automatic differentiation using [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl). For PEtab models, there are typically two sets of parameters to estimate: the parameters that are part of the ODE system $\theta_p$ and those that are not $\theta_q$. This method computes the Hessian for each block and assumes that cross-terms are zero-valued. The resulting Hessian block takes the form:

```math
H_{block} = 
\begin{bmatrix}
H_{p} & \mathbf{0} \\
\mathbf{0} & \mathbf{H}_q
\end{bmatrix}
```
    
* `:GaussNewton`: This method computes a Hessian approximation using the [Gauss-Newton](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm) method. It often performs better than a (L)-BFGS approximation, but requires access to sensitivities, which may only be feasible to compute for smaller models with 75 or fewer parameters. 
