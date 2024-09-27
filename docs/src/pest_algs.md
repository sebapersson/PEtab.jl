# [Available and Recommended Optimization Algorithms](@id options_optimizers)

For the `calibrate` and `calibrate_multistart` functions, PEtab.jl supports optimization algorithms from several popular optimization packages: [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl), and [Fides.py](https://github.com/fides-dev/fides). This page provides information on each package, as well as recommendations.

## Recommended Optimization Algorithm

When choosing an optimization algorithm, it is important to keep the **no free lunch** principle in mind: while an algorithm may work well for one problem, there is no universally best method. Nevertheless, benchmark studies have identified algorithms that often perform well for ODE models in biology (and likely beyond) [raue2013lessons, hass2019benchmark, villaverde2019benchmarking](@cite). In particular, the best algorithm to use depends on the size of the parameter estimation problem. This is because the problem considered here is a non-linear continuous optimization problem, and for such problems, having access to a good Hessian approximation improves performance. And, the problem size dictates which type of Hessian approximation can be computed (see this [page](@ref gradient_support) for more details). Following this, we recommend:

- For **small** models (fewer than 10 ODEs and fewer than 20 parameters to estimate) where computing the Hessian is often computationally feasible, the `IPNewton()` method from Optim.jl.
- For **medium sized** models (roughly more than 10 ODEs and fewer than 75 parameters), where a Gauss-Newton Hessian can be computed, Fides. The Gauss-Newton Hessian approximation typically outperforms the more common (L)-BFGS approximation, and benchmarks have shown that Fides performs well with such a Hessian approximation [frohlich2022fides](@cite). If Fides is difficult to install, `Optim.BFGS` also performs well.
- For **large** models (more than 20 ODEs and more than 75 parameters to estimate), where a Gauss-Newton approximation is too computationally expensive, a (L)-BFGS optimizer is recommended, such as Ipopt or `Optim.BFGS`.

## [Optim.jl](@id Optim_alg)

PEtab.jl supports three optimization algorithms from [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/): `LBFGS`, `BFGS`, and `IPNewton` (Interior-point Newton). Options for these algorithms can be specified via `Optim.Options()`, and a complete list of options can be found [here](https://julianlsolvers.github.io/Optim.jl/v0.9.3/user/config/). For example, to use `LBFGS` with 10,000 iterations, do:

```julia
using Optim
res = calibrate(petab_prob, x0, Optim.LBFGS();
                options=Optim.Options(iterations = 10000))
```

If no options are provided, the default ones are used:

```julia
Optim.Options(iterations = 1000,
              show_trace = false,
              allow_f_increases = true,
              successive_f_tol = 3,
              f_tol = 1e-8,
              g_tol = 1e-6,
              x_tol = 0.0)
```

For more details on each algorithm and tunable options, see the Optim.jl [documentation](https://julianlsolvers.github.io/Optim.jl/stable/).

## Ipopt

[Ipopt](https://coin-or.github.io/Ipopt/) is an Interior-point Newton method for nonlinear optimization [wachter2006implementation](@cite). In PEtab.jl, Ipopt can be configured to either use the Hessian from the `PEtabODEProblem` or a LBFGS Hessian approximation through the `IpoptOptimizer`:

```@docs; canonical=false
IpoptOptimizer
```

Ipopt offers a wide range of options (perhaps too many, in the words of the authors). A subset of these options can be specified using `IpoptOptions`:

```@docs; canonical=false
IpoptOptions
```

For example, to use Ipopt with 10,000 iterations and the LBFGS Hessian approximation, do:

```julia
using Ipopt
res = calibrate(petab_prob, x0, IpoptOptimizer(true); 
                options=IpoptOptions(max_iter = 10000))
```

For more information on Ipopt and its available options, see the Ipopt [documentation](https://coin-or.github.io/Ipopt/) and the original publication [wachter2006implementation](@cite).

!!! note
    To use Ipopt, the [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl) package must be loaded with `using Ipopt` before running parameter estimation.

## Fides

[Fides.py](https://github.com/fides-dev/fides) is a trust-region Newton method designed for box-constrained optimization problems [frohlich2022fides](@cite). It is particularly efficient when the Hessian is approximated using the [Gauss-Newton](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm) method.

The only drawback with Fides is that it is a Python package, but fortunately, it can be used from PEtab.jl through PyCall. To this end, you must build PyCall with a Python environment that has Fides installed:

```julia
using PyCall
# Path to Python executable with Fides installed
path_python_exe = "path_python"
ENV["PYTHON"] = path_python_exe
# Build PyCall with the Fides Python environment
import Pkg
Pkg.build("PyCall")
```

Fides supports several Hessian approximations, which can be specified in the `Fides` constructor:

```@docs; canonical=false
Fides
```

A notable feature of Fides is that in each optimization step, the objective, gradient, and Hessian are computed simultaneously. This opens up the possibility for efficient reuse of computed quantities, especially when the Hessian is computed via the Gauss-Newton approximation. Because, to compute the Gauss-Newton Hessian the forward sensitivities are used, which can also be used to compute the gradient. Hence, a good `PEtabODEProblem` configuration for Fides with Gauss-Newton is:

```julia
petab_prob = PEtabODEProblem(model; gradient_method = :ForwardEquations, 
                             hessian_method = :GaussNewton,
                             reuse_sensitivities = true)
```

Given this setup, the Hessian method from the `PEtabODEProblem` can be used to run Fides for 200 iterations with:

```julia
using PyCall
res = calibrate(petab_prob, x0, Fides(nothing);
                options=py"{'maxiter' : 1000}")
```

As noted above, for Fides options are specified using a Python dictionary. Available options and their default values can be found in the Fides [documentation](https://fides-optimizer.readthedocs.io/en/latest/generated/fides.constants.html), and more information on the algorithm can be found in the original publication [frohlich2022fides](@cite).

## References

```@bibliography
Pages = ["pest_algs.md"]
Canonical = false
```
