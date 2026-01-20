"""
    calibrate(prob::PEtabODEProblem, x0, alg; kwargs...)::PEtabOptimisationResult

From starting point `x0` using optimization algorithm `alg`, estimate unknown model
parameters for `prob`, and get results as a `PEtabOptimisationResult`.

`x0` can be a `Vector` or a `ComponentArray`, where the individual parameters must be in the
order expected by `prob`. To get a vector in the correct order, see [`get_x`](@ref).

A list of available and recommended optimization algorithms (`alg`) can be found in the
documentation. Briefly, supported algorithms are from:

- [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/): `LBFGS()`, `BFGS()`,
    or `IPNewton()` methods.
- [Ipopt.jl](https://coin-or.github.io/Ipopt/): `IpoptOptimizer()` interior-point Newton
    method.
- [Fides.jl](https://fides-dev.github.io/Fides.jl/stable/): Newton trust region for
    box-constrained problem. Fides can either use the Hessian in `prob`
    (`alg = Fides.CustomHessian()`), or any of its built in Hessian approximations (e.g.
    `alg = Fides.BFGS()`). A full list of Hessian approximations can be found in the Fides
    [documentation](https://fides-dev.github.io/Fides.jl/stable/)

Different ways to visualize the parameter estimation result can be found in the
documentation.

See also [`PEtabOptimisationResult`](@ref) and [`IpoptOptimizer`](@ref)

## Keyword Arguments
- `save_trace::Bool = false`: Whether to save the optimization trace of the objective
    function and parameter vector. Only applicable for some algorithms; see the
    documentation for details.
- `options = DEFAULT_OPTIONS`: Configurable options for `alg`. The type and available
    options depend on which package `alg` belongs to. For example, if `alg = IPNewton()`
    from Optim.jl, `options` should be provided as an `Optim.Options()` struct. A list of
    configurable options can be found in the documentation.
"""
function calibrate end

"""
    OptimizationProblem(prob::PEtabODEProblem; box_constraints::Bool = true)

Create an [Optimization.jl](https://github.com/SciML/Optimization.jl) `OptimizationProblem`
from `prob`.

To use algorithms not compatible with box constraints (e.g., Optim.jl `NewtonTrustRegion`),
set `box_constraints = false`. Note that with this option, optimizers may move outside
the bounds, which can negatively impact performance. More information on how to use an
`OptimizationProblem` can be found in the Optimization.jl
[documentation](https://docs.sciml.ai/Optimization/stable/).
"""
function OptimizationProblem end
