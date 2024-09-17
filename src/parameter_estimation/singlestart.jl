"""
    calibrate_model(prob::PEtabODEProblem, x::Vector, alg; kwargs...)

Parameter estimate a model for a PEtabODEProblem using an optimization algorithm `alg` and an initial guess `p0`.

The optimization algorithm `alg` can be one of the following:
- [Optim](https://julianlsolvers.github.io/Optim.jl/stable/) LBFGS, BFGS, or IPNewton methods
- [IpoptOptimiser](https://coin-or.github.io/Ipopt/) interior-point optimizer
- [Fides](https://github.com/fides-dev/fides) Newton trust region method

Each algorithm accepts specific optimizer options in the format of the respective package. For a
comprehensive list of available options, please refer to the main documentation.


!!! note
    To use Optim optimizers, you must load Optim with `using Optim`. To use Ipopt, you must load Ipopt with `using Ipopt`.
    To use Fides, load PyCall with `using PyCall` and ensure Fides is installed (see documentation for setup).

## Examples
```julia
# Perform parameter estimation using Optim's IPNewton with a given initial guess
using Optim
res = calibrate_model(prob, p0, Optim.IPNewton();
                     options=Optim.Options(iterations = 1000))
```
```julia
# Perform parameter estimation using Fides with a given initial guess
using PyCall
res = calibrate_model(prob, p0, Fides(nothing);
                     options=py"{'maxiter' : 1000}"o)
```
```julia
# Perform parameter estimation using Ipopt and save the trace
using Ipopt
res = calibrate_model(prob, p0, IpoptOptimiser(false);
                     options=IpoptOptions(max_iter = 1000),
                     save_trace=true)
```
```julia
# Perform parameter estimation using Optimization
using Optimization
using OptimizationOptimJL
prob = PEtab.OptimizationProblem(prob, interior_point_alg=true)
res = calibrate_model(prob, prob, p0, IPNewton())
```
"""
function calibrate end

"""
    OptimizationProblem(prob::PEtabODEProblem;
                        interior_point_alg::Bool = false,
                        box_constraints::Bool = true)

Create an Optimization.jl `OptimizationProblem` from a `PEtabODEProblem`.

To utilize interior-point Newton methods (e.g. Optim `IPNewton` or `Ipopt`), set `interior_point_alg` to true.

To use algorithms not compatible with box-constraints (e.g., `NewtonTrustRegion`), set `box_constraints` to false.
Note, with this options optimizers may move outside exceed the parameter bounds in the `prob`, which can
negatively impact performance.

# Examples
```julia
# Use IPNewton with startguess u0
using OptimizationOptimJL
prob = PEtab.OptimizationProblem(prob, interior_point=true)
prob.u0 .= u0
sol = solve(prob, IPNewton())
```
```julia
# Use Optim ParticleSwarm with startguess u0
using OptimizationOptimJL
prob = PEtab.OptimizationProblem(prob)
prob.u0 .= u0
sol = solve(prob, Optim.ParticleSwarm())
```
"""
function OptimizationProblem end
