module PEtabOptimizationExtension

using SciMLBase
using QuasiMonteCarlo
using Random
using PEtab
using Catalyst: @unpack
using Optimization: OptimizationProblem, OptimizationFunction

function SciMLBase.OptimizationProblem(
        prob::PEtabODEProblem; box_constraints::Bool = true
    )::OptimizationProblem
    # OptimizationFunction with PEtab.jl objective, gradient and Hessian
    _f = (u, p) -> prob.nllh(u)
    _grad! = (G, u, p) -> prob.grad!(G, u)
    _hess! = (H, u, p) -> prob.hess!(H, u)
    optf = OptimizationFunction(_f; grad = _grad!, hess = _hess!)

    # OptimizationProblem
    @unpack lower_bounds, upper_bounds = prob
    x0 = deepcopy(prob.xnominal_transformed)
    if box_constraints == false
        lower_bounds, upper_bounds = nothing, nothing
    end
    return OptimizationProblem(
        optf, x0, SciMLBase.NullParameters(); lb = lower_bounds, ub = upper_bounds
    )
end

end
