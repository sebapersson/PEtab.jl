module PEtabOptimizationExtension

import Catalyst: @unpack
import Optimization: OptimizationFunction, OptimizationProblem
import PEtab
import SciMLBase

function SciMLBase.OptimizationProblem(
        prob::PEtab.PEtabODEProblem; box_constraints::Bool = true
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
