module PEtabLikelihoodProfilerExtension

import PEtab
import LikelihoodProfiler: LikelihoodProfiler, OptimizationProblem

function LikelihoodProfiler.ProfileLikelihoodProblem(
        res::PEtab.EstimationResult, prob::PEtab.PEtabODEProblem; kwargs...
    )::LikelihoodProfiler.ProfileLikelihoodProblem
    opt_prob = OptimizationProblem(prob)
    x = PEtab._get_x(res)

    # Bounds must be finite for profiling
    lb = deepcopy(prob.lower_bounds)
    ub = deepcopy(prob.upper_bounds)
    lb[isinf.(abs.(lb))] .= -1.0e20
    ub[isinf.(ub)] .= 1.0e20

    return LikelihoodProfiler.ProfileLikelihoodProblem(
        opt_prob, x; profile_lower = lb, profile_upper = ub, kwargs...
    )
end

end
