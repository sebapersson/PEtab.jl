module PEtabLikelihoodProfilerExtension

import ComponentArrays
import PEtab
import LikelihoodProfiler: LikelihoodProfiler, OptimizationProblem

const FORBIDDEN_KWARGS = [:profile_lower, :profile_upper, :labels]

function LikelihoodProfiler.ProfileLikelihoodProblem(
        res::PEtab.EstimationResult, prob::PEtab.PEtabODEProblem; kwargs...
    )::LikelihoodProfiler.ProfileLikelihoodProblem
    keys_kwargs = collect(keys(kwargs))
    if !isempty(intersect(keys_kwargs, FORBIDDEN_KWARGS))
        key_value = keys_kwargs[findfirst(x -> x in FORBIDDEN_KWARGS, keys_kwargs)]
        throw(ArgumentError("Keyword argument `$(key_value)` is not allowed when building \
            a `ProfileLikelihoodProblem` from a `PEtabODEProblem`, as its value is \
            provided by the PEtab problem."))
    end

    opt_prob = OptimizationProblem(prob)
    x = PEtab._get_x(res)

    # Bounds must be finite for profiling
    lb = deepcopy(prob.lower_bounds)
    ub = deepcopy(prob.upper_bounds)
    lb[isinf.(abs.(lb))] .= -1.0e20
    ub[isinf.(ub)] .= 1.0e20
    labels_sym = Symbol.(ComponentArrays.labels(x))

    return LikelihoodProfiler.ProfileLikelihoodProblem(
        opt_prob, x; profile_lower = lb, profile_upper = ub, idxs = labels_sym, kwargs...
    )
end

end
