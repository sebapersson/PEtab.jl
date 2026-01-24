module PEtabFidesExtension

using Fides: HessianUpdate, CustomHessian, FidesOptions, FidesProblem, InputVector, solve
using Catalyst: @unpack
using ComponentArrays: ComponentArray
using PEtab
using QuasiMonteCarlo: LatinHypercubeSample, SamplingAlgorithm
import Random

const DEFAULT_OPT = FidesOptions()

function PEtab.calibrate_multistart(
        rng::Random.AbstractRNG, prob::PEtabODEProblem, alg::HessianUpdate,
        nmultistarts; nprocs = 1, save_trace::Bool = false, dirsave = nothing,
        sample_prior = true, sampling_method = LatinHypercubeSample(), init_weight = nothing,
        init_bias = nothing, options::Union{Nothing, FidesOptions} = DEFAULT_OPT
    )::PEtab.PEtabMultistartResult
    options = isnothing(options) ? DEFAULT_OPT : options

    return PEtab._calibrate_multistart(
        rng, prob, alg, nmultistarts, dirsave, sampling_method, options, sample_prior,
        save_trace, nprocs, init_weight, init_bias
    )
end

function PEtab.calibrate(prob::PEtabODEProblem, x::InputVector, alg::HessianUpdate;
                         save_trace::Bool = false, options::FidesOptions = FidesOptions())::PEtab.PEtabOptimisationResult
    if save_trace == true
        @warn "For Fides the x and f trace cannot currently be saved (we are working on it)" maxlog=10
        save_trace = false
    end
    ftrace = Vector{Float64}(undef, 0)
    xtrace = Vector{Vector{Float64}}(undef, 0)

    # If custom Hessian is used, evaluate objective once, to build internal structs
    # correctly
    if alg isa CustomHessian
        _ = prob.nllh(x)
    end

    fides_problem = _get_fides_problem(prob, x, alg)
    local xmin, fmin, converged, runtime, niterations, res
    try
        res = solve(fides_problem, alg; options = options)
        @unpack xmin, fmin, niterations, runtime = res
        converged = res.retcode
    catch
        niterations = 0
        fmin = NaN
        xmin = similar(x) .* NaN
        converged = :Optmisation_failed
        runtime = NaN
        res = nothing
    end
    alg_used = :Fides
    xnames_ps = propertynames(prob.xnominal_transformed)
    xstart = ComponentArray(; (xnames_ps .=> x)...)
    xmin = ComponentArray(; (xnames_ps .=> xmin)...)
    return PEtabOptimisationResult(xmin, fmin, xstart, alg_used, niterations, runtime,
                                   xtrace, ftrace, converged, res)
end

function _get_fides_problem(prob::PEtabODEProblem, x::InputVector,
                            ::CustomHessian)::FidesProblem
    fides_objective = let _prob = prob
        (x) -> begin
            f, grad = _prob.nllh_grad(x)
            hess = _prob.hess(x)
            return f, grad, hess
        end
    end
    return FidesProblem(fides_objective, x; lb = prob.lower_bounds, ub = prob.upper_bounds)
end
function _get_fides_problem(prob::PEtabODEProblem, x::InputVector, ::Any)::FidesProblem
    fides_objective = let _prob = prob
        (x) -> _prob.nllh_grad(x)
    end
    return FidesProblem(fides_objective, x; lb = prob.lower_bounds, ub = prob.upper_bounds)
end

end
