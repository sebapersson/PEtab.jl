module PEtabOptimExtension

using Optim
import QuasiMonteCarlo: LatinHypercubeSample, SamplingAlgorithm
import Random
using Setfield
using Catalyst: @unpack
using ComponentArrays
using PEtab

const DEFAULT_OPT = Optim.Options(
    iterations = 1000, show_trace = false, allow_f_increases = true, successive_f_tol = 3,
    f_reltol = 1e-8, g_tol = 1e-6, x_abstol = 0.0
)

const SUPPORTED_ALGS = Union{Optim.LBFGS, Optim.BFGS, Optim.IPNewton}

function PEtab.calibrate_multistart(
        rng::Random.AbstractRNG, prob::PEtabODEProblem, alg::SUPPORTED_ALGS,
        nmultistarts; nprocs = 1, save_trace = false, dirsave = nothing,
        sample_prior = true, sampling_method = LatinHypercubeSample(),
        init_weight = nothing, init_bias = nothing,
        options::Union{Optim.Options, Nothing} = nothing
    )::PEtab.PEtabMultistartResult
    options = isnothing(options) ? DEFAULT_OPT : options

    return PEtab._calibrate_multistart(
        rng, prob, alg, nmultistarts, dirsave, sampling_method, options, sample_prior,
        save_trace, nprocs, init_weight, init_bias
    )
end

function PEtab.calibrate(prob::PEtabODEProblem,
                         x::Union{Vector{<:AbstractFloat}, ComponentArray},
                         alg::Union{Optim.LBFGS, Optim.BFGS, Optim.IPNewton};
                         save_trace::Bool = false,
                         options::Optim.Options = DEFAULT_OPT)::PEtab.PEtabOptimisationResult
    options = @set options.store_trace = save_trace
    options = @set options.extended_trace = save_trace
    xstart = x |> collect
    optim_problem = _get_optim_problem(prob, alg, options)

    local niterations, fmin, _xmin, converged, runtime, ftrace, xtrace, res
    try
        res = optim_problem(xstart)
        niterations = Optim.iterations(res)
        fmin = Optim.minimum(res)
        _xmin = Optim.minimizer(res)
        converged = Optim.converged(res)
        runtime = res.time_run # In seconds
        if save_trace == true
            ftrace = Optim.f_trace(res)
            xtrace = Optim.x_trace(res)
        else
            ftrace = Vector{Float64}(undef, 0)
            xtrace = Vector{Vector{Float64}}(undef, 0)
        end
    catch
        niterations = 0
        fmin = NaN
        _xmin = similar(xstart) .* NaN
        ftrace = Vector{Float64}(undef, 0)
        xtrace = Vector{Vector{Float64}}(undef, 0)
        converged = :Optmisation_failed
        runtime = NaN
        res = nothing
    end
    xnames_ps = propertynames(prob.xnominal_transformed)
    xstart = ComponentArray(; (xnames_ps .=> xstart)...)
    xmin = ComponentArray(; (xnames_ps .=> _xmin)...)

    if alg isa Optim.IPNewton
        alg_used = :Optim_IPNewton
    elseif alg isa Optim.BFGS
        alg_used = :Optim_BFGS
    elseif alg isa Optim.LBFGS
        alg_used = :Optim_LBFGS
    end
    return PEtabOptimisationResult(xmin, fmin, xstart, alg_used, niterations, runtime,
                                   xtrace, ftrace, converged, res)
end

function _get_optim_problem(prob::PEtabODEProblem, alg, options::Optim.Options)::Function
    if alg isa IPNewton
        return _optim_IPNewton(prob, alg, options)
    else
        return _optim_fminbox(prob, alg, options)
    end
end

function _optim_IPNewton(prob::PEtabODEProblem, alg::Optim.IPNewton,
                         options::Optim.Options)::Function
    @unpack lower_bounds, upper_bounds = prob
    lb = lower_bounds |> collect
    ub = upper_bounds |> collect
    nparameters = length(lb)
    x0 = zeros(Float64, nparameters)
    df = TwiceDifferentiable(prob.nllh, prob.grad!, prob.hess!, x0)
    dfc = TwiceDifferentiableConstraints(lb, ub)

    _calibrate = (x) -> begin
        # Move points within bounds, and IPNewton does not accept points on the border
        ibelow = x .<= prob.lower_bounds
        iabove = x .>= prob.upper_bounds
        x[ibelow] .= prob.lower_bounds[ibelow] .+ 1e-6
        x[iabove] .= prob.upper_bounds[iabove] .- 1e-6
        # Need to evaluate nllh to be able to compute the Hessian
        df.f(x)
        return Optim.optimize(df, dfc, x, alg, options)
    end
    return _calibrate
end

function _optim_fminbox(prob::PEtabODEProblem, alg, options::Optim.Options)::Function
    @unpack lower_bounds, upper_bounds = prob
    lb = lower_bounds |> collect
    ub = upper_bounds |> collect

    _calibrate = (x) -> Optim.optimize(prob.nllh, prob.grad!, lb, ub, x, Optim.Fminbox(alg),
                                       options)
    return _calibrate
end

end
