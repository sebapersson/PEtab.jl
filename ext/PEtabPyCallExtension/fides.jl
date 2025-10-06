function PEtab.calibrate_multistart(rng::Random.AbstractRNG, prob::PEtab.PEtabODEProblem,
                                    alg::PEtab.Fides, nmultistarts::Signed;
                                    save_trace::Bool = false,
                                    dirsave::Union{Nothing, String} = nothing,
                                    sampling_method::SamplingAlgorithm = LatinHypercubeSample(),
                                    sample_prior::Bool = true, nprocs::Int64 = 1,
                                    options = nothing)::PEtab.PEtabMultistartResult
    options = isnothing(options) ? py"{'maxiter' : 1000}"o : options
    return PEtab._calibrate_multistart(rng, prob, alg, nmultistarts, dirsave,
                                       sampling_method, options, sample_prior, save_trace,
                                       nprocs)
end

function PEtab.calibrate(prob::PEtabODEProblem,
                         x::Union{Vector{<:AbstractFloat}, ComponentArray}, alg::Fides;
                         save_trace::Bool = false,
                         options = py"{'maxiter' : 1000}"o)::PEtab.PEtabOptimisationResult
    xstart = x |> collect
    if save_trace == true
        @warn "For Fides the x and f trace cannot currently be saved (we are working on it)" maxlog=10
    end
    fides_prob = _get_fides_prob(prob, alg, options)
    # Create a runnable function taking parameter as input
    local niterations, fmin, _xmin, converged, runtime, ftrace, xtrace, res
    try
        runtime = @elapsed res, niterations, converged = fides_prob(xstart)
        fmin = res[1]
        _xmin = res[2]
        ftrace = Vector{Float64}(undef, 0)
        xtrace = Vector{Vector{Float64}}(undef, 0)
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
    alg_used = "Fides($(alg.hessian_method))" |> Symbol
    xnames_ps = propertynames(prob.xnominal_transformed)
    xstart = ComponentArray(; (xnames_ps .=> xstart)...)
    xmin = ComponentArray(; (xnames_ps .=> _xmin)...)
    return PEtabOptimisationResult(xmin, fmin, xstart, alg_used, niterations, runtime,
                                   xtrace, ftrace, converged, res)
end

function _get_fides_prob(prob::PEtabODEProblem, alg::Fides, options)::Function
    funargs = py"None"o
    resfun::Bool = false
    ub = prob.upper_bounds |> collect
    lb = prob.lower_bounds |> collect

    # The Fides objective functions depends on whether or not the user wants to use a
    # Hessian approximation or not
    if isnothing(alg.hessian_method)
        fides_fun = let prob = prob
            x -> _fides_obj_hess(x, prob)
        end
    else
        fides_fun = let prob = prob
            x -> _fides_obj_hess_approx(x, prob)
        end
    end

    fides_prob = _setup_fides(fides_fun, ub, lb, alg.verbose, options, funargs,
                              string(alg.hessian_method), resfun)
    return fides_prob
end

function _setup_fides(fun::Function, ub::Vector{Float64}, lb::Vector{Float64},
                      verbose::Bool, options, funargs, hessian_update::String,
                      resfun::Bool)::Function
    py"""
    import numpy as np
    import fides
    import logging

    def run_fides_python(x0, fun, ub, lb, verbose, options, funargs, hessian_update, resfun):
        if hessian_update == "BFGS":
            hessian_update = fides.hessian_approximation.BFGS()
        elif hessian_update == "BB":
            hessian_update = fides.hessian_approximation.BB()
        elif hessian_update == "BG":
            hessian_update = fides.hessian_approximation.BG()
        elif hessian_update == "Broyden":
            hessian_update = fides.hessian_approximation.Broyden()
        elif hessian_update == "DFB":
            hessian_update = fides.hessian_approximation.DFB()
        elif hessian_update == "FX":
            hessian_update = fides.hessian_approximation.FX()
        elif hessian_update == "SR1":
            hessian_update = fides.hessian_approximation.SR1()
        elif hessian_update == "SSM":
            hessian_update = fides.hessian_approximation.SSM()
        elif hessian_update == "TSSM":
            hessian_update = fides.hessian_approximation.TSSM()
        else:
            hessian_update = None

        fides_opt = fides.Optimizer(fun, ub, lb, verbose=verbose, options=options,
                                    funargs=funargs, hessian_update=hessian_update,
                                    resfun=resfun)
        res = fides_opt.minimize(x0)
        niter = fides_opt.iteration
        converged = fides_opt.converged
        return res, niter, converged

    """
    run_fides = (x0; verbose = verbose, options = options, funargs = funargs,
    hessian_update = hessian_update, resfun = resfun) -> begin
        py"run_fides_python"(x0, fun, ub, lb, verbose, options, funargs, hessian_update,
                             resfun)
    end
    return run_fides
end

function _fides_obj_hess(x::Vector{Float64},
                         prob::PEtabODEProblem)::Tuple{Float64, Vector{Float64},
                                                       Matrix{Float64}}
    nllh = prob.nllh(x)
    grad = prob.grad(x)
    hess = prob.hess(x)
    return (nllh, grad, hess)
end

function _fides_obj_hess_approx(x::Vector{Float64},
                                prob::PEtabODEProblem)::Tuple{Float64, Vector{Float64}}
    nllh, grad = prob.nllh_grad(x)
    return (nllh, grad)
end
