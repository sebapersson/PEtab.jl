#=
    Fides wrapper 
=#


function PEtab.calibrate_model_multistart(petab_problem::PEtab.PEtabODEProblem, 
                                          alg::PEtab.Fides, 
                                          n_multistarts::Signed, 
                                          dir_save::Union{Nothing, String};
                                          sampling_method::T=QuasiMonteCarlo.LatinHypercubeSample(),
                                          options=py"{'maxiter' : 1000}"o,
                                          seed::Union{Nothing, Integer}=nothing, 
                                          save_trace::Bool=false)::PEtab.PEtabMultistartOptimisationResult where T <: QuasiMonteCarlo.SamplingAlgorithm
    if !isnothing(seed)
        Random.seed!(seed)
    end
    res = PEtab._multistartModelCallibration(petab_problem, alg, n_multistarts, dir_save, sampling_method, options, save_trace)
    return res
end


function PEtab.calibrate_model(petab_problem::PEtabODEProblem, 
                               p0::Vector{Float64},
                               alg::Fides; 
                               save_trace::Bool=false, 
                               options=py"{'maxiter' : 1000}"o)::PEtab.PEtabOptimisationResult

    _p0 = deepcopy(p0)                               

    if save_trace == true                         
        @warn "For Fides the x and f trace cannot currently be saved (we are working on it)" maxlog=10     
    end          

    runFides = createFidesProblem(petab_problem, alg, options=options)

    # Create a runnable function taking parameter as input                            
    local n_iterations, fmin, xmin, converged, runtime, ftrace, xtrace
    try
        runtime = @elapsed res, n_iterations, converged = runFides(p0)
        fmin = res[1]
        xmin = res[2]
        ftrace = Vector{Float64}(undef, 0)
        xtrace = Vector{Vector{Float64}}(undef, 0)
    catch
        n_iterations = 0
        fmin = NaN
        xmin = similar(p0) .* NaN
        ftrace = Vector{Float64}(undef, 0)
        xtrace = Vector{Vector{Float64}}(undef, 0)
        converged = :Code_crashed
        runtime = NaN
    end
    algUsed = :Fides

    return PEtabOptimisationResult(algUsed,
                                   xtrace, 
                                   ftrace, 
                                   n_iterations, 
                                   fmin, 
                                   _p0,
                                   xmin, 
                                   converged, 
                                   runtime)
end


function createFidesProblem(petab_problem::PEtabODEProblem,
                            fidesSetting::Fides; # In case you want to use any Fides optimizer
                            options=py"{'maxiter' : 1000}"o,
                            funargs=py"None"o,
                            resfun::Bool=false)

    nParam = length(petab_problem.lower_bounds)
    if !isnothing(fidesSetting.hessian_method)
        useHessianApproximation = true
    else
        # Put hessian function into acceptable Fides format
        useHessianApproximation = false
        hessian = zeros(Float64, (nParam, nParam))
        compute_hessian! = (p) -> evalAutoDiffHess(p, petab_problem.compute_hessian!, hessian)
    end

    gradient = zeros(Float64, nParam)
    compute_gradient! = (p) -> evalAutoDiffGrad(p, petab_problem.compute_gradient!, gradient)

    # Fides objective funciton
    if useHessianApproximation == false
        fidesFunc = (p) -> fidesObjHess(p, petab_problem.compute_cost, compute_gradient!, compute_hessian!)
    else
        fidesFunc = (p) -> fidesObjApprox(p, petab_problem.compute_cost, compute_gradient!)
    end

    # Set up a runnable executeble for Fides
    fidesObj = setUpFidesClass(fidesFunc, petab_problem.upper_bounds, petab_problem.lower_bounds,
                               fidesSetting.verbose,
                               options,
                               funargs,
                               string(fidesSetting.hessian_method),
                               resfun)

    return fidesObj
end


function setUpFidesClass(fun,
                         ub,
                         lb,
                         verbose,
                         options,
                         funargs,
                         hessian_update,
                         resfun::Bool)

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

        fides_opt = fides.Optimizer(fun, ub, lb,
                                    verbose=verbose,
                                    options=options,
                                    funargs=funargs,
                                    hessian_update=hessian_update,
                                    resfun=resfun)

        opt_res = fides_opt.minimize(x0)
        n_iter = fides_opt.iteration
        converged = fides_opt.converged
        return opt_res, n_iter, converged

    """

    runFidesJulia = (x0; verbose=verbose, options=options, funargs=funargs, hessian_update=hessian_update, resfun=resfun) -> py"run_fides_python"(x0, fun, ub, lb, verbose, options, funargs, hessian_update, resfun)
    return runFidesJulia
end


# Helper function ensuring the hessian is returned as required by fides
function evalAutoDiffHess(p,
                          evalHess!::Function,
                          hessMat::Matrix{Float64})
    hessMat .= 0
    evalHess!(hessMat, p)
    return hessMat[:, :]
end


# Helper function ensuring the gradient is returned as required by fides
function evalAutoDiffGrad(p,
                          evelGrad!::Function,
                          gradient::Vector{Float64})
    gradient .= 0.0
    evelGrad!(gradient, p)
    return gradient[:]
end


# Helper functions wrapping the output into a Fides acceptable format when hessian is provided
function fidesObjHess(p, evalF::Function, evalGradF::Function, evalHessF::Function)
    fVal = evalF(p)
    fGradVal = evalGradF(p)
    fHessVal = evalHessF(p)
    return (fVal, fGradVal, fHessVal)
end


# Helper functions wrapping the output into a Fides acceptable format when hessian approximation is used
function fidesObjApprox(p, evalF::Function, evalGradF::Function)
    fVal = evalF(p)
    fGradVal = evalGradF(p)
    return (fVal, fGradVal)
end