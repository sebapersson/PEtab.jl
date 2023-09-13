#=
    Fides wrapper 
=#


function PEtab.callibrateModelMultistart(petabProblem::PEtabODEProblem, 
                                         alg::Fides, 
                                         nMultiStarts::Signed, 
                                         dirSave::Union{Nothing, String};
                                         samplingMethod::T=QuasiMonteCarlo.LatinHypercubeSample(),
                                         options=py"{'maxiter' : 1000}"o,
                                         seed::Union{Nothing, Integer}=nothing, 
                                         saveTrace::Bool=false)::PEtab.PEtabMultistartOptimisationResult where T <: QuasiMonteCarlo.SamplingAlgorithm
    if !isnothing(seed)
        Random.seed!(seed)
    end
    res = PEtab._multistartModelCallibration(petabProblem, alg, nMultiStarts, dirSave, samplingMethod, options, saveTrace)
    return res
end


function PEtab.callibrateModel(petabProblem::PEtabODEProblem, 
                               p0::Vector{Float64},
                               alg::Fides; 
                               saveTrace::Bool=false, 
                               options=py"{'maxiter' : 1000}"o)::PEtab.PEtabOptimisationResult

    _p0 = deepcopy(p0)                               

    if saveTrace == true                         
        @warn "For Fides the x and f trace cannot currently be saved (we are working on it)" maxlog=10     
    end          

    runFides = createFidesProblem(petabProblem, alg, options=options)

    # Create a runnable function taking parameter as input                            
    local nIterations, fMin, xMin, converged, runTime, fTrace, xTrace
    try
        runTime = @elapsed res, nIterations, converged = runFides(p0)
        fMin = res[1]
        xMin = res[2]
        fTrace = Vector{Float64}(undef, 0)
        xTrace = Vector{Vector{Float64}}(undef, 0)
    catch
        nIterations = 0
        fMin = NaN
        xMin = similar(p0) .* NaN
        fTrace = Vector{Float64}(undef, 0)
        xTrace = Vector{Vector{Float64}}(undef, 0)
        converged = :Code_crashed
        runTime = NaN
    end
    algUsed = :Fides

    return PEtabOptimisationResult(algUsed,
                                   xTrace, 
                                   fTrace, 
                                   nIterations, 
                                   fMin, 
                                   _p0,
                                   xMin, 
                                   converged, 
                                   runTime)
end


function createFidesProblem(petabProblem::PEtabODEProblem,
                            fidesSetting::Fides; # In case you want to use any Fides optimizer
                            options=py"{'maxiter' : 1000}"o,
                            funargs=py"None"o,
                            resfun::Bool=false)

    nParam = length(petabProblem.lowerBounds)
    if !isnothing(fidesSetting.hessianApproximation)
        useHessianApproximation = true
    else
        # Put hessian function into acceptable Fides format
        useHessianApproximation = false
        hessian = zeros(Float64, (nParam, nParam))
        computeHessian! = (p) -> evalAutoDiffHess(p, petabProblem.computeHessian!, hessian)
    end

    gradient = zeros(Float64, nParam)
    computeGradient! = (p) -> evalAutoDiffGrad(p, petabProblem.computeGradient!, gradient)

    # Fides objective funciton
    if useHessianApproximation == false
        fidesFunc = (p) -> fidesObjHess(p, petabProblem.computeCost, computeGradient!, computeHessian!)
    else
        fidesFunc = (p) -> fidesObjApprox(p, petabProblem.computeCost, computeGradient!)
    end

    # Set up a runnable executeble for Fides
    fidesObj = setUpFidesClass(fidesFunc, petabProblem.upperBounds, petabProblem.lowerBounds,
                               fidesSetting.verbose,
                               options,
                               funargs,
                               string(fidesSetting.hessianApproximation),
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