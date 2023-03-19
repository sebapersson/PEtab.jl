function setUpFides(petabProblem::PEtabODEProblem; 
                    fidesHessianApproximation=py"None"o, # In case you want to use any Fides optimizer
                    verbose::Integer=1,
                    options=py"{'maxiter' : 1000}"o,
                    funargs=py"None"o,
                    resfun::Bool=false)

    nParam = length(petabProblem.lowerBounds)
    if fidesHessianApproximation !=  py"None"o
        useHessianApproximation = true
    else
        # Put hessian function into acceptable Fides format 
        useHessianApproximation = false
        hessian = zeros(Float64, (nParam, nParam))
        computeHessian! = (p) -> evalAutoDiffHess(p, petabProblem.computeHessian!, hessian)        
    end

    println("useHessianApproximation = ", useHessianApproximation)

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
                               verbose, 
                               options, 
                               funargs, 
                               fidesHessianApproximation, 
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
