"""
    createNLoptProb(petabProblem::PEtabODEProblem,
                    NLoptAlg::Symbol; 
                    maxeval=5000)
    
    For a PeTab model optimization struct (petabProblem) create a NLopt optimization
    struct where the optmization is performed using NLoptAlg. In principle any 
    NLopt-alg is supported.
"""
function createNLoptProb(petabProblem::PEtabODEProblem, NLoptAlg::Symbol; maxeval::Integer=2000, verbose::Bool=false)

    algSupport = [:LD_TNEWTON_PRECOND_RESTART, :LD_LBFGS]
    if !(NLoptAlg in algSupport)
        println("Alg = $NLoptAlg is not currently supported")
        println("Supported methods are :LD_TNEWTON_PRECOND_RESTART and :LD_LBFGS")
        return 
    end

    NLoptObj = NLopt.Opt(NLoptAlg, petabProblem.nParametersToEstimate)
    NLoptObj.lower_bounds = petabProblem.lowerBounds .- 1e-9
    NLoptObj.upper_bounds = petabProblem.upperBounds .+ 1e-9
    NLoptObj.ftol_rel = 1e-3
    NLoptObj.xtol_rel = 1e-3
    NLoptObj.maxeval = maxeval # Prevent never ending optmization
    NLoptObj.min_objective = (x, grad) -> NLoptF(x, grad, petabProblem.computeCost, petabProblem.computeGradientAutoDiff, verbose=verbose)

    return NLoptObj
end


# Function on the form required by NLopt
function NLoptF(x::T1, grad::T1, evalF::Function, evalGradF::Function; verbose::Bool=false) where T1 <: Vector{<:AbstractFloat}

    if length(grad) > 0
        evalGradF(grad, x)
    end
    cost = evalF(x)

    if verbose == true
        @printf("Cost = %.5e\n", cost)
    end

    return cost 
end