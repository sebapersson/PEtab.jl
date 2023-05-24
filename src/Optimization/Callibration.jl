"""
    calibrateModel(petabProblem::PEtabODEProblem, 
                   optimizer;
                   <keyword arguments>)

Perform multi-start local optimization for a given PEtabODEProblem and return (fmin, minimizer) for all runs.

# Arguments
- `petabProblem::PEtabODEProblem`: The PEtabODEProblem to be calibrated.
- `optimizer`: The optimizer algorithm to be used. Currently, we support three different algorithms:
    1. `Fides()`: The Newton trust-region Fides optimizer from Python. Please refer to the documentation for setup 
        examples. This optimizer performs well when computing the full Hessian is not possible, and the Gauss-Newton Hessian approximation can be used.
    2. `IPNewton()`: The interior-point Newton method from Optim.jl. This optimizer performs well when it is 
        computationally feasible to compute the full Hessian.
    3. `LBFGS()` or `BFGS()` from Optim.jl: These optimizers are suitable when the computation of the Gauss-Newton 
        Hessian approximation is too expensive, such as when adjoint sensitivity analysis is required for the gradient.
- `nOptimisationStarts::Int`: Number of multi-starts to be performed. Defaults to 100.
- `samplingMethod`: Method for generating start guesses. Any method from QuasiMonteCarlo.jl is supported, with LatinHypercube as the default.
- `options`: Optimization options. For Optim.jl optimizers, it accepts an `Optim.Options` struct. For Fides, please refer to the Fides documentation and the PEtab.jl documentation for information on setting options.
"""
function callibrateModel(petabProblem::PEtabODEProblem,
                         optimizer::Union{Optim.LBFGS, Optim.BFGS, Optim.IPNewton};
                         nOptimisationStarts=100, 
                         seed=123, 
                         options=Optim.Options(iterations = 1000, 
                                               show_trace = false, 
                                               allow_f_increases=true, 
                                               successive_f_tol = 3, 
                                               f_tol=1e-8, 
                                               g_tol=1e-6, 
                                               x_tol=0.0), 
                         samplingMethod::T=QuasiMonteCarlo.LatinHypercubeSample()) where T <: QuasiMonteCarlo.SamplingAlgorithm

    Random.seed!(seed)
    nParameters = length(petabProblem.lowerBounds)
    objValues = zeros(Float64, nOptimisationStarts)
    parameterValues = zeros(Float64, nOptimisationStarts, nParameters)
    runOptim = createOptimProblem(petabProblem, optimizer, options=options)

    startGuesses = QuasiMonteCarlo.sample(nOptimisationStarts, petabProblem.lowerBounds, petabProblem.upperBounds, samplingMethod)

    # Randomly generate startguesses from a Uniform distribution, will add something like a cube later 
    # (downstream package)
    for i in 1:nOptimisationStarts
        p0 = startGuesses[:, i]
        if !isempty(p0)

            cost0 = petabProblem.computeCost(p0)
            if isinf(cost0)
                objValues[i] = Inf
                parameterValues[i, :] .= p0
                continue
            end

            res = runOptim(p0)
            objValues[i] = res.minimum 
            parameterValues[i, :] .= res.minimizer
        else
            objValues[i] = petabProblem.computeCost(Float64[])
        end
    end

    return objValues, parameterValues
end
function callibrateModel(petabProblem::PEtabODEProblem, 
                         optimizer::Fides=Fides(verbose=false);
                         nOptimisationStarts=100, 
                         seed=123, 
                         options=py"{'maxiter' : 1000}"o,
                         samplingMethod::T=QuasiMonteCarlo.LatinHypercubeSample()) where T <: QuasiMonteCarlo.SamplingAlgorithm

    Random.seed!(seed)
    nParameters = length(petabProblem.lowerBounds)
    objValues = zeros(Float64, nOptimisationStarts)
    parameterValues = zeros(Float64, nOptimisationStarts, nParameters)
    runFides = createFidesProblem(petabProblem, optimizer, options=options)

    startGuesses = QuasiMonteCarlo.sample(nOptimisationStarts, petabProblem.lowerBounds, petabProblem.upperBounds, samplingMethod)

    # Randomly generate startguesses from a Uniform distribution, will add something like a cube later 
    # (downstream package)
    for i in 1:nOptimisationStarts
        p0 = startGuesses[:, i]
        if !isempty(p0)

            cost0 = petabProblem.computeCost(p0)
            if isinf(cost0)
                objValues[i] = Inf
                parameterValues[i, :] .= p0
                continue
            end

            res, niter, converged = runFides(p0)
            objValues[i] = res[1]
            parameterValues[i, :] .= res[2]
        else
            objValues[i] = petabProblem.computeCost(Float64[])
        end
    end

    return objValues, parameterValues
end