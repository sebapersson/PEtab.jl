function callibrateModel(petabProblem::PEtabODEProblem,
                         optimAlg::Union{Optim.LBFGS, Optim.BFGS, Optim.IPNewton};
                         nStartGuesses=100, 
                         seed=123, 
                         options=Optim.Options(iterations = 1000, 
                                               show_trace = false, 
                                               allow_f_increases=true, 
                                               successive_f_tol = 3, 
                                               f_tol=1e-8, 
                                               g_tol=1e-6, 
                                               x_tol=0.0))

    Random.seed!(seed)
    nParameters = length(petabProblem.lowerBounds)
    objValues = zeros(Float64, nStartGuesses)
    parameterValues = zeros(Float64, nStartGuesses, nParameters)
    runOptim = createOptimProblem(petabProblem, optimAlg, options=options)

    # Randomly generate startguesses from a Uniform distribution, will add something like a cube later 
    # (downstream package)
    for i in 1:nStartGuesses
        p0 = [rand() * (petabProblem.upperBounds[i] - petabProblem.lowerBounds[i]) + petabProblem.lowerBounds[i] for i in eachindex(petabProblem.lowerBounds)]
        if !isempty(p0)
            res = runOptim(p0)
            objValues[i] = res.minimum 
            parameterValues[i, :] .= res.minimizer
        else
            objValues[i] = petabProblem.computeCost(Float64[])
        end
    end

    whichMin = argmin(objValues)
    return objValues[whichMin], parameterValues[whichMin, :]
end
function callibrateModel(petabProblem::PEtabODEProblem, 
                         optimAlg::Fides=Fides(verbose=false);
                         nStartGuesses=100, 
                         seed=123, 
                         options=py"{'maxiter' : 1000}"o)

    Random.seed!(seed)
    nParameters = length(petabProblem.lowerBounds)
    objValues = zeros(Float64, nStartGuesses)
    parameterValues = zeros(Float64, nStartGuesses, nParameters)
    runFides = createFidesProblem(petabProblem, optimAlg, options=options)

    # Randomly generate startguesses from a Uniform distribution, will add something like a cube later 
    # (downstream package)
    for i in 1:nStartGuesses
        p0 = [rand() * (petabProblem.upperBounds[i] - petabProblem.lowerBounds[i]) + petabProblem.lowerBounds[i] for i in eachindex(petabProblem.lowerBounds)]
        if !isempty(p0)
            res, niter, converged = runFides(p0)
            objValues[i] = res[1]
            parameterValues[i, :] .= res[2]
        else
            objValues[i] = petabProblem.computeCost(Float64[])
        end
    end

    whichMin = argmin(objValues)
    return objValues[whichMin], parameterValues[whichMin, :]
end