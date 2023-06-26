function PEtab.callibrateModel(petabProblem::PEtabODEProblem,
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

    startGuesses = generateStartGuesses(petabProblem, samplingMethod, nOptimisationStarts)

    # Randomly generate startguesses from a Uniform distribution, will add something like a cube later
    # (downstream package)
    for i in 1:nOptimisationStarts
        if isnothing(startGuesses)
            objValues[i] = petabProblem.computeCost(Float64[])
            continue
        end

        p0 = startGuesses[:, i]
        cost0 = petabProblem.computeCost(p0)
        if isinf(cost0)
            objValues[i] = Inf
            parameterValues[i, :] .= p0
            continue
        end

        res = runOptim(p0)
        objValues[i] = res.minimum
        parameterValues[i, :] .= res.minimizer
    end

    return objValues, parameterValues
end
function PEtab.callibrateModel(petabProblem::PEtabODEProblem,
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

    startGuesses = generateStartGuesses(petabProblem, samplingMethod, nOptimisationStarts)

    # Randomly generate startguesses from a Uniform distribution, will add something like a cube later
    # (downstream package)
    for i in 1:nOptimisationStarts
        if isnothing(startGuesses)
            objValues[i] = petabProblem.computeCost(Float64[])
            continue
        end

        p0 = startGuesses[:, i]
        cost0 = petabProblem.computeCost(p0)
        if isinf(cost0)
            objValues[i] = Inf
            parameterValues[i, :] .= p0
            continue
        end

        res, niter, converged = runFides(p0)
        objValues[i] = res[1]
        parameterValues[i, :] .= res[2]
    end

    return objValues, parameterValues
end


function generateStartGuesses(petabProblem::PEtabODEProblem,
                              samplingMethod::T,
                              nOptimisationStarts::Int) where T <: QuasiMonteCarlo.SamplingAlgorithm

    # Nothing prevents the user from sending in a parameter vector with zero parameters
    if length(petabProblem.lowerBounds) == 0
        return nothing
    end

    # Return a random number sampled from uniform distribution
    if nOptimisationStarts == 1
        return [rand() * (petabProblem.upperBounds[i] - petabProblem.lowerBounds[i]) + petabProblem.lowerBounds[i] for i in eachindex(petabProblem.lowerBounds)]
    end

    return QuasiMonteCarlo.sample(nOptimisationStarts, petabProblem.lowerBounds, petabProblem.upperBounds, samplingMethod)
end