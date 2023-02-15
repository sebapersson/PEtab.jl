using LatinHypercubeSampling, CSV, DataFrames, Random
using Distributions


"""
    createCube(pathSave::String, petabProblem::PEtabODEProblem, nSamples::Integer; seed=123, verbose::Bool=false)

    For a PeTab-optimization struct create a Lathin-hypercube of nSamples parameter vectors which 
    is stored at pathSave (default in petabModel.dirModel). 
"""
function createCube(pathSave::String, petabProblem::PEtabODEProblem, nSamples::Integer; seed=123, verbose::Bool=false)
    _createCube(pathSave, petabProblem, nSamples, seed=seed, verbose=verbose)
end
"""
    createCube(petabProblem::PEtabODEProblem, nSamples::Integer; seed=123, verbose::Bool=false)

    For a PeTab-optimization struct create a Lathin-hypercube of nSamples parameter vectors which 
    is stored at petabProblem.pathCube (which by default is in petabModel.dirModel). 
"""
function createCube(petabProblem::PEtabODEProblem, nSamples::Integer; seed=123, verbose::Bool=false)
    _createCube(petabProblem.pathCube, petabProblem, nSamples, seed=seed, verbose=verbose)
end


function _createCube(pathSave::String, petabProblem::PEtabODEProblem, nSamples::Integer; seed=123, verbose::Bool=false)

    Random.seed!(seed)

    if isfile(pathSave)
        println("Cube already exists will not build")
        return 
    end
    println("Cube does not exist - will build cube")

    lowerBounds, upperBounds = petabProblem.lowerBounds, petabProblem.upperBounds
    nDims = length(lowerBounds)
    
    paramSave = zeros(Float64, (nSamples, nDims))
    useCube = true
    k, maxIter = 1, 1
    # k = number of sucessfully found parameters 
    while k - 1 != nSamples && maxIter < nSamples*10

        # Generate a cube with iMax samples. This part regenerates cube 
        # in case the cost could could not be evaluated for some previous 
        # cuble value 
        iMax = nSamples - (k - 1)
        # If less than 15 parameter-vectors are left to generate use random sampling 
        # as new cube can be biased towards bad parameter regions.
        if iMax > 15
            plan = LHCoptim(iMax, nDims, 10)[1]
            bounds = [(lowerBounds[i], upperBounds[i]) for i in eachindex(lowerBounds)] 
            scaledPlan = Matrix(scaleLHC(plan, bounds))
            useCube = true
        else
            iMax = 1
            useCube = false
        end
        
        for i in 1:iMax
            if useCube == true
                paramI = scaledPlan[i, :]
            else
                paramI = [rand(Uniform(lowerBounds[i], upperBounds[i])) for i in eachindex(lowerBounds)]
            end

            local cost
            try 
                cost = petabProblem.computeCost(paramI)
            catch
                cost = Inf
            end
            
            if !(isinf(cost) || isnan(cost))
                paramSave[k, :] .= paramI
                k += 1
            end

            if k % 50 == 0 && verbose == true
                println("Have found $k start guesses")
            end

            if k - 1 == nSamples
                break
            end

            maxIter += 1
        end
    end

    # In case maximum number of iterations were exceeded trying to find parameters 
    if k - 1 != nSamples
        println("Error : Did not find $nSamples start guesses for the estimation")
    end

    dataSave = DataFrame(paramSave, :auto)
    rename!(dataSave, petabProblem.θ_estNames)
    CSV.write(pathSave, dataSave)

    println("Cube saved at $pathSave")
end


