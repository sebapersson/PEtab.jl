#=
    Functions for better printing of relevant PEtab-structs which are
    exported to the user.
=#
import Base.show


# Helper function for printing ODE-solver options
function getStringSolverOptions(a::ODESolverOptions)
    solverStr = string(a.solver)
    iEnd = findfirst(x -> x == '{', solverStr)
    if isnothing(iEnd)
        iEnd = findfirst(x -> x == '(', solverStr)
    end
    solverStrWrite = solverStr[1:iEnd-1] * "()"
    optionsStr = @sprintf("(abstol, reltol, maxiters) = (%.1e, %.1e, %.1e)", a.abstol, a.reltol, a.maxiters)
    return solverStrWrite, optionsStr
end


function show(io::IO, a::PEtabModel)

    modelName = @sprintf("%s", a.modelName)
    numberOfODEStates = @sprintf("%d", length(a.stateNames))
    numberOfODEParameters = @sprintf("%d", length(a.parameterNames))

    printstyled("PEtabModel", color=116)
    print(" for model ")
    printstyled(modelName, color=116)
    print(". ODE-system has ")
    printstyled(numberOfODEStates * " states", color=116)
    print(" and ")
    printstyled(numberOfODEParameters * " parameters.", color=116)
    if !isempty(a.dirJulia)
        @printf("\nGenerated Julia files are at %s", a.dirJulia)
    end
end
function show(io::IO, a::ODESolverOptions)
    # Extract ODE solver as a readable string (without everything between)
    solverStrWrite, optionsStr = getStringSolverOptions(a)
    printstyled("ODESolverOptions", color=116)
    print(" with ODE solver ")
    printstyled(solverStrWrite, color=116)
    @printf(". Options %s", optionsStr)
end
function show(io::IO, a::PEtabODEProblem)

    modelName = a.petabModel.modelName
    numberOfODEStates = length(a.petabModel.stateNames)
    numberOfParametersToEstimate = length(a.θ_estNames)
    θ_indices = a.θ_indices
    numberOfDynamicParameters = length(intersect(θ_indices.θ_dynamicNames, a.θ_estNames))

    solverStrWrite, optionsStr = getStringSolverOptions(a.odeSolverOptions)
    solverGradStrWrite, optionsGradStr = getStringSolverOptions(a.odeSolverGradientOptions)

    gradientMethod = string(a.gradientMethod)
    hessianMethod = string(a.hessianMethod)

    printstyled("PEtabODEProblem", color=116)
    print(" for ")
    printstyled(modelName, color=116)
    @printf(". ODE-states: %d. Parameters to estimate: %d where %d are dynamic.\n---------- Problem settings ----------\nGradient method : ",
            numberOfODEStates, numberOfParametersToEstimate, numberOfDynamicParameters)
    printstyled(gradientMethod, color=116)
    if !isnothing(hessianMethod)
        print("\nHessian method : ")
        printstyled(hessianMethod, color=116)
    end
    print("\n--------- ODE-solver settings --------")
    printstyled("\nCost ")
    printstyled(solverStrWrite, color=116)
    @printf(". Options %s", optionsStr)
    printstyled("\nGradient ")
    printstyled(solverGradStrWrite, color=116)
    @printf(". Options %s", optionsGradStr)

    if a.simulationInfo.haspreEquilibrationConditionId == true
        print("\n--------- SS solver settings ---------")
        # Print cost steady state solver
        print("\nCost ")
        printstyled(string(a.ssSolverOptions.method), color=116)
        if a.ssSolverOptions.method === :Simulate && a.ssSolverOptions.howCheckSimulationReachedSteadyState === :wrms
            @printf(". Option wrms with (abstol, reltol) = (%.1e, %.1e)", a.ssSolverOptions.abstol, a.ssSolverOptions.reltol)
        elseif a.ssSolverOptions.method === :Simulate && a.ssSolverOptions.howCheckSimulationReachedSteadyState === :Newton
            @printf(". Option small Newton-step with (abstol, reltol) = (%.1e, %.1e)", a.ssSolverOptions.abstol, a.ssSolverOptions.reltol)
        elseif a.ssSolverOptions.method === :Rootfinding
            algStr = string(a.ssSolverOptions.rootfindingAlgorithm)
            iEnd = findfirst(x -> x == '{', algStr)
            algStr = algStr[1:iEnd-1] * "()"
            @printf(". Algorithm %s with (abstol, reltol, maxiters) = (%.1e, %.1e, %.1e)", algStr, a.ssSolverOptions.abstol, a.ssSolverOptions.reltol, a.ssSolverOptions.maxiters)
        end

        # Print gradient steady state solver
        print("\nGradient ")
        printstyled(string(a.ssSolverGradientOptions.method), color=116)
        if a.ssSolverGradientOptions.method === :Simulate && a.ssSolverGradientOptions.howCheckSimulationReachedSteadyState === :wrms
            @printf(". Options wrms with (abstol, reltol) = (%.1e, %.1e)", a.ssSolverGradientOptions.abstol, a.ssSolverGradientOptions.reltol)
        elseif a.ssSolverGradientOptions.method === :Simulate && a.ssSolverGradientOptions.howCheckSimulationReachedSteadyState === :Newton
            @printf(". Option small Newton-step with (abstol, reltol) = (%.1e, %.1e)", a.ssSolverGradientOptions.abstol, a.ssSolverGradientOptions.reltol)
        elseif a.ssSolverGradientOptions.method === :Rootfinding
            algStr = string(a.ssSolverGradientOptions.rootfindingAlgorithm)
            iEnd = findfirst(x -> x == '{', algStr)
            algStr = algStr[1:iEnd-1] * "()"
            @printf(". Algorithm %s with (abstol, reltol, maxiters) = (%.1e, %.1e, %.1e)", algStr, a.ssSolverGradientOptions.abstol, a.ssSolverGradientOptions.reltol, a.ssSolverGradientOptions.maxiters)
        end
    end
end
function show(io::IO, a::SteadyStateSolverOptions)

    printstyled("SteadyStateSolverOptions", color=116)
    if a.method === :Simulate
        print(" with method ")
        printstyled(":Simulate", color=116)
        print(" ODE-model until du = f(u, p, t) ≈ 0.")
        if a.howCheckSimulationReachedSteadyState === :wrms
            @printf("\nSimulation terminated if wrms fulfill;\n√(∑((du ./ (reltol * u .+ abstol)).^2) / length(u)) < 1\n")
        else
            @printf("\nSimulation terminated if Newton-step Δu fulfill;\n√(∑((Δu ./ (reltol * u .+ abstol)).^2) / length(u)) < 1\n")
        end
        if isnothing(a.abstol)
            @printf("with (abstol, reltol) = default values.")
        else
            @printf("with (abstol, reltol) = (%.1e, %.1e)", a.abstol, a.reltol)
        end
    end

    if a.method === :Rootfinding
        print(" with method ")
        printstyled(":Rootfinding", color=116)
        print(" to solve du = f(u, p, t) ≈ 0.")
        if isnothing(a.rootfindingAlgorithm)
            @printf("\nAlgorithm : NonlinearSolve's heruistic. Options ")
        else
            algStr = string(a.rootfindingAlgorithm)
            iEnd = findfirst(x -> x == '{', algStr)
            algStr = algStr[1:iEnd-1] * "()"
            @printf("\nAlgorithm : %s. Options ", algStr)
        end
        @printf("(abstol, reltol, maxiters) = (%.1e, %.1e, %d)", a.abstol, a.reltol, a.maxiters)
    end
end
function show(io::IO, a::PEtabOptimisationResult) 
    printstyled("PEtabOptimisationResult", color=116)
    print("\n--------- Summary ---------\n")
    @printf("min(f)                = %.2e\n", a.fBest)
    @printf("Parameters esimtated  = %d\n", length(a.x0))
    @printf("Optimiser iterations  = %d\n", a.nIterations)
    @printf("Run time              = %.1es\n", a.runTime)
    @printf("Optimiser algorithm   = %s\n", a.alg)
end
function show(io::IO, a::PEtabMultistartOptimisationResult)

    printstyled("PEtabMultistartOptimisationResult", color=116)
    print("\n--------- Summary ---------\n")
    @printf("min(f)                = %.2e\n", a.fMin)
    @printf("Parameters esimtated  = %d\n", length(a.xMin))
    @printf("Number of multistarts = %d\n", a.nMultistarts)
    @printf("Optimiser algorithm   = %s\n", a.alg)
    if !isnothing(a.dirSave)
        @printf("Results saved at %s\n", a.dirSave)
    end
end
