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

    printstyled(io, "PEtabModel", color=116)
    print(io, " for model ")
    printstyled(io, modelName, color=116)
    print(io, ". ODE-system has ")
    printstyled(io, numberOfODEStates * " states", color=116)
    print(io, " and ")
    printstyled(io, numberOfODEParameters * " parameters.", color=116)
    if !isempty(a.dirJulia)
        @printf(io, "\nGenerated Julia files are at %s", a.dirJulia)
    end
end
function show(io::IO, a::ODESolverOptions)
    # Extract ODE solver as a readable string (without everything between)
    solverStrWrite, optionsStr = getStringSolverOptions(a)
    printstyled(io, "ODESolverOptions", color=116)
    print(io, " with ODE solver ")
    printstyled(io, solverStrWrite, color=116)
    @printf(io, ". Options %s", optionsStr)
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

    printstyled(io, "PEtabODEProblem", color=116)
    print(io, " for ")
    printstyled(io, modelName, color=116)
    @printf(io, ". ODE-states: %d. Parameters to estimate: %d where %d are dynamic.\n---------- Problem settings ----------\nGradient method : ",
            numberOfODEStates, numberOfParametersToEstimate, numberOfDynamicParameters)
    printstyled(io, gradientMethod, color=116)
    if !isnothing(hessianMethod)
        print(io, "\nHessian method : ")
        printstyled(io, hessianMethod, color=116)
    end
    print(io, "\n--------- ODE-solver settings --------")
    printstyled(io, "\nCost ")
    printstyled(io, solverStrWrite, color=116)
    @printf(io, ". Options %s", optionsStr)
    printstyled(io, "\nGradient ")
    printstyled(io, solverGradStrWrite, color=116)
    @printf(io, ". Options %s", optionsGradStr)

    if a.simulationInfo.haspreEquilibrationConditionId == true
        print(io, "\n--------- SS solver settings ---------")
        # Print cost steady state solver
        print(io, "\nCost ")
        printstyled(io, string(a.ssSolverOptions.method), color=116)
        if a.ssSolverOptions.method === :Simulate && a.ssSolverOptions.howCheckSimulationReachedSteadyState === :wrms
            @printf(io, ". Option wrms with (abstol, reltol) = (%.1e, %.1e)", a.ssSolverOptions.abstol, a.ssSolverOptions.reltol)
        elseif a.ssSolverOptions.method === :Simulate && a.ssSolverOptions.howCheckSimulationReachedSteadyState === :Newton
            @printf(io, ". Option small Newton-step with (abstol, reltol) = (%.1e, %.1e)", a.ssSolverOptions.abstol, a.ssSolverOptions.reltol)
        elseif a.ssSolverOptions.method === :Rootfinding
            algStr = string(a.ssSolverOptions.rootfindingAlgorithm)
            iEnd = findfirst(x -> x == '{', algStr)
            algStr = algStr[1:iEnd-1] * "()"
            @printf(io, ". Algorithm %s with (abstol, reltol, maxiters) = (%.1e, %.1e, %.1e)", algStr, a.ssSolverOptions.abstol, a.ssSolverOptions.reltol, a.ssSolverOptions.maxiters)
        end

        # Print gradient steady state solver
        print(io, "\nGradient ")
        printstyled(io, string(a.ssSolverGradientOptions.method), color=116)
        if a.ssSolverGradientOptions.method === :Simulate && a.ssSolverGradientOptions.howCheckSimulationReachedSteadyState === :wrms
            @printf(io, ". Options wrms with (abstol, reltol) = (%.1e, %.1e)", a.ssSolverGradientOptions.abstol, a.ssSolverGradientOptions.reltol)
        elseif a.ssSolverGradientOptions.method === :Simulate && a.ssSolverGradientOptions.howCheckSimulationReachedSteadyState === :Newton
            @printf(io, ". Option small Newton-step with (abstol, reltol) = (%.1e, %.1e)", a.ssSolverGradientOptions.abstol, a.ssSolverGradientOptions.reltol)
        elseif a.ssSolverGradientOptions.method === :Rootfinding
            algStr = string(a.ssSolverGradientOptions.rootfindingAlgorithm)
            iEnd = findfirst(x -> x == '{', algStr)
            algStr = algStr[1:iEnd-1] * "()"
            @printf(io, ". Algorithm %s with (abstol, reltol, maxiters) = (%.1e, %.1e, %.1e)", algStr, a.ssSolverGradientOptions.abstol, a.ssSolverGradientOptions.reltol, a.ssSolverGradientOptions.maxiters)
        end
    end
end
function show(io::IO, a::SteadyStateSolverOptions)

    printstyled(io, "SteadyStateSolverOptions", color=116)
    if a.method === :Simulate
        print(io, " with method ")
        printstyled(io, ":Simulate", color=116)
        print(io, " ODE-model until du = f(u, p, t) ≈ 0.")
        if a.howCheckSimulationReachedSteadyState === :wrms
            @printf(io, "\nSimulation terminated if wrms fulfill;\n√(∑((du ./ (reltol * u .+ abstol)).^2) / length(u)) < 1\n")
        else
            @printf(io, "\nSimulation terminated if Newton-step Δu fulfill;\n√(∑((Δu ./ (reltol * u .+ abstol)).^2) / length(u)) < 1\n")
        end
        if isnothing(a.abstol)
            @printf(io, "with (abstol, reltol) = default values.")
        else
            @printf(io, "with (abstol, reltol) = (%.1e, %.1e)", a.abstol, a.reltol)
        end
    end

    if a.method === :Rootfinding
        print(io, " with method ")
        printstyled(io, ":Rootfinding", color=116)
        print(io, " to solve du = f(u, p, t) ≈ 0.")
        if isnothing(a.rootfindingAlgorithm)
            @printf(io, "\nAlgorithm : NonlinearSolve's heruistic. Options ")
        else
            algStr = string(a.rootfindingAlgorithm)
            iEnd = findfirst(x -> x == '{', algStr)
            algStr = algStr[1:iEnd-1] * "()"
            @printf(io, "\nAlgorithm : %s. Options ", algStr)
        end
        @printf(io, "(abstol, reltol, maxiters) = (%.1e, %.1e, %d)", a.abstol, a.reltol, a.maxiters)
    end
end
function show(io::IO, a::PEtabOptimisationResult) 
    printstyled(io, "PEtabOptimisationResult", color=116)
    print(io, "\n--------- Summary ---------\n")
    @printf(io, "min(f)                = %.2e\n", a.fMin)
    @printf(io, "Parameters esimtated  = %d\n", length(a.x0))
    @printf(io, "Optimiser iterations  = %d\n", a.nIterations)
    @printf(io, "Run time              = %.1es\n", a.runTime)
    @printf(io, "Optimiser algorithm   = %s\n", a.alg)
end
function show(io::IO, a::PEtabMultistartOptimisationResult)

    printstyled(io, "PEtabMultistartOptimisationResult", color=116)
    print(io, "\n--------- Summary ---------\n")
    @printf(io, "min(f)                = %.2e\n", a.fMin)
    @printf(io, "Parameters esimtated  = %d\n", length(a.xMin))
    @printf(io, "Number of multistarts = %d\n", a.nMultistarts)
    @printf(io, "Optimiser algorithm   = %s\n", a.alg)
    if !isnothing(a.dirSave)
        @printf(io, "Results saved at %s\n", a.dirSave)
    end
end
function show(io::IO, a::PEtabParameter)

    printstyled(io, "PEtabParameter", color=116)
    @printf(io, " %s", a.parameter)
    if a.estimate != true
        return nothing
    end
    @printf(io, ". Estimated on %s-scale with bounds [%.1e, %.1e]", a.scale, a.lb, a.ub)
    if isnothing(a.prior)
        return nothing
    end
    @printf(io, " and prior %s", string(a.prior))
    return nothing
end
function show(io::IO, a::PEtabObservable)
    printstyled(io, "PEtabObservable", color=116)
    @printf(io, ": h = %s, noise-formula = %s and ", string(a.obs), string(a.noiseFormula))
    if a.transformation ∈ [:log, :log10]
        @printf(io, "log-normal measurement noise")
    else
        @printf(io, "normal (Gaussian) measurement noise")
    end
end
