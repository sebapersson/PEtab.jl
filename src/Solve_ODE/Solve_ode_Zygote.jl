# Solve the ODE system for one experimental conditions in a Zygote compatible manner. Not well maintained and lacks
# full support because Zygote code is currently the slowest (by far)
function solveOdeModelAtExperimentalCondZygote(odeProblem::ODEProblem,
                                               experimentalId::Symbol,
                                               dynParamEst,
                                               t_max,
                                               changeToExperimentalCondUsePre::Function,
                                               measurementInfo::MeasurementsInfo,
                                               simulationInfo::SimulationInfo,
                                               solver::Union{SciMLAlgorithm, Vector{Symbol}},
                                               absTol::Float64,
                                               relTol::Float64,
                                               sensealg,
                                               calcTStops::Function)

    changeToExperimentalCondUse = (pVec, u0Vec, expID) -> changeToExperimentalCondUsePre(pVec, u0Vec, expID, dynParamEst)

    # For storing ODE solution (required for split gradient computations)
    whichCondID = findfirst(x -> x == experimentalId, simulationInfo.experimentalConditionId)

    # In case the model is first simulated to a steady state
    local success = true
    if simulationInfo.haspreEquilibrationConditionId == true

        firstExpId = simulationInfo.preEquilibrationConditionId[whichCondID]
        shiftExpId = simulationInfo.simulationConditionId[whichCondID]
        tSave = simulationInfo.timeObserved[experimentalId]

        u0Pre = odeProblem.u0[:]
        pUsePre, u0UsePre = changeToExperimentalCondUse(odeProblem.p, odeProblem.u0, firstExpId)
        probUsePre = remake(odeProblem, tspan=(0.0, 1e8), u0 = convert.(eltype(dynParamEst), u0UsePre), p = convert.(eltype(dynParamEst), pUsePre))
        ssProb = SteadyStateProblem(probUsePre)
        solSS = solve(ssProb, DynamicSS(solver, abstol=simulationInfo.absTolSS, reltol=simulationInfo.relTolSS), abstol=absTol, reltol=relTol)

        # Terminate if a steady state was not reached in preequilibration simulations
        if solSS.retcode != :Success
            return sol_pre, false
        end

        # Change to parameters for the post steady state parameters
        pUsePost, u0UsePostTmp = changeToExperimentalCondUse(odeProblem.p, odeProblem.u0, shiftExpId)

        # Given the standard the experimentaCondition-file can change the initial values for a state
        # whose value was changed in the preequilibration-simulation. The experimentalCondition
        # value is prioritized by only changing u0 to the steady state value for those states
        # that were not affected by change to shiftExpId.
        hasNotChanged = (u0UsePostTmp .== u0Pre)
        u0UsePost = [hasNotChanged[i] == true ? solSS[i] : u0UsePostTmp[i] for i in eachindex(u0UsePostTmp)]
        probUsePost = remake(odeProblem, tspan=(0.0, t_max), u0 = convert.(eltype(dynParamEst), u0UsePost), p = convert.(eltype(dynParamEst), pUsePost))

        # Different funcion calls to solve are required if a solver or a Alg-hint are provided.
        # The preequilibration simulations are terminated upon a steady state using the TerminateSteadyState callback.
        tStops = calcTStops(probUsePost.u0, probUsePost.p)
        solveCallPost = (odeProblem) -> solve(odeProblem,
                                        solver,
                                        abstol=absTol,
                                        reltol=relTol,
                                        saveat=tSave,
                                        sensealg=sensealg,
                                        callback=simulationInfo.callbacks[experimentalId],
                                        tstops=tStops)


        sol = solveCallPost(probUsePost)

        if sol.retcode != :Success
            sucess = false
        end

    # In case the model is not first simulated to a steady state
    elseif simulationInfo.haspreEquilibrationConditionId == false

        firstExpId = simulationInfo.simulationConditionId[whichCondID]
        tSave = simulationInfo.timeObserved[experimentalId]
        t_max_use = isinf(t_max) ? 1e8 : t_max

        pUse, u0Use = changeToExperimentalCondUse(odeProblem.p, odeProblem.u0, firstExpId)
        probUse = remake(odeProblem, tspan=(0.0, t_max_use), u0 = convert.(eltype(dynParamEst), u0Use), p = convert.(eltype(dynParamEst), pUse))

        # Different funcion calls to solve are required if a solver or a Alg-hint are provided.
        # If t_max = inf the model is simulated to steady state using the TerminateSteadyState callback.
        tStops = calcTStops(probUse.u0, probUse.p)
        if !(typeof(solver) <: Vector{Symbol}) && isinf(t_max)
            solveCall = (probArg) -> solve(probArg,
                                           solver,
                                           abstol=absTol,
                                           reltol=relTol,
                                           save_on=false,
                                           save_end=true,
                                           dense=dense,
                                           callback=TerminateSteadyState(absTolSS, relTolSS))

        elseif !(typeof(solver) <: Vector{Symbol}) && !isinf(t_max)
            solveCall = (probArg) -> solve(probArg,
                                           solver,
                                           p = pUse,
                                           u0 = u0Use,
                                           abstol=absTol,
                                           reltol=relTol,
                                           saveat=tSave,
                                           sensealg=sensealg,
                                           callback=simulationInfo.callbacks[experimentalId],
                                           tstops=tStops)
        else
            println("Error : Solver option does not exist")
        end

        sol = solveCall(probUse)

        Zygote.@ignore simulationInfo.odeSolutions[experimentalId] = sol

        if typeof(sol) <: ODESolution && !(sol.retcode == :Success || sol.retcode == :Terminated)
            sucess = false
        end
    end

    return sol, success
end
