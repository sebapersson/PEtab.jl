# Solve the ODE models for all experimental condiditions and return the run-time for all solve-calls.
function solveODEModelAllConditionsBenchmark(odeProblem::ODEProblem,
                                             changeExperimentalCondition!::Function,
                                             simulationInfo::SimulationInfo,
                                             solver,
                                             absTol::Float64,
                                             relTol::Float64,
                                             computeTStops::Function;
                                             onlySaveAtObservedTimes::Bool=false,
                                             savePreEqTime::Bool=true)

    bPreEq = 0.0
    bSim = 0.0
    success = true

    # We only compute accuracy for post-equlibrium models
    if simulationInfo.haspreEquilibrationConditionId == true

        # Arrays to store steady state (pre-eq) values.
        preEquilibrationId = unique(simulationInfo.preEquilibrationConditionId)
        uAtSS = Matrix{eltype(odeProblem.p)}(undef, (length(odeProblem.u0), length(preEquilibrationId)))
        u0AtT0 = Matrix{eltype(odeProblem.p)}(undef, (length(odeProblem.u0), length(preEquilibrationId)))

        for i in eachindex(preEquilibrationId)
            _odeSolutions = simulationInfo.odePreEqulibriumSolutions
            bPreEq += @elapsed _odeSolutions[preEquilibrationId[i]] = solveODEPreEqulibrium!((@view uAtSS[:, i]),
                                                                                             (@view u0AtT0[:, i]),
                                                                                             odeProblem,
                                                                                             changeExperimentalCondition!,
                                                                                             preEquilibrationId[i],
                                                                                             absTol,
                                                                                             relTol,
                                                                                             solver,
                                                                                             simulationInfo.callbackSS,
                                                                                             false)

            if _odeSolutions[preEquilibrationId[i]].retcode != ReturnCode.Terminated
                success = false
                break
            end
        end
    end

    if success == false
        return success, Inf
    end

    for i in eachindex(simulationInfo.experimentalConditionId)
        experimentalId = simulationInfo.experimentalConditionId[i]

        if onlySaveAtObservedTimes == true
            nTimePointsSave = 0
            tSave = simulationInfo.timeObserved[experimentalId]
        end

        # Sanity check and process user input.
        tMax = simulationInfo.timeMax[experimentalId]

        # In case we have a simulation with PreEqulibrium
        if simulationInfo.preEquilibrationConditionId[i] != :None
            whichIndex = findfirst(x -> x == simulationInfo.preEquilibrationConditionId[i], preEquilibrationId)
            bSim += @elapsed odeSolution = solveODEPostEqulibrium(odeProblem,
                                                                 (@view uAtSS[:, whichIndex]),
                                                                 (@view u0AtT0[:, whichIndex]),
                                                                 changeExperimentalCondition!,
                                                                 simulationInfo,
                                                                 simulationInfo.simulationConditionId[i],
                                                                 experimentalId,
                                                                 absTol,
                                                                 relTol,
                                                                 tMax,
                                                                 solver,
                                                                 computeTStops,
                                                                 tSave=tSave,
                                                                 denseSolution=false)

        # In case we have an ODE solution without Pre-equlibrium
        else
            bSim += @elapsed odeSolution = solveODENoPreEqulibrium!(odeProblem,
                                                                    changeExperimentalCondition!,
                                                                    simulationInfo,
                                                                    simulationInfo.simulationConditionId[i],
                                                                    absTol,
                                                                    relTol,
                                                                    solver,
                                                                    tMax,
                                                                    computeTStops,
                                                                    tSave=tSave,
                                                                    denseSolution=false)

        end

        if !(odeSolution.retcode == ReturnCode.Success || odeSolution.retcode == ReturnCode.Terminated)
            success = false
            break
        end
    end

    if success == false
        return success, Inf
    end
    if savePreEqTime == true
        return success, bSim + bPreEq
    else
        return success, bSim
    end
end
