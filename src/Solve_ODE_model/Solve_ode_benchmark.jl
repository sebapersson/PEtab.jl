# Solve the ODE models for all experimental condiditions and return the run-time for all solve-calls.
function solveOdeModelAllExperimentalCondBench(prob::ODEProblem, 
                                               changeToExperimentalCondUse!::Function, 
                                               measurementData::DataFrame,
                                               simulationInfo::SimulationInfo,
                                               solver, 
                                               absTol::Float64,
                                               relTol::Float64;
                                               nTSave::Int64=0, 
                                               onlySaveAtTobs::Bool=false,
                                               denseSol::Bool=true)

    absTolSS, relTolSS = simulationInfo.absTolSS, simulationInfo.relTolSS
    local sucess = true
    runTime = 0.0
    # In case the model is first simulated to a steady state 
    if simulationInfo.simulateSS == true
        k = 1
        @inbounds for i in eachindex(simulationInfo.firstExpIds)
            for j in eachindex(simulationInfo.shiftExpIds[i])

                firstExpId = simulationInfo.firstExpIds[i]
                shiftExpId = simulationInfo.shiftExpIds[i][j]

                # Whether or not we only want to save solution at observed time-points 
                if onlySaveAtTobs == true
                    nTSave = 0
                    # Extract t-save point for specific condition ID 
                    tSave = simulationInfo.tVecSave[firstExpId * shiftExpId]
                else
                    tSave=Float64[]
                end

                solveCallPre, solveCallPost = getSolCallSolveOdeSS(absTol, relTol, t_max_ss, solver, tSave, nTSave, denseSol, absTolSS, relTolSS)

                # Change to parameters for the preequilibration simulations 
                changeToExperimentalCondUse!(prob.p, prob.u0, firstExpId)
                u0_pre = deepcopy(prob.u0)
                prob = remake(prob, tspan = (0.0, 1e8), p = prob.p[:], u0 = prob.u0[:])

                # Terminate if a steady state was not reached in preequilibration simulations 
                sol_pre = solveCallPre(prob)
                b = @elapsed sol_pre = solveCallPre(prob)
                simulationInfo.solArray[i] = sol
                runTime += minimum(b).time
                
                if sol_pre.retcode != :Terminated
                    return false, NaN
                end

                # Change to parameters for the post steady state parameters 
                changeToExperimentalCondUse!(prob.p, prob.u0, shiftExpId)
                # Sometimes the experimentaCondition-file changes the initial values for a state 
                # whose value was changed in the preequilibration-simulation. The experimentaCondition
                # value is prioritized by only changing u0 to the steady state value for those states  
                # that were not affected by change to shiftExpId.
                has_not_changed = (prob.u0 .== u0_pre)
                prob.u0[has_not_changed] .= sol_pre.u[end][has_not_changed]
                prob = remake(prob, tspan = (0.0, t_max_ss))
                
                sol = solveCallPost(prob) 
                b = @elapsed sol = solveCallPost(prob) 
                simulationInfo.solArray[i] = sol
                runTime += b
                simulationInfo.solArray[k] = sol

                if simulationInfo.solArray[k].retcode != :Success
                    sucess = false
                    runTime = NaN
                    break 
                end
                k += 1
            end
        end

    # In case the model is not first simulated to a steady state 
    elseif simulationInfo.simulateSS == false

        @inbounds for i in eachindex(simulationInfo.firstExpIds)
            
            firstExpId = simulationInfo.firstExpIds[i]
            # Keep index of which forward solution index i corresponds for calculating cost 
            simulationInfo.conditionIdSol[i] = firstExpId
            # Whether or not we only want to save solution at observed time-points 
            if onlySaveAtTobs == true
                nTSave = 0
                # Extract t-save point for specific condition ID 
                tSave = simulationInfo.tVecSave[firstExpId]
            else
                tSave=Float64[]
            end
            t_max = simulationInfo.tMaxForwardSim[i]

            # Account for different solver algorithms, and if end-time is infinity 
            solveCall = getSolCallSolveOdeNoSS(absTol, relTol, t_max, solver, tSave, nTSave, denseSol, absTolSS, relTolSS)

            # Change parameters to those for the specific experimental condition 
            probUse = getOdeProbSolveOdeNoSS(prob, changeToExperimentalCondUse!, firstExpId, t_max)

            sol = solveCall(probUse)
            b = @elapsed sol = solveCall(probUse) 
            simulationInfo.solArray[i] = sol
            runTime += b

            if !(simulationInfo.solArray[i].retcode == :Success || simulationInfo.solArray[i].retcode == :Terminated)
                sucess = false
                runTime = NaN
                break 
            end
        end
    end

    return success, runTime
end