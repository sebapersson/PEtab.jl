#=
    Functionallity for solving a PEtab ODE-model across all experimental conditions. Code is compatible with ForwardDiff,
    and there is functionallity for computing the sensitivity matrix.
=#


function solveODEAllExperimentalConditions!(odeSolutions::Dict{Symbol, Union{Nothing, ODESolution}},
                                            _odeProblem::ODEProblem,
                                            petabModel::PEtabModel,
                                            θ_dynamic::AbstractVector,
                                            petabODESolverCache::PEtabODESolverCache,
                                            simulationInfo::SimulationInfo,
                                            θ_indices::ParameterIndices,
                                            odeSolverOptions::ODESolverOptions; 
                                            expIDSolve::Vector{Symbol} = [:all],
                                            nTimePointsSave::Int64=0,
                                            onlySaveAtObservedTimes::Bool=false,
                                            denseSolution::Bool=true, 
                                            trackCallback::Bool=false, 
                                            computeForwardSensitivites::Bool=false)::Bool # Required for adjoint sensitivity analysis

    changeExperimentalCondition! = (pODEProblem, u0, conditionId) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices, computeForwardSensitivites=computeForwardSensitivites)

    local sucess::Bool = true
    # In case the model is first simulated to a steady state
    if simulationInfo.haspreEquilibrationConditionId == true

        # Extract all unique Pre-equlibrium conditions. If the code is run in parallell
        # (expIDSolve != [["all]]) the number of preEq cond. might be smaller than the
        # total number of preEq cond.
        if expIDSolve[1] == :all
            preEquilibrationId = unique(simulationInfo.preEquilibrationConditionId)
        else
            whichId = findall(x -> x ∈ simulationInfo.experimentalConditionId, expIDSolve)
            preEquilibrationId = unique(simulationInfo.preEquilibrationConditionId[whichId])
        end

        # Arrays to store steady state (pre-eq) values.
        uAtSS = Matrix{eltype(θ_dynamic)}(undef, (length(_odeProblem.u0), length(preEquilibrationId)))
        u0AtT0 = Matrix{eltype(θ_dynamic)}(undef, (length(_odeProblem.u0), length(preEquilibrationId)))

        for i in eachindex(preEquilibrationId)
            
            odeProblem = setODEProblemParameters(_odeProblem, petabODESolverCache, preEquilibrationId[i])
            # Sometimes due to strongly ill-conditioned Jacobian the linear-solve runs
            # into a domain error or bounds error. This is treated as integration error.
            try
                _odeSolutions = simulationInfo.odePreEqulibriumSolutions
                _odeSolutions[preEquilibrationId[i]] = solveODEPreEqulibrium!((@view uAtSS[:, i]),
                                                                              (@view u0AtT0[:, i]),
                                                                               odeProblem,
                                                                               changeExperimentalCondition!,
                                                                               preEquilibrationId[i],
                                                                               odeSolverOptions,
                                                                               simulationInfo.callbackSS,
                                                                               petabModel.convertTspan)
                if _odeSolutions[preEquilibrationId[i]].retcode != ReturnCode.Terminated
                    return false
                end
            catch e
                checkError(e)
                return false
            end
        end
    end

    @inbounds for i in eachindex(simulationInfo.experimentalConditionId)
        experimentalId = simulationInfo.experimentalConditionId[i]

        if expIDSolve[1] != :all && experimentalId ∉ expIDSolve
            continue
        end

        # Check if we want to only save the ODE at specific time-points
        if onlySaveAtObservedTimes == true
            nTimePointsSave = 0
            tSave = simulationInfo.timeObserved[experimentalId]
        else
            tSave=Float64[]
        end

        # Sanity check and process user input.
        tMax = simulationInfo.timeMax[experimentalId]
        if length(tSave) != 0 && nTimePointsSave != 0
            println("Error : Can only provide tSave (vector to save at) or nTimePointsSave as saveat argument to solvers")
        elseif nTimePointsSave != 0
            tSave = collect(LinRange(0.0, tMax, nTimePointsSave))
        end
        if !(isempty(tSave) && nTimePointsSave == 0)
            denseSolution = false
        end

        odeProblem = setODEProblemParameters(_odeProblem, petabODESolverCache, experimentalId)

        # In case we have a simulation with PreEqulibrium
        if simulationInfo.preEquilibrationConditionId[i] != :None
            whichIndex = findfirst(x -> x == simulationInfo.preEquilibrationConditionId[i], preEquilibrationId)
            # See comment above on domain error
            try
                odeSolutions[experimentalId] = solveODEPostEqulibrium(odeProblem,
                                                                      (@view uAtSS[:, whichIndex]),
                                                                      (@view u0AtT0[:, whichIndex]),
                                                                      changeExperimentalCondition!,
                                                                      simulationInfo,
                                                                      simulationInfo.simulationConditionId[i],
                                                                      experimentalId,
                                                                      tMax,
                                                                      odeSolverOptions,
                                                                      petabModel.computeTStops,
                                                                      tSave=tSave,
                                                                      denseSolution=denseSolution,
                                                                      trackCallback=trackCallback,
                                                                      convertTspan=petabModel.convertTspan)

                if odeSolutions[experimentalId].retcode != ReturnCode.Success
                    sucess = false
                end
            catch e
                checkError(e)
                return false
            end
            if sucess == false
                return false
            end

        # In case we have an ODE solution without Pre-equlibrium
        else
            try
                odeSolutions[experimentalId] = solveODENoPreEqulibrium!(odeProblem,
                                                                        changeExperimentalCondition!,
                                                                        simulationInfo,
                                                                        simulationInfo.simulationConditionId[i],
                                                                        odeSolverOptions,
                                                                        tMax,
                                                                        petabModel.computeTStops,
                                                                        tSave=tSave,
                                                                        denseSolution=denseSolution,
                                                                        trackCallback=trackCallback,
                                                                        convertTspan=petabModel.convertTspan)
                retcode = odeSolutions[experimentalId].retcode
                if !(retcode == ReturnCode.Success || retcode == ReturnCode.Terminated)
                    sucess = false
                end
            catch e
                checkError(e)
                return false
            end
            if sucess == false
                return false
            end
        end
    end

    return sucess
end
function solveODEAllExperimentalConditions!(odeSolutionValues::AbstractMatrix,
                                            θ_dynamic::AbstractVector,
                                            petabODESolverCache::PEtabODESolverCache,
                                            odeSolutions::Dict{Symbol, Union{Nothing, ODESolution}},                                        
                                            __odeProblem::ODEProblem,
                                            petabModel::PEtabModel,
                                            simulationInfo::SimulationInfo,
                                            odeSolverOptions::ODESolverOptions,
                                            θ_indices::ParameterIndices;
                                            expIDSolve::Vector{Symbol} = [:all],
                                            nTimePointsSave::Int64=0,
                                            onlySaveAtObservedTimes::Bool=false,
                                            denseSolution::Bool=true,
                                            trackCallback::Bool=false, 
                                            computeForwardSensitivites::Bool=false)

    _odeProblem = remake(__odeProblem, p = convert.(eltype(θ_dynamic), __odeProblem.p), u0 = convert.(eltype(θ_dynamic), __odeProblem.u0))
    changeODEProblemParameters!(_odeProblem.p, _odeProblem.u0, θ_dynamic, θ_indices, petabModel)

    sucess = solveODEAllExperimentalConditions!(odeSolutions,
                                                _odeProblem,
                                                petabModel,
                                                θ_dynamic,
                                                petabODESolverCache,
                                                simulationInfo,
                                                θ_indices,
                                                odeSolverOptions,
                                                expIDSolve=expIDSolve,
                                                nTimePointsSave=nTimePointsSave,
                                                onlySaveAtObservedTimes=onlySaveAtObservedTimes,
                                                denseSolution=denseSolution,
                                                trackCallback=trackCallback, 
                                                computeForwardSensitivites=computeForwardSensitivites)

    # Effectively we return a big-array with the ODE-solutions accross all experimental conditions, where
    # each column is a time-point.
    if sucess != true
        odeSolutionValues .= 0.0
        return
    end

    # iStart and iEnd tracks which entries in odeSolutionValues we store a specific experimental condition
    iStart, iEnd = 1, 0
    for i in eachindex(simulationInfo.experimentalConditionId)
        experimentalId = simulationInfo.experimentalConditionId[i]
        iEnd += length(simulationInfo.timeObserved[experimentalId])
        if expIDSolve[1] == :all || simulationInfo.experimentalConditionId[i] ∈ expIDSolve
            @views odeSolutionValues[:, iStart:iEnd] .= Array(odeSolutions[experimentalId])
        end
        iStart = iEnd + 1
    end
end


function solveODEAllExperimentalConditions(_odeProblem::ODEProblem,
                                           petabModel::PEtabModel,
                                           θ_dynamic::AbstractVector,
                                           petabODESolverCache::PEtabODESolverCache,
                                           simulationInfo::SimulationInfo,
                                           θ_indices::ParameterIndices,
                                           odeSolverOptions::ODESolverOptions;
                                           expIDSolve::Vector{Symbol} = [:all],
                                           nTimePointsSave::Int64=0,
                                           onlySaveAtObservedTimes::Bool=false,
                                           denseSolution::Bool=true,
                                           trackCallback::Bool=false, 
                                           computeForwardSensitivites::Bool=false)::Tuple{Dict{Symbol, Union{Nothing, ODESolution}}, Bool}

    odeSolutions = deepcopy(simulationInfo.odeSolutions)
    success = solveODEAllExperimentalConditions!(odeSolutions,
                                                 _odeProblem,
                                                 petabModel,
                                                 θ_dynamic,
                                                 petabODESolverCache,
                                                 simulationInfo,
                                                 θ_indices,
                                                 odeSolverOptions,
                                                 expIDSolve=expIDSolve,
                                                 nTimePointsSave=nTimePointsSave,
                                                 onlySaveAtObservedTimes=onlySaveAtObservedTimes,
                                                 denseSolution=denseSolution,
                                                 trackCallback=trackCallback, 
                                                 computeForwardSensitivites=computeForwardSensitivites)

    return odeSolutions, success
end


function solveODEPreEqulibrium!(uAtSS::AbstractVector,
                                uAtT0::AbstractVector,
                                odeProblem::ODEProblem,
                                changeExperimentalCondition!::Function,
                                preEquilibrationId::Symbol,
                                odeSolverOptions::ODESolverOptions,
                                callbackSS::SciMLBase.DECallback,
                                convertTspan::Bool)::ODESolution

    # Change to parameters for the preequilibration simulations
    changeExperimentalCondition!(odeProblem.p, odeProblem.u0, preEquilibrationId)
    _odeProblem = remakeODEProblemPreSolve(odeProblem, Inf, odeSolverOptions.solver, convertTspan)
    uAtT0 .= _odeProblem.u0

    # Terminate if a steady state was not reached in preequilibration simulations
    odeSolution = computeODEPreEqulibriumSolution(_odeProblem, odeSolverOptions, callbackSS)
    if odeSolution.retcode == ReturnCode.Terminated
        uAtSS .= odeSolution.u[end]
    end
    return odeSolution
end


function computeODEPreEqulibriumSolution(odeProblem::ODEProblem,
                                         odeSolverOptions::ODESolverOptions,
                                         callbackSS::SciMLBase.DECallback)::ODESolution


    solver, abstol, reltol, force_dtmin, dtmin, maxiters = odeSolverOptions.solver, odeSolverOptions.abstol, odeSolverOptions.reltol, odeSolverOptions.force_dtmin, odeSolverOptions.dtmin, odeSolverOptions.maxiters
    return solve(odeProblem, solver, abstol=abstol, reltol=reltol, force_dtmin=force_dtmin, maxiters=maxiters, dense=false, callback=callbackSS)
end


function solveODEPostEqulibrium(odeProblem::ODEProblem,
                                uAtSS::AbstractVector,
                                u0AtT0::AbstractVector,
                                changeExperimentalCondition!::Function,
                                simulationInfo::SimulationInfo,
                                simulationConditionId::Symbol,
                                experimentalId::Symbol,
                                tMax::Float64,
                                odeSolverOptions::ODESolverOptions,
                                computeTStops::Function;
                                tSave::Vector{Float64}=Float64[],
                                denseSolution::Bool=true,
                                trackCallback::Bool=false,
                                convertTspan::Bool=false)::ODESolution

    changeExperimentalCondition!(odeProblem.p, odeProblem.u0, simulationConditionId)
    # Sometimes the experimentaCondition-file changes the initial values for a state
    # whose value was changed in the preequilibration-simulation. The experimentaCondition
    # value is prioritized by only changing u0 to the steady state value for those states
    # that were not affected by change to shiftExpId.
    hasNotChanged = (odeProblem.u0 .== u0AtT0)
    @views odeProblem.u0[hasNotChanged] .= uAtSS[hasNotChanged]

    # Here it is IMPORTANT that we copy odeProblem.p[:] else different experimental conditions will
    # share the same parameter vector p. This will, for example, cause the lower level adjoint
    # sensitivity interface to fail.
    _odeProblem = remakeODEProblemPreSolve(odeProblem, tMax, odeSolverOptions.solver, convertTspan)
    @views _odeProblem.u0 .= odeProblem.u0[:] # This is needed due as remake does not work correctly for forward sensitivity equations

    # If case of adjoint sensitivity analysis we need to track the callback to get correct gradients
    tStops = computeTStops(_odeProblem.u0, _odeProblem.p)
    callbackSet = getCallbackSet(_odeProblem, simulationInfo, experimentalId, trackCallback)
    sol = computeODESolution(_odeProblem, odeSolverOptions, simulationInfo.absTolSS, simulationInfo.relTolSS,
                             tSave, denseSolution, callbackSet, tStops)

    return sol
end


function solveODENoPreEqulibrium!(odeProblem::ODEProblem,
                                 changeExperimentalCondition!::Function,
                                 simulationInfo::SimulationInfo,
                                 simulationConditionId::Symbol,
                                 odeSolverOptions::ODESolverOptions,
                                 _tMax::Float64,
                                 computeTStops::Function;
                                 tSave=Float64[],
                                 denseSolution::Bool=true,
                                 trackCallback::Bool=false,
                                 convertTspan::Bool=false)::ODESolution

    # Change experimental condition
    tMax = isinf(_tMax) ? 1e8 : _tMax
    changeExperimentalCondition!(odeProblem.p, odeProblem.u0, simulationConditionId)
    _odeProblem = remakeODEProblemPreSolve(odeProblem, tMax, odeSolverOptions.solver, convertTspan)
    @views _odeProblem.u0 .= odeProblem.u0[:] # Required remake does not handle Senstivity-problems correctly

    tStops = computeTStops(_odeProblem.u0, _odeProblem.p)
    callbackSet = getCallbackSet(_odeProblem, simulationInfo, simulationConditionId, trackCallback)
    sol = computeODESolution(_odeProblem, odeSolverOptions, simulationInfo.absTolSS, simulationInfo.relTolSS,
                             tSave, denseSolution, callbackSet, tStops)

    return sol
end


function computeODESolution(odeProblem::ODEProblem,
                            odeSolverOptions::ODESolverOptions,
                            absTolSS::Float64,
                            relTolSS::Float64,
                            tSave::Vector{Float64},
                            denseSolution::Bool,
                            callbackSet::SciMLBase.DECallback,
                            tStops::AbstractVector)::ODESolution

    solver, abstol, reltol, force_dtmin, dtmin, maxiters = odeSolverOptions.solver, odeSolverOptions.abstol, odeSolverOptions.reltol, odeSolverOptions.force_dtmin, odeSolverOptions.dtmin, odeSolverOptions.maxiters                            

    # Different funcion calls to solve are required if a solver or a Alg-hint are provided.
    # If t_max = inf the model is simulated to steady state using the TerminateSteadyState callback.
    if isinf(odeProblem.tspan[2]) || odeProblem.tspan[2] == 1e8
        sol = solve(odeProblem, solver, abstol=abstol, reltol=reltol, force_dtmin=force_dtmin, maxiters=maxiters, save_on=false, save_end=true, dense=denseSolution, callback=TerminateSteadyState(absTolSS, relTolSS))
    else
        sol = solve(odeProblem, solver, abstol=abstol, reltol=reltol, force_dtmin=force_dtmin, maxiters=maxiters, saveat=tSave, dense=denseSolution, tstops=tStops, callback=callbackSet)
    end
    return sol
end


function getCallbackSet(odeProblem::ODEProblem,
                        simulationInfo::SimulationInfo,
                        simulationConditionId::Symbol,
                        trackCallback::Bool)::SciMLBase.DECallback

    if trackCallback == true
        cbSet = SciMLSensitivity.track_callbacks(simulationInfo.callbacks[simulationConditionId], odeProblem.tspan[1], 
                                                 odeProblem.u0, odeProblem.p, simulationInfo.sensealg) 
        simulationInfo.trackedCallbacks[simulationConditionId] = cbSet
        return cbSet
    end
    return simulationInfo.callbacks[simulationConditionId]
end


function checkError(e)
    if e isa BoundsError
        println("Bounds error ODE solve")
    elseif e isa DomainError
        println("Domain error on ODE solve")
    elseif e isa SingularException
        println("Singular exception on ODE solve")
    else
        rethrow(e)
    end
end


function remakeODEProblemPreSolve(odeProblem::ODEProblem,
                                  tmax::Float64,
                                  odeSolver::SciMLAlgorithm,
                                  convertTspan::Bool)::ODEProblem

    # When tmax=Inf and a multistep BDF Julia method, e.g. QNDF, is used tmax must be inf, else if it is a large
    # number such as 1e8 the dt_min is set to a large value making the solver fail. Sundials solvers on the other
    # hand are not compatible with timespan = (0.0, Inf), hence for these we use timespan = (0.0, 1e8)
    tmax::Float64 = _getTmax(tmax, odeSolver)
    if convertTspan == false
        return remake(odeProblem, tspan = (0.0, tmax))
    else
        return remake(odeProblem, tspan = convert.(eltype(odeProblem.p), (0.0, tmax)))
    end
end


function _getTmax(tmax::Float64, odeSolver::Union{CVODE_BDF, CVODE_Adams})::Float64
    return isinf(tmax) ? 1e8 : tmax
end
function _getTmax(tmax::Float64, odeSolver::Union{Vector{Symbol}, SciMLAlgorithm})::Float64
    return tmax
end


# Each soluation needs to have a unique vector associated with it such that the gradient 
# is correct computed for non-dynamic parameters (condition specific parameters are mapped 
# correctly) as these computations employ odeProblem.p
function setODEProblemParameters(_odeProblem::ODEProblem, petabODESolverCache::PEtabODESolverCache, conditionId::Symbol)::ODEProblem 
    
    # Ensure parameters constant between conditions are set correctly 
    pODEProblem = get_tmp(petabODESolverCache.pODEProblemCache[conditionId], _odeProblem.p)
    u0 = get_tmp(petabODESolverCache.u0Cache[conditionId], _odeProblem.p)
    pODEProblem .= _odeProblem.p
    @views u0 .= _odeProblem.u0[1:length(u0)]
    
    return remake(_odeProblem, p = pODEProblem, u0=u0)
    #return remake(_odeProblem, p = _odeProblem.p[:], u0=_odeProblem.u0[:])
end

