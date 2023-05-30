#=
    Functionallity for solving a PEtab ODE-model across all experimental conditions. Code is compatible with ForwardDiff,
    and there is functionallity for computing the sensitivity matrix.
=#


function solveODEAllExperimentalConditions!(odeSolutions::Dict{Symbol, Union{Nothing, ODESolution}},
                                            odeProblem::ODEProblem,
                                            petabModel::PEtabModel,
                                            θ_dynamic::AbstractVector,
                                            petabODESolverCache::PEtabODESolverCache,
                                            simulationInfo::SimulationInfo,
                                            θ_indices::ParameterIndices,
                                            odeSolverOptions::ODESolverOptions,
                                            ssSolverOptions::SteadyStateSolverOptions;
                                            expIDSolve::Vector{Symbol} = [:all],
                                            nTimePointsSave::Int64=0,
                                            onlySaveAtObservedTimes::Bool=false,
                                            denseSolution::Bool=true,
                                            trackCallback::Bool=false,
                                            computeForwardSensitivites::Bool=false)::Bool # Required for adjoint sensitivity analysis


    changeExperimentalCondition! = (pODEProblem, u0, conditionId) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices, computeForwardSensitivites=computeForwardSensitivites)
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
        uAtSS = Matrix{eltype(θ_dynamic)}(undef, (length(odeProblem.u0), length(preEquilibrationId)))
        u0AtT0 = Matrix{eltype(θ_dynamic)}(undef, (length(odeProblem.u0), length(preEquilibrationId)))

        for i in eachindex(preEquilibrationId)

            _odeProblem = setODEProblemParameters(odeProblem, petabODESolverCache, preEquilibrationId[i])
            # Sometimes due to strongly ill-conditioned Jacobian the linear-solve runs
            # into a domain error or bounds error. This is treated as integration error.
            try
                _odeSolutions = simulationInfo.odePreEqulibriumSolutions
                _odeSolutions[preEquilibrationId[i]] = solveODEPreEqulibrium!((@view uAtSS[:, i]),
                                                                              (@view u0AtT0[:, i]),
                                                                               _odeProblem,
                                                                               changeExperimentalCondition!,
                                                                               preEquilibrationId[i],
                                                                               odeSolverOptions,
                                                                               ssSolverOptions,
                                                                               petabModel.convertTspan)
            catch e
                checkError(e)
                return false
            end
            if simulationInfo.odePreEqulibriumSolutions[preEquilibrationId[i]].retcode != ReturnCode.Terminated
                return false
            end
        end
    end

    @inbounds for i in eachindex(simulationInfo.experimentalConditionId)
        experimentalId = simulationInfo.experimentalConditionId[i]

        if expIDSolve[1] != :all && experimentalId ∉ expIDSolve
            continue
        end

        # In case onlySaveAtObservedTimes=true all other options are overridden and we only save the data
        # at observed time points.
        _tMax = simulationInfo.timeMax[experimentalId]
        _tSave = getTimePointsSaveat(Val(onlySaveAtObservedTimes), simulationInfo, experimentalId, _tMax, nTimePointsSave)
        _denseSolution = shouldSaveDenseSolution(Val(onlySaveAtObservedTimes), nTimePointsSave, denseSolution)
        _odeProblem = setODEProblemParameters(odeProblem, petabODESolverCache, experimentalId)

        # In case we have a simulation with PreEqulibrium
        if simulationInfo.preEquilibrationConditionId[i] != :None
            whichIndex = findfirst(x -> x == simulationInfo.preEquilibrationConditionId[i], preEquilibrationId)
            # See comment above on domain error
            try
                odeSolutions[experimentalId] = solveODEPostEqulibrium(_odeProblem,
                                                                      (@view uAtSS[:, whichIndex]),
                                                                      (@view u0AtT0[:, whichIndex]),
                                                                      changeExperimentalCondition!,
                                                                      simulationInfo,
                                                                      simulationInfo.simulationConditionId[i],
                                                                      experimentalId,
                                                                      _tMax,
                                                                      odeSolverOptions,
                                                                      petabModel.computeTStops,
                                                                      _tSave,
                                                                      _denseSolution,
                                                                      trackCallback,
                                                                      petabModel.convertTspan)
            catch e
                checkError(e)
                return false
            end
            if odeSolutions[experimentalId].retcode != ReturnCode.Success
                return false
            end

        # In case we have an ODE solution without Pre-equlibrium
        else
            try
                odeSolutions[experimentalId] = solveODENoPreEqulibrium(_odeProblem,
                                                                       changeExperimentalCondition!,
                                                                       simulationInfo,
                                                                       simulationInfo.simulationConditionId[i],
                                                                       odeSolverOptions,
                                                                       _tMax,
                                                                       petabModel.computeTStops,
                                                                       _tSave,
                                                                       _denseSolution,
                                                                       trackCallback,
                                                                       petabModel.convertTspan)
            catch e
                checkError(e)
                return false
            end
            retcode = odeSolutions[experimentalId].retcode
            if !(retcode == ReturnCode.Success || retcode == ReturnCode.Terminated)
                return false
            end
        end
    end

    return true
end
function solveODEAllExperimentalConditions!(odeSolutionValues::AbstractMatrix,
                                            _θ_dynamic::AbstractVector,
                                            petabODESolverCache::PEtabODESolverCache,
                                            odeSolutions::Dict{Symbol, Union{Nothing, ODESolution}},
                                            odeProblem::ODEProblem,
                                            petabModel::PEtabModel,
                                            simulationInfo::SimulationInfo,
                                            odeSolverOptions::ODESolverOptions,
                                            ssSolverOptions::SteadyStateSolverOptions,
                                            θ_indices::ParameterIndices,
                                            petabODECache::PEtabODEProblemCache;
                                            expIDSolve::Vector{Symbol} = [:all],
                                            nTimePointsSave::Int64=0,
                                            onlySaveAtObservedTimes::Bool=false,
                                            denseSolution::Bool=true,
                                            trackCallback::Bool=false,
                                            computeForwardSensitivites::Bool=false,
                                            computeForwardSensitivitesAD::Bool=false)

    if computeForwardSensitivitesAD == true && petabODECache.nθ_dynamicEst[1] != length(_θ_dynamic)
        θ_dynamic = _θ_dynamic[petabODECache.θ_dynamicOutputOrder]
    else
        θ_dynamic = _θ_dynamic
    end

    _odeProblem = remake(odeProblem, p = convert.(eltype(θ_dynamic), odeProblem.p), u0 = convert.(eltype(θ_dynamic), odeProblem.u0))
    changeODEProblemParameters!(_odeProblem.p, _odeProblem.u0, θ_dynamic, θ_indices, petabModel)

    sucess = solveODEAllExperimentalConditions!(odeSolutions,
                                                _odeProblem,
                                                petabModel,
                                                θ_dynamic,
                                                petabODESolverCache,
                                                simulationInfo,
                                                θ_indices,
                                                odeSolverOptions,
                                                ssSolverOptions,
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


function solveODEAllExperimentalConditions(odeProblem::ODEProblem,
                                           petabModel::PEtabModel,
                                           θ_dynamic::AbstractVector,
                                           petabODESolverCache::PEtabODESolverCache,
                                           simulationInfo::SimulationInfo,
                                           θ_indices::ParameterIndices,
                                           odeSolverOptions::ODESolverOptions,
                                           ssSolverOptions::SteadyStateSolverOptions;
                                           expIDSolve::Vector{Symbol} = [:all],
                                           nTimePointsSave::Int64=0,
                                           onlySaveAtObservedTimes::Bool=false,
                                           denseSolution::Bool=true,
                                           trackCallback::Bool=false,
                                           computeForwardSensitivites::Bool=false)::Tuple{Dict{Symbol, Union{Nothing, ODESolution}}, Bool}

    odeSolutions = deepcopy(simulationInfo.odeSolutions)
    success = solveODEAllExperimentalConditions!(odeSolutions,
                                                 odeProblem,
                                                 petabModel,
                                                 θ_dynamic,
                                                 petabODESolverCache,
                                                 simulationInfo,
                                                 θ_indices,
                                                 odeSolverOptions,
                                                 ssSolverOptions,
                                                 expIDSolve=expIDSolve,
                                                 nTimePointsSave=nTimePointsSave,
                                                 onlySaveAtObservedTimes=onlySaveAtObservedTimes,
                                                 denseSolution=denseSolution,
                                                 trackCallback=trackCallback,
                                                 computeForwardSensitivites=computeForwardSensitivites)

    return odeSolutions, success
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
                                computeTStops::Function,
                                tSave::Vector{Float64},
                                denseSolution,
                                trackCallback,
                                convertTspan)::ODESolution

    changeExperimentalCondition!(odeProblem.p, odeProblem.u0, simulationConditionId)
    # Sometimes the experimentaCondition-file changes the initial values for a state
    # whose value was changed in the preequilibration-simulation. The experimentaCondition
    # value is prioritized by only changing u0 to the steady state value for those states
    # that were not affected by change to shiftExpId.
    hasNotChanged = (odeProblem.u0 .== u0AtT0)
    @views odeProblem.u0[hasNotChanged] .= uAtSS[hasNotChanged]

    # According to the PEtab standard we can sometimes have that initial assignment is overridden for
    # pre-eq simulation, but we do not want to override for main simulation which is done automatically
    # by changeExperimentalCondition!. These cases are marked as NaN
    isNan = isnan.(odeProblem.u0)
    @views odeProblem.u0[isNan] .= uAtSS[isNan]

    # Here it is IMPORTANT that we copy odeProblem.p[:] else different experimental conditions will
    # share the same parameter vector p. This will, for example, cause the lower level adjoint
    # sensitivity interface to fail.
    _odeProblem = setTspanODEProblem(odeProblem, tMax, odeSolverOptions.solver, convertTspan)
    @views _odeProblem.u0 .= odeProblem.u0[:] # This is needed due as remake does not work correctly for forward sensitivity equations

    # If case of adjoint sensitivity analysis we need to track the callback to get correct gradients
    tStops = computeTStops(_odeProblem.u0, _odeProblem.p)
    callbackSet = getCallbackSet(_odeProblem, simulationInfo, experimentalId, trackCallback)
    sol = computeODESolution(_odeProblem, odeSolverOptions.solver, odeSolverOptions, odeSolverOptions.abstol, odeSolverOptions.reltol,
                             tSave, denseSolution, callbackSet, tStops)

    return sol
end


function solveODENoPreEqulibrium(odeProblem::ODEProblem,
                                 changeExperimentalCondition!::F1,
                                 simulationInfo::SimulationInfo,
                                 simulationConditionId::Symbol,
                                 odeSolverOptions::ODESolverOptions,
                                 _tMax::Float64,
                                 computeTStops::F2,
                                 tSave::Vector{Float64},
                                 denseSolution::Bool,
                                 trackCallback::Bool,
                                 convertTspan::Bool)::ODESolution where {F1<:Function, F2<:Function}

    # Change experimental condition
    tMax = isinf(_tMax) ? 1e8 : _tMax
    changeExperimentalCondition!(odeProblem.p, odeProblem.u0, simulationConditionId)
    _odeProblem = setTspanODEProblem(odeProblem, tMax, odeSolverOptions.solver, convertTspan)
    @views _odeProblem.u0 .= odeProblem.u0[:] # Required remake does not handle Senstivity-problems correctly

    tStops = computeTStops(_odeProblem.u0, _odeProblem.p)
    callbackSet = getCallbackSet(_odeProblem, simulationInfo, simulationConditionId, trackCallback)
    sol = computeODESolution(_odeProblem, odeSolverOptions.solver, odeSolverOptions, odeSolverOptions.abstol, odeSolverOptions.reltol,
                             tSave, denseSolution, callbackSet, tStops)

    return sol
end


function computeODESolution(odeProblem::ODEProblem,
                            solver::S,
                            odeSolverOptions::ODESolverOptions,
                            absTolSS::Float64,
                            relTolSS::Float64,
                            tSave::Vector{Float64},
                            denseSolution::Bool,
                            callbackSet::SciMLBase.DECallback,
                            tStops::AbstractVector)::ODESolution where S<:SciMLAlgorithm

    abstol, reltol, force_dtmin, dtmin, maxiters = odeSolverOptions.abstol, odeSolverOptions.reltol, odeSolverOptions.force_dtmin, odeSolverOptions.dtmin, odeSolverOptions.maxiters

    # Different funcion calls to solve are required if a solver or a Alg-hint are provided.
    # If t_max = inf the model is simulated to steady state using the TerminateSteadyState callback.
    if isinf(odeProblem.tspan[2]) || odeProblem.tspan[2] == 1e8
        sol = solve(odeProblem, solver, abstol=abstol, reltol=reltol, force_dtmin=force_dtmin, maxiters=maxiters, save_on=false, save_start=false, save_end=true, dense=denseSolution, callback=TerminateSteadyState(absTolSS, relTolSS))
    else
        sol = solve(odeProblem, solver, abstol=abstol, reltol=reltol, force_dtmin=force_dtmin, maxiters=maxiters, saveat=tSave, dense=denseSolution, tstops=tStops, callback=callbackSet)
    end
    return sol
end


function checkError(e)
    if e isa BoundsError
        @warn "Bounds error ODE solve"
    elseif e isa DomainError
        @warn "Domain error on ODE solve"
    elseif e isa SingularException
        @warn "Singular exception on ODE solve"
    else
        rethrow(e)
    end
end