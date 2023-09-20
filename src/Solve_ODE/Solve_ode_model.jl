#=
    Functionallity for solving a PEtab ODE-model across all experimental conditions. Code is compatible with ForwardDiff,
    and there is functionallity for computing the sensitivity matrix.
=#


function solveODEAllExperimentalConditions!(odeSolutions::Dict{Symbol, Union{Nothing, ODESolution}},
                                            odeProblem::ODEProblem,
                                            petab_model::PEtabModel,
                                            θ_dynamic::AbstractVector,
                                            petabODESolverCache::PEtabODESolverCache,
                                            simulation_info::SimulationInfo,
                                            θ_indices::ParameterIndices,
                                            ode_solver::ODESolver,
                                            ss_solver::SteadyStateSolver;
                                            expIDSolve::Vector{Symbol} = [:all],
                                            nTimePointsSave::Int64=0,
                                            onlySaveAtObservedTimes::Bool=false,
                                            denseSolution::Bool=true,
                                            trackCallback::Bool=false,
                                            computeForwardSensitivites::Bool=false)::Bool # Required for adjoint sensitivity analysis


    changeExperimentalCondition! = (pODEProblem, u0, conditionId) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petab_model, θ_indices, computeForwardSensitivites=computeForwardSensitivites)
    # In case the model is first simulated to a steady state
    if simulation_info.haspreEquilibrationConditionId == true

        # Extract all unique Pre-equlibrium conditions. If the code is run in parallell
        # (expIDSolve != [["all]]) the number of preEq cond. might be smaller than the
        # total number of preEq cond.
        if expIDSolve[1] == :all
            preEquilibrationId = unique(simulation_info.preEquilibrationConditionId)
        else
            whichId = findall(x -> x ∈ simulation_info.experimentalConditionId, expIDSolve)
            preEquilibrationId = unique(simulation_info.preEquilibrationConditionId[whichId])
        end

        # Arrays to store steady state (pre-eq) values.
        uAtSS = Matrix{eltype(θ_dynamic)}(undef, (length(odeProblem.u0), length(preEquilibrationId)))
        u0AtT0 = Matrix{eltype(θ_dynamic)}(undef, (length(odeProblem.u0), length(preEquilibrationId)))

        for i in eachindex(preEquilibrationId)

            _odeProblem = setODEProblemParameters(odeProblem, petabODESolverCache, preEquilibrationId[i])
            # Sometimes due to strongly ill-conditioned Jacobian the linear-solve runs
            # into a domain error or bounds error. This is treated as integration error.
            try
                _odeSolutions = simulation_info.odePreEqulibriumSolutions
                _odeSolutions[preEquilibrationId[i]] = solveODEPreEqulibrium!((@view uAtSS[:, i]),
                                                                              (@view u0AtT0[:, i]),
                                                                               _odeProblem,
                                                                               changeExperimentalCondition!,
                                                                               preEquilibrationId[i],
                                                                               ode_solver,
                                                                               ss_solver,
                                                                               petab_model.convert_tspan)
            catch e
                checkError(e)
                simulation_info.couldSolve[1] = false
                return false
            end
            if simulation_info.odePreEqulibriumSolutions[preEquilibrationId[i]].retcode != ReturnCode.Terminated
                simulation_info.couldSolve[1] = false
                return false
            end
        end
    end

    @inbounds for i in eachindex(simulation_info.experimentalConditionId)
        experimentalId = simulation_info.experimentalConditionId[i]

        if expIDSolve[1] != :all && experimentalId ∉ expIDSolve
            continue
        end

        # In case onlySaveAtObservedTimes=true all other options are overridden and we only save the data
        # at observed time points.
        _tMax = simulation_info.timeMax[experimentalId]
        _tSave = getTimePointsSaveat(Val(onlySaveAtObservedTimes), simulation_info, experimentalId, _tMax, nTimePointsSave)
        _denseSolution = shouldSaveDenseSolution(Val(onlySaveAtObservedTimes), nTimePointsSave, denseSolution)
        _odeProblem = setODEProblemParameters(odeProblem, petabODESolverCache, experimentalId)

        # In case we have a simulation with PreEqulibrium
        if simulation_info.preEquilibrationConditionId[i] != :None
            whichIndex = findfirst(x -> x == simulation_info.preEquilibrationConditionId[i], preEquilibrationId)
            # See comment above on domain error
            try
                odeSolutions[experimentalId] = solveODEPostEqulibrium(_odeProblem,
                                                                      (@view uAtSS[:, whichIndex]),
                                                                      (@view u0AtT0[:, whichIndex]),
                                                                      changeExperimentalCondition!,
                                                                      simulation_info,
                                                                      simulation_info.simulationConditionId[i],
                                                                      experimentalId,
                                                                      _tMax,
                                                                      ode_solver,
                                                                      petab_model.compute_tstops,
                                                                      _tSave,
                                                                      _denseSolution,
                                                                      trackCallback,
                                                                      petab_model.convert_tspan)
            catch e
                checkError(e)
                simulation_info.couldSolve[1] = false
                return false
            end
            if odeSolutions[experimentalId].retcode != ReturnCode.Success
                simulation_info.couldSolve[1] = false
                return false
            end

        # In case we have an ODE solution without Pre-equlibrium
        else
            try
                odeSolutions[experimentalId] = solveODENoPreEqulibrium(_odeProblem,
                                                                       changeExperimentalCondition!,
                                                                       simulation_info,
                                                                       simulation_info.simulationConditionId[i],
                                                                       ode_solver,
                                                                       _tMax,
                                                                       petab_model.compute_tstops,
                                                                       _tSave,
                                                                       _denseSolution,
                                                                       trackCallback,
                                                                       petab_model.convert_tspan)
            catch e
                checkError(e)
                simulation_info.couldSolve[1] = false
                return false
            end
            retcode = odeSolutions[experimentalId].retcode
            if !(retcode == ReturnCode.Success || retcode == ReturnCode.Terminated)
                simulation_info.couldSolve[1] = false
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
                                            petab_model::PEtabModel,
                                            simulation_info::SimulationInfo,
                                            ode_solver::ODESolver,
                                            ss_solver::SteadyStateSolver,
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
    changeODEProblemParameters!(_odeProblem.p, _odeProblem.u0, θ_dynamic, θ_indices, petab_model)

    sucess = solveODEAllExperimentalConditions!(odeSolutions,
                                                _odeProblem,
                                                petab_model,
                                                θ_dynamic,
                                                petabODESolverCache,
                                                simulation_info,
                                                θ_indices,
                                                ode_solver,
                                                ss_solver,
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
    for i in eachindex(simulation_info.experimentalConditionId)
        experimentalId = simulation_info.experimentalConditionId[i]
        iEnd += length(simulation_info.timeObserved[experimentalId])
        if expIDSolve[1] == :all || simulation_info.experimentalConditionId[i] ∈ expIDSolve
            @views odeSolutionValues[:, iStart:iEnd] .= Array(odeSolutions[experimentalId])
        end
        iStart = iEnd + 1
    end
end


function solveODEAllExperimentalConditions(odeProblem::ODEProblem,
                                           petab_model::PEtabModel,
                                           θ_dynamic::AbstractVector,
                                           petabODESolverCache::PEtabODESolverCache,
                                           simulation_info::SimulationInfo,
                                           θ_indices::ParameterIndices,
                                           ode_solver::ODESolver,
                                           ss_solver::SteadyStateSolver;
                                           expIDSolve::Vector{Symbol} = [:all],
                                           nTimePointsSave::Int64=0,
                                           onlySaveAtObservedTimes::Bool=false,
                                           denseSolution::Bool=true,
                                           trackCallback::Bool=false,
                                           computeForwardSensitivites::Bool=false)::Tuple{Dict{Symbol, Union{Nothing, ODESolution}}, Bool}

    odeSolutions = deepcopy(simulation_info.odeSolutions)
    success = solveODEAllExperimentalConditions!(odeSolutions,
                                                 odeProblem,
                                                 petab_model,
                                                 θ_dynamic,
                                                 petabODESolverCache,
                                                 simulation_info,
                                                 θ_indices,
                                                 ode_solver,
                                                 ss_solver,
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
                                simulation_info::SimulationInfo,
                                simulationConditionId::Symbol,
                                experimentalId::Symbol,
                                tMax::Float64,
                                ode_solver::ODESolver,
                                compute_tstops::Function,
                                tSave::Vector{Float64},
                                denseSolution,
                                trackCallback,
                                convert_tspan)::ODESolution

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
    _odeProblem = setTspanODEProblem(odeProblem, tMax, ode_solver.solver, convert_tspan)
    @views _odeProblem.u0 .= odeProblem.u0[:] # This is needed due as remake does not work correctly for forward sensitivity equations

    # If case of adjoint sensitivity analysis we need to track the callback to get correct gradients
    tStops = compute_tstops(_odeProblem.u0, _odeProblem.p)
    callbackSet = getCallbackSet(_odeProblem, simulation_info, experimentalId, simulation_info.sensealg)
    sol = computeODESolution(_odeProblem, ode_solver.solver, ode_solver, ode_solver.abstol, ode_solver.reltol,
                             tSave, denseSolution, callbackSet, tStops)

    return sol
end


function solveODENoPreEqulibrium(odeProblem::ODEProblem,
                                 changeExperimentalCondition!::F1,
                                 simulation_info::SimulationInfo,
                                 simulationConditionId::Symbol,
                                 ode_solver::ODESolver,
                                 _tMax::Float64,
                                 compute_tstops::F2,
                                 tSave::Vector{Float64},
                                 denseSolution::Bool,
                                 trackCallback::Bool,
                                 convert_tspan::Bool)::ODESolution where {F1<:Function, F2<:Function}

    # Change experimental condition
    tMax = isinf(_tMax) ? 1e8 : _tMax
    changeExperimentalCondition!(odeProblem.p, odeProblem.u0, simulationConditionId)
    _odeProblem = setTspanODEProblem(odeProblem, tMax, ode_solver.solver, convert_tspan)
    @views _odeProblem.u0 .= odeProblem.u0[:] # Required remake does not handle Senstivity-problems correctly

    tStops = compute_tstops(_odeProblem.u0, _odeProblem.p)
    callbackSet = getCallbackSet(_odeProblem, simulation_info, simulationConditionId, simulation_info.sensealg)
    sol = computeODESolution(_odeProblem, ode_solver.solver, ode_solver, ode_solver.abstol, ode_solver.reltol,
                             tSave, denseSolution, callbackSet, tStops)

    return sol
end


function computeODESolution(odeProblem::ODEProblem,
                            solver::S,
                            ode_solver::ODESolver,
                            absTolSS::Float64,
                            relTolSS::Float64,
                            tSave::Vector{Float64},
                            denseSolution::Bool,
                            callbackSet::SciMLBase.DECallback,
                            tStops::AbstractVector)::ODESolution where S<:SciMLAlgorithm

    abstol, reltol, force_dtmin, dtmin, maxiters = ode_solver.abstol, ode_solver.reltol, ode_solver.force_dtmin, ode_solver.dtmin, ode_solver.maxiters

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