#=
    Functionallity for solving a PEtab ODE-model across all experimental conditions. Code is compatible with ForwardDiff,
    and there is functionallity for computing the sensitivity matrix.
=#


function solveODEAllExperimentalConditions!(odeSolutions::Dict{Symbol, Union{Nothing, ODESolution}},
                                            odeProblem::ODEProblem,
                                            changeExperimentalCondition!::Function,
                                            simulationInfo::SimulationInfo,
                                            solver::Union{SciMLAlgorithm, Vector{Symbol}},
                                            absTol::Float64,
                                            relTol::Float64,
                                            computeTStops::Function; # For callbacks
                                            expIDSolve::Vector{Symbol} = [:all],
                                            nTimePointsSave::Int64=0,
                                            onlySaveAtObservedTimes::Bool=false,
                                            denseSolution::Bool=true,
                                            trackCallback::Bool=false,
                                            convertTspan::Bool=false)::Bool # Required for adjoint sensitivity analysis

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
        uAtSS = Matrix{eltype(odeProblem.p)}(undef, (length(odeProblem.u0), length(preEquilibrationId)))
        u0AtT0 = Matrix{eltype(odeProblem.p)}(undef, (length(odeProblem.u0), length(preEquilibrationId)))

        for i in eachindex(preEquilibrationId)
            # Sometimes due to strongly ill-conditioned Jacobian the linear-solve runs
            # into a domain error or bounds error. This is treated as integration error.
            try
                _odeSolutions = simulationInfo.odePreEqulibriumSolutions
                _odeSolutions[preEquilibrationId[i]] = solveODEPreEqulibrium!((@view uAtSS[:, i]),
                                                                              (@view u0AtT0[:, i]),
                                                                               odeProblem,
                                                                               changeExperimentalCondition!,
                                                                               preEquilibrationId[i],
                                                                               absTol,
                                                                               relTol,
                                                                               solver,
                                                                               simulationInfo.callbackSS,
                                                                               convertTspan)
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
                                                                      absTol,
                                                                      relTol,
                                                                      tMax,
                                                                      solver,
                                                                      computeTStops,
                                                                      tSave=tSave,
                                                                      denseSolution=denseSolution,
                                                                      trackCallback=trackCallback,
                                                                      convertTspan=convertTspan)

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
                                                                        absTol,
                                                                        relTol,
                                                                        solver,
                                                                        tMax,
                                                                        computeTStops,
                                                                        tSave=tSave,
                                                                        denseSolution=denseSolution,
                                                                        trackCallback=trackCallback,
                                                                        convertTspan=convertTspan)
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
function solveODEAllExperimentalConditions!(odeSolutions::Dict{Symbol, Union{Nothing, ODESolution}},
                                            odeProblem::ODEProblem,
                                            θ_dynamic::AbstractVector,
                                            __changeExperimentalCondition!::Function,
                                            simulationInfo::SimulationInfo,
                                            solver::Union{SciMLAlgorithm, Vector{Symbol}},
                                            absTol::Float64,
                                            relTol::Float64,
                                            computeTStops::Function;
                                            expIDSolve::Vector{Symbol} = [:all],
                                            nTimePointsSave::Int64=0,
                                            onlySaveAtObservedTimes::Bool=false,
                                            denseSolution::Bool=true,
                                            trackCallback::Bool=false,
                                            convertTspan::Bool=false)::Bool

    changeExperimentalCondition! = (pOdeProblem, u0, conditionId) -> __changeExperimentalCondition!(pOdeProblem, u0, conditionId, θ_dynamic)
    sucess = solveODEAllExperimentalConditions!(odeSolutions,
                                                odeProblem,
                                                changeExperimentalCondition!,
                                                simulationInfo,
                                                solver,
                                                absTol,
                                                relTol,
                                                computeTStops,
                                                expIDSolve=expIDSolve,
                                                nTimePointsSave=nTimePointsSave,
                                                onlySaveAtObservedTimes=onlySaveAtObservedTimes,
                                                denseSolution=denseSolution,
                                                trackCallback=trackCallback,
                                                convertTspan=convertTspan)
    return sucess
end
function solveODEAllExperimentalConditions!(odeSolutions::Dict{Symbol, Union{Nothing, ODESolution}},
                                            S::Matrix{Float64},
                                            odeProblem::ODEProblem,
                                            θ_dynamic::AbstractVector,
                                            __changeExperimentalCondition!::Function,
                                            changeODEProblemParameters::Function,
                                            simulationInfo::SimulationInfo,
                                            θ_indices::ParameterIndices,
                                            solver::Union{SciMLAlgorithm, Vector{Symbol}},
                                            absTol::Float64,
                                            relTol::Float64,
                                            computeTStops::Function,
                                            odeSolutionValues::Matrix{Float64};
                                            expIDSolve::Vector{Symbol} = [:all],
                                            nTimePointsSave::Int64=0,
                                            onlySaveAtObservedTimes::Bool=false,
                                            denseSolution::Bool=true,
                                            trackCallback::Bool=false,
                                            convertTspan::Bool=false,
                                            splitOverConditions::Bool=false,
                                            chunkSize::Union{Nothing, Int64}=nothing)::Bool

    function computeSensitivityMatrix!(odeSolutionValues::AbstractMatrix, θ::AbstractVector; _expIDSolve=expIDSolve)

        if convertTspan == false
            _odeProblem = remake(odeProblem, p = convert.(eltype(θ), odeProblem.p), u0 = convert.(eltype(θ), odeProblem.u0))
        else
            _odeProblem = remake(odeProblem, p = convert.(eltype(θ), odeProblem.p), u0 = convert.(eltype(θ), odeProblem.u0), tspan=convert.(eltype(θ), odeProblem.tspan))
        end

        changeODEProblemParameters(_odeProblem.p, _odeProblem.u0, θ)
        _changeExperimentalCondition! = (pOdeProblem, u0, conditionId) -> __changeExperimentalCondition!(pOdeProblem, u0, conditionId, θ)

        sucess = solveODEAllExperimentalConditions!(odeSolutions,
                                                    _odeProblem,
                                                    _changeExperimentalCondition!,
                                                    simulationInfo,
                                                    solver,
                                                    absTol,
                                                    relTol,
                                                    computeTStops,
                                                    expIDSolve=_expIDSolve,
                                                    nTimePointsSave=nTimePointsSave,
                                                    onlySaveAtObservedTimes=onlySaveAtObservedTimes,
                                                    denseSolution=denseSolution,
                                                    trackCallback=trackCallback,
                                                    convertTspan=convertTspan)

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
            if _expIDSolve[1] == :all || simulationInfo.experimentalConditionId[i] ∈ _expIDSolve
                @views odeSolutionValues[:, iStart:iEnd] .= Array(odeSolutions[experimentalId])
            end
            iStart = iEnd + 1
        end
    end

    if splitOverConditions == false
        # Compute sensitivity matrix via forward mode automatic differentation
        if !isnothing(chunkSize)
            cfg = ForwardDiff.JacobianConfig(computeSensitivityMatrix!, odeSolutionValues, θ_dynamic, ForwardDiff.Chunk(chunkSize))
        else
            cfg = ForwardDiff.JacobianConfig(computeSensitivityMatrix!, odeSolutionValues, θ_dynamic, ForwardDiff.Chunk(θ_dynamic))
        end
        ForwardDiff.jacobian!(S, computeSensitivityMatrix!, odeSolutionValues, θ_dynamic, cfg)

    elseif splitOverConditions == true && simulationInfo.haspreEquilibrationConditionId == false
        S .= 0.0
        Stmp = similar(S)
        for conditionId in simulationInfo.experimentalConditionId
            mapConditionId = θ_indices.mapsConiditionId[conditionId]
            iθ_experimentalCondition = unique(vcat(θ_indices.mapODEProblem.iθDynamic, mapConditionId.iθDynamic))
            θ_input = θ_dynamic[iθ_experimentalCondition]
            computeSensitivityMatrixExpCond! = (odeSolutionValues, θ_arg) ->    begin
                                                                                    _θ_dynamic = convert.(eltype(θ_arg), θ_dynamic)
                                                                                    _θ_dynamic[iθ_experimentalCondition] .= θ_arg
                                                                                    computeSensitivityMatrix!(odeSolutionValues, _θ_dynamic, _expIDSolve=[conditionId])
                                                                                end
            @views ForwardDiff.jacobian!(Stmp[:, iθ_experimentalCondition], computeSensitivityMatrixExpCond!, odeSolutionValues, θ_input)
            @views S[:, iθ_experimentalCondition] .+= Stmp[:, iθ_experimentalCondition]
        end

    else splitOverConditions == true && simulationInfo.haspreEquilibrationConditionId == true
        println("Compatabillity error : Currently we only support to split gradient compuations accross experimentalConditionId:s for models without preequilibration")
    end

    # Check retcode of sensitivity matrix ODE solutions
    sucess = true
    for experimentalId in simulationInfo.experimentalConditionId
        if expIDSolve[1] != :all && experimentalId ∉ expIDSolve
            continue
        end
        retcode = odeSolutions[experimentalId].retcode
        if !(retcode == ReturnCode.Success || retcode == ReturnCode.Terminated)
            sucess = false
        end
    end

    return sucess
end


function solveODEAllExperimentalConditions(odeProblem::ODEProblem,
                                           changeExperimentalCondition!::Function,
                                           simulationInfo::SimulationInfo,
                                           solver::Union{SciMLAlgorithm, Vector{Symbol}},
                                           absTol::Float64,
                                           relTol::Float64,
                                           computeTStops::Function; # For callbacks
                                           expIDSolve::Vector{Symbol} = [:all],
                                           nTimePointsSave::Int64=0,
                                           onlySaveAtObservedTimes::Bool=false,
                                           denseSolution::Bool=true,
                                           trackCallback::Bool=false,
                                           convertTspan::Bool=false)::Tuple{Dict{Symbol, Union{Nothing, ODESolution}}, Bool}

    odeSolutions = deepcopy(simulationInfo.odeSolutions)
    success = solveODEAllExperimentalConditions!(odeSolutions,
                                                 odeProblem,
                                                 changeExperimentalCondition!,
                                                 simulationInfo,
                                                 solver,
                                                 absTol,
                                                 relTol,
                                                 computeTStops,
                                                 expIDSolve=expIDSolve,
                                                 nTimePointsSave=nTimePointsSave,
                                                 onlySaveAtObservedTimes=onlySaveAtObservedTimes,
                                                 denseSolution=denseSolution,
                                                 trackCallback=trackCallback,
                                                 convertTspan=convertTspan)

    return odeSolutions, success
end


function solveODEPreEqulibrium!(uAtSS::AbstractVector,
                                uAtT0::AbstractVector,
                                odeProblem::ODEProblem,
                                changeExperimentalCondition!::Function,
                                preEquilibrationId::Symbol,
                                absTol::Float64,
                                relTol::Float64,
                                solver::Union{SciMLAlgorithm, Vector{Symbol}},
                                callbackSS::SciMLBase.DECallback,
                                convertTspan::Bool)::ODESolution

    # Change to parameters for the preequilibration simulations
    changeExperimentalCondition!(odeProblem.p, odeProblem.u0, preEquilibrationId)
    _odeProblem = remakeODEProblemPreSolve(odeProblem, Inf, solver, convertTspan)
    uAtT0 .= _odeProblem.u0

    # Terminate if a steady state was not reached in preequilibration simulations
    odeSolution = computeODEPreEqulibriumSolution(_odeProblem, solver, absTol, relTol, callbackSS)
    if odeSolution.retcode == ReturnCode.Terminated
        uAtSS .= odeSolution.u[end]
    end
    return odeSolution
end


function computeODEPreEqulibriumSolution(odeProblem::ODEProblem,
                                         solver::Vector{Symbol},
                                         absTol::Float64,
                                         relTol::Float64,
                                         callbackSS::SciMLBase.DECallback)::ODESolution

    return solve(odeProblem, alg_hints=solver, abstol=absTol, reltol=relTol, dense=false, callback=callbackSS)
end
function computeODEPreEqulibriumSolution(odeProblem::ODEProblem,
                                         solver::SciMLAlgorithm,
                                         absTol::Float64,
                                         relTol::Float64,
                                         callbackSS::SciMLBase.DECallback)::ODESolution

    return solve(odeProblem, solver, abstol=absTol, reltol=relTol, dense=false, callback=callbackSS)
end


function solveODEPostEqulibrium(odeProblem::ODEProblem,
                                uAtSS::AbstractVector,
                                u0AtT0::AbstractVector,
                                changeExperimentalCondition!::Function,
                                simulationInfo::SimulationInfo,
                                simulationConditionId::Symbol,
                                experimentalId::Symbol,
                                absTol::Float64,
                                relTol::Float64,
                                tMax::Float64,
                                solver::Union{SciMLAlgorithm, Vector{Symbol}},
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
    odeProblem.u0[hasNotChanged] .= uAtSS[hasNotChanged]

    # Here it is IMPORTANT that we copy odeProblem.p[:] else different experimental conditions will
    # share the same parameter vector p. This will, for example, cause the lower level adjoint
    # sensitivity interface to fail.
    _odeProblem = remakeODEProblemPreSolve(odeProblem, tMax, solver, convertTspan)
    _odeProblem.u0 .= odeProblem.u0[:] # This is needed due as remake does not work correctly for forward sensitivity equations

    # If case of adjoint sensitivity analysis we need to track the callback to get correct gradients
    tStops = computeTStops(_odeProblem.u0, _odeProblem.p)
    callbackSet = getCallbackSet(_odeProblem, simulationInfo, experimentalId, trackCallback)
    sol = computeODESolution(_odeProblem, solver, absTol, relTol, simulationInfo.absTolSS, simulationInfo.relTolSS,
                             tSave, denseSolution, callbackSet, tStops)

    return sol
end


function solveODENoPreEqulibrium!(odeProblem::ODEProblem,
                                 changeExperimentalCondition!::Function,
                                 simulationInfo::SimulationInfo,
                                 simulationConditionId::Symbol,
                                 absTol::Float64,
                                 relTol::Float64,
                                 solver::Union{SciMLAlgorithm, Vector{Symbol}},
                                 _tMax::Float64,
                                 computeTStops::Function;
                                 tSave=Float64[],
                                 denseSolution::Bool=true,
                                 trackCallback::Bool=false,
                                 convertTspan::Bool=false)::ODESolution

    # Change experimental condition
    tMax = isinf(_tMax) ? 1e8 : _tMax
    changeExperimentalCondition!(odeProblem.p, odeProblem.u0, simulationConditionId)
    _odeProblem = remakeODEProblemPreSolve(odeProblem, tMax, solver, convertTspan)
    _odeProblem.u0 .= odeProblem.u0[:] # Required remake does not handle Senstivity-problems correctly

    tStops = computeTStops(_odeProblem.u0, _odeProblem.p)
    callbackSet = getCallbackSet(_odeProblem, simulationInfo, simulationConditionId, trackCallback)
    sol = computeODESolution(_odeProblem, solver, absTol, relTol, simulationInfo.absTolSS, simulationInfo.relTolSS,
                             tSave, denseSolution, callbackSet, tStops)

    return sol
end


function computeODESolution(odeProblem::ODEProblem,
                            solver::Vector{Symbol},
                            absTol::Float64,
                            relTol::Float64,
                            absTolSS::Float64,
                            relTolSS::Float64,
                            tSave::Vector{Float64},
                            denseSolution::Bool,
                            callbackSet::SciMLBase.DECallback,
                            tStops::AbstractVector)::ODESolution

    # Different funcion calls to solve are required if a solver or a Alg-hint are provided.
    # If t_max = inf the model is simulated to steady state using the TerminateSteadyState callback.
    if isinf(odeProblem.tspan[2]) || odeProblem.tspan[2] == 1e8
        sol = solve(odeProblem, alg_hints=solver, abstol=absTol, reltol=relTol, save_on=false, save_end=true, dense=denseSolution, callback=TerminateSteadyState(absTolSS, relTolSS))
    else
        sol = solve(odeProblem, alg_hints=solver, abstol=absTol, reltol=relTol, saveat=tSave, dense=denseSolution, tstops=tStops, callback=callbackSet)
    end
    return sol
end
function computeODESolution(odeProblem::ODEProblem,
                            solver::SciMLAlgorithm,
                            absTol::Float64,
                            relTol::Float64,
                            absTolSS::Float64,
                            relTolSS::Float64,
                            tSave::Vector{Float64},
                            denseSolution::Bool,
                            callbackSet::SciMLBase.DECallback,
                            tStops::AbstractVector)::ODESolution

    # Different funcion calls to solve are required if a solver or a Alg-hint are provided.
    # If t_max = inf the model is simulated to steady state using the TerminateSteadyState callback.
    if isinf(odeProblem.tspan[2]) || odeProblem.tspan[2] == 1e8
        sol = solve(odeProblem, solver, abstol=absTol, reltol=relTol, save_on=false, save_end=true, dense=denseSolution, callback=TerminateSteadyState(absTolSS, relTolSS))
    else
        sol = solve(odeProblem, solver, abstol=absTol, reltol=relTol, saveat=tSave, dense=denseSolution, tstops=tStops, callback=callbackSet)
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
        simulationInfo.callbacks[simulationConditionId] = cbSet
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
                                  odeSolver::Union{Vector{Symbol}, SciMLAlgorithm},
                                  convertTspan::Bool)::ODEProblem

    # When tmax=Inf and a multistep BDF Julia method, e.g. QNDF, is used tmax must be inf, else if it is a large
    # number such as 1e8 the dt_min is set to a large value making the solver fail. Sundials solvers on the other
    # hand are not compatible with timespan = (0.0, Inf), hence for these we use timespan = (0.0, 1e8)
    tmax::Float64 = _getTmax(tmax, odeSolver)
    if convertTspan == false
        return remake(odeProblem, tspan = (0.0, tmax), p = odeProblem.p[:], u0 = odeProblem.u0[:])
    else
        return remake(odeProblem, tspan = convert.(eltype(odeProblem.p), (0.0, tmax)), p = odeProblem.p[:], u0 = odeProblem.u0[:])
    end
end


function _getTmax(tmax::Float64, odeSolver::Union{CVODE_BDF, CVODE_Adams})::Float64
    return isinf(tmax) ? 1e8 : tmax
end
function _getTmax(tmax::Float64, odeSolver::Union{Vector{Symbol}, SciMLAlgorithm})::Float64
    return tmax
end
