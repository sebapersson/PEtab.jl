#=
    Helper function for correctly solving (simulating) the ODE model
=#


function getTimePointsSaveat(::Val{onlySaveAtObservedTimes},
                             simulationInfo::SimulationInfo,
                             experimentalId::Symbol,
                             tMax::Float64,
                             nTimePointsSave::Int64)::Vector{Float64} where onlySaveAtObservedTimes

    # Check if we want to only save the ODE at specific time-points
    if onlySaveAtObservedTimes == true
        return simulationInfo.timeObserved[experimentalId]
    elseif nTimePointsSave > 0
        return collect(LinRange(0.0, tMax, nTimePointsSave))
    else
        return Float64[]
    end
end


function shouldSaveDenseSolution(::Val{onlySaveAtObservedTimes}, nTimePointsSave::Int64, denseSolution::Bool)::Bool where onlySaveAtObservedTimes

    # Check if we want to only save the ODE at specific time-points
    if onlySaveAtObservedTimes == true
        return false
    elseif nTimePointsSave > 0
        return false
    else
        return denseSolution
    end
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



function setTspanODEProblem(odeProblem::ODEProblem,
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
end
