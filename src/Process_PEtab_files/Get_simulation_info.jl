#=
    Functions for processing the PEtab observables and measurements file into a Julia
    SimulationInfo struct, which contains information required to carry out forward ODE simulations
    (condition-id), and indices for mapping simulations to measurement data.
=#


function processSimulationInfo(petabModel::PEtabModel,
                               measurementInfo::MeasurementsInfo;
                               sensealg::Union{Nothing, Symbol, SciMLSensitivity.AbstractForwardSensitivityAlgorithm, SciMLSensitivity.AbstractAdjointSensitivityAlgorithm}=InterpolatingAdjoint())::SimulationInfo

    # An experimental Id is uniqely defined by a Pre-equlibrium- and Simulation-Id, where the former can be
    # empty. For each experimental ID we store three indices, i) preEqulibriumId, ii) simulationId and iii)
    # experimentalId (concatenation of two).
    preEquilibrationConditionId::Vector{Symbol} = Vector{Symbol}(undef, 0)
    simulationConditionId::Vector{Symbol} = Vector{Symbol}(undef, 0)
    experimentalConditionId::Vector{Symbol} = Vector{Symbol}(undef, 0)
    for i in eachindex(measurementInfo.preEquilibrationConditionId)
        # In case model has steady-state simulations prior to matching against data
        _preEquilibrationConditionId = measurementInfo.preEquilibrationConditionId[i]
        if _preEquilibrationConditionId == :None
            measurementInfo.simulationConditionId[i] ∈ experimentalConditionId && continue
            preEquilibrationConditionId = vcat(preEquilibrationConditionId, :None)
            simulationConditionId = vcat(simulationConditionId, measurementInfo.simulationConditionId[i])
            experimentalConditionId = vcat(experimentalConditionId, measurementInfo.simulationConditionId[i])
            continue
        end

        # For cases with no steady-state simulations
        _experimentalConditionId = Symbol(string(measurementInfo.preEquilibrationConditionId[i]) * string(measurementInfo.simulationConditionId[i]))
        if _experimentalConditionId ∉ experimentalConditionId
            preEquilibrationConditionId = vcat(preEquilibrationConditionId, measurementInfo.preEquilibrationConditionId[i])
            simulationConditionId = vcat(simulationConditionId, measurementInfo.simulationConditionId[i])
            experimentalConditionId = vcat(experimentalConditionId, _experimentalConditionId)
            continue
        end
    end

    haspreEquilibrationConditionId::Bool = all(preEquilibrationConditionId.== :None) ? false : true

    # When computing the gradient and hessian the ODE-system needs to be resolved to compute the gradient
    # of the dynamic parameters, while for the observable/sd parameters the system should not be resolved.
    # Hence we need a specific dictionary with ODE solutions when compuating derivatives.
    odeSolutions::Dict{Symbol, Union{Nothing, ODESolution}} = Dict{Symbol, ODESolution}()
    odePreEqulibriumSolutions::Dict{Symbol, Union{Nothing, ODESolution, SciMLBase.NonlinearSolution}} = Dict{Symbol, ODESolution}()
    odeSolutionsDerivatives::Dict{Symbol, Union{Nothing, ODESolution}} = Dict{Symbol, ODESolution}()
    for i in eachindex(experimentalConditionId)
        odeSolutions[experimentalConditionId[i]] = nothing
        odeSolutionsDerivatives[experimentalConditionId[i]] = nothing
        if preEquilibrationConditionId[i] != :None && preEquilibrationConditionId[i] ∉ keys(odePreEqulibriumSolutions)
            odePreEqulibriumSolutions[preEquilibrationConditionId[i]] = nothing
        end
    end

    # Precompute the max simulation time for each experimentalConditionId
    _timeMax = Tuple(computeTimeMax(preEquilibrationConditionId[i], simulationConditionId[i], measurementInfo) for i in eachindex(preEquilibrationConditionId))
    timeMax::Dict{Symbol, Float64} = Dict([experimentalConditionId[i] => _timeMax[i] for i in eachindex(_timeMax)])

    # Precompute which time-points we have observed data at for experimentalConditionId (used in saveat for ODE solution)
    _timeObserved = Tuple(computeTimeObserved(preEquilibrationConditionId[i], simulationConditionId[i], measurementInfo) for i in eachindex(preEquilibrationConditionId))
    timeObserved::Dict{Symbol, Vector{Float64}} = Dict([experimentalConditionId[i] => _timeObserved[i] for i in eachindex(_timeObserved)])

    # Precompute indices in measurementInfo (iMeasurement) for each experimentalConditionId
    _iMeasurementsObserved = Tuple(_computeTimeIndices(preEquilibrationConditionId[i], simulationConditionId[i], measurementInfo) for i in eachindex(preEquilibrationConditionId))
    iMeasurementsObserved::Dict{Symbol, Vector{Int64}} = Dict([experimentalConditionId[i] => _iMeasurementsObserved[i] for i in eachindex(_iMeasurementsObserved)])

    # Precompute for each measurement (entry in iMeasurement) a vector which holds the corresponding index in odeSolution.t
    # accounting for experimentalConditionId
    iTimeODESolution::Vector{Int64} = computeIndexTimeODESolution(preEquilibrationConditionId, simulationConditionId, measurementInfo)

    # When computing the gradients via forward sensitivity equations we need to track where, in the concatanated
    # odeSolution.t (accross all condition) the time-points for an experimental conditions start as we can only
    # compute the sensitivity matrix accross all conditions.
    timePositionInODESolutions::Dict{Symbol, UnitRange{Int64}} = getTimePositionInODESolutions(experimentalConditionId, timeObserved)

    # Precompute a vector of vector where vec[i] gives the indices for time-point ti in measurementInfo for an
    # experimentalConditionId. Needed for the lower level adjoint interface where we must track the number of
    # repats per time-point (when using dgdu_discrete and dgdp_discrete)
    _iPerTimePoint = Tuple(computeTimeIndices(preEquilibrationConditionId[i], simulationConditionId[i], measurementInfo) for i in eachindex(preEquilibrationConditionId))
    iPerTimePoint::Dict{Symbol, Vector{Vector{Int64}}} = Dict([(experimentalConditionId[i], _iPerTimePoint[i]) for i in eachindex(_iPerTimePoint)])

    # Some models, e.g those with time dependent piecewise statements, have callbacks encoded. When doing adjoint
    # sensitivity analysis we need to track these callbacks, hence they must be stored in simulationInfo.
    callbacks = Dict{Symbol, SciMLBase.DECallback}()
    trackedCallbacks = Dict{Symbol, SciMLBase.DECallback}()
    for name in experimentalConditionId
        callbacks[name] = deepcopy(petabModel.modelCallbackSet)
    end

    if typeof(sensealg) <: Symbol || isnothing(sensealg)
        sensealg = InterpolatingAdjoint()
    end

    simulationInfo = SimulationInfo(preEquilibrationConditionId,
                                    simulationConditionId,
                                    experimentalConditionId,
                                    haspreEquilibrationConditionId,
                                    odeSolutions,
                                    odeSolutionsDerivatives,
                                    odePreEqulibriumSolutions,
                                    timeMax,
                                    timeObserved,
                                    iMeasurementsObserved,
                                    iTimeODESolution,
                                    iPerTimePoint,
                                    timePositionInODESolutions,
                                    callbacks,
                                    trackedCallbacks,
                                    sensealg)
    return simulationInfo
end


function _computeTimeIndices(preEquilibrationConditionId::Symbol, simulationConditionId::Symbol, measurementInfo::MeasurementsInfo)::Vector{Int64}
    iTimePoints = findall(i -> preEquilibrationConditionId == measurementInfo.preEquilibrationConditionId[i] && simulationConditionId == measurementInfo.simulationConditionId[i], eachindex(measurementInfo.time))
    return iTimePoints
end


function computeTimeIndices(preEquilibrationConditionId::Symbol, simulationConditionId::Symbol, measurementInfo::MeasurementsInfo)::Vector{Vector{Int64}}
    _iTimePoints = _computeTimeIndices(preEquilibrationConditionId, simulationConditionId, measurementInfo)
    timePoints = measurementInfo.time[_iTimePoints]
    timePointsUnique = sort(unique(timePoints))
    iTimePoints = Vector{Vector{Int64}}(undef, length(timePointsUnique))
    for i in eachindex(iTimePoints)
        iTimePoints[i] = _iTimePoints[findall(x -> x == timePointsUnique[i], timePoints)]
    end

    return iTimePoints
end


function computeTimeMax(preEquilibrationConditionId::Symbol, simulationConditionId::Symbol, measurementInfo::MeasurementsInfo)::Float64
    iTimePoints = _computeTimeIndices(preEquilibrationConditionId, simulationConditionId, measurementInfo)
    return Float64(maximum(measurementInfo.time[iTimePoints]))
end


function computeTimeObserved(preEquilibrationConditionId::Symbol, simulationConditionId::Symbol, measurementInfo::MeasurementsInfo)::Vector{Float64}
    iTimePoints = _computeTimeIndices(preEquilibrationConditionId, simulationConditionId, measurementInfo)
    return sort(unique(measurementInfo.time[iTimePoints]))
end


# For each experimental condition (forward ODE-solution) compute index in odeSolution.t for any index
# iMeasurement in measurementInfo.time[iMeasurement]
function computeIndexTimeODESolution(preEquilibrationConditionId::Vector{Symbol},
                                     simulationConditionId::Vector{Symbol},
                                     measurementInfo::MeasurementsInfo)::Vector{Int64}

    iTimeODESolution::Vector{Int64} = Vector{Int64}(undef, length(measurementInfo.time))
    for i in eachindex(simulationConditionId)
        iTimePoints = _computeTimeIndices(preEquilibrationConditionId[i], simulationConditionId[i], measurementInfo)
        timePoints = measurementInfo.time[iTimePoints]
        timePointsUnique = sort(unique(timePoints))
        for iT in iTimePoints
            t = measurementInfo.time[iT]
            iTimeODESolution[iT] = findfirst(x -> x == t, timePointsUnique)
        end
    end
    return iTimeODESolution
end


# For each time-point in the concatanated odeSolution.t (accross all conditions) get which index it corresponds
# to in the concatanted timeObserved (accross all conditions). This is needed when computing forward sensitivites
# via forward mode automatic differentiation because here we get a big sensitivity matrix accross all experimental
# conditions, where S[i:(i+nStates)] row corresponds to the sensitivites at a specific time-point.
# An assumption made here is that we solve the ODE:s in the order of experimentalConditionId (which is true)
function getTimePositionInODESolutions(experimentalConditionId::Vector{Symbol},
                                       timeObserved::Dict)::Dict{Symbol, UnitRange{Int64}}

    iStart::Int64 = 1
    positionInODESolutions::Dict{Symbol, UnitRange{Int64}} = Dict{Symbol, UnitRange{Int64}}()
    for i in eachindex(experimentalConditionId)
        timeObservedCondition = timeObserved[experimentalConditionId[i]]
        _positionInODESolutionExpId = iStart:(iStart-1+length(timeObservedCondition))
        iStart = _positionInODESolutionExpId[end] + 1
        positionInODESolutions[experimentalConditionId[i]] = _positionInODESolutionExpId
    end

    return positionInODESolutions
end
