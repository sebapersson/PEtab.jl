#=
    Functions for processing the PEtab observables and measurements file into a Julia
    SimulationInfo struct, which contains information required to carry out forward ODE simulations
    (condition-id), and indices for mapping simulations to measurement data.
=#


function processSimulationInfo(petabModel::PEtabModel,
                               measurementInfo::MeasurementsInfo,
                               parameterInfo::ParametersInfo;
                               absTolSS::Float64=1e-8,
                               relTolSS::Float64=1e-6,
                               sensealg::Union{SciMLSensitivity.AbstractForwardSensitivityAlgorithm, SciMLSensitivity.AbstractAdjointSensitivityAlgorithm}=InterpolatingAdjoint(),
                               terminateSSMethod::Symbol=:Norm,
                               sensealgForwardEquations::Union{Symbol, SciMLSensitivity.AbstractForwardSensitivityAlgorithm}=ForwardSensitivity())::SimulationInfo

    # An experimental Id is uniqely defined by a Pre-equlibrium- and Simulation-Id, where the former can be
    # empty. For each experimental ID we store three indices, i) preEqulibriumId, ii) simulationId and iii)
    # experimentalId (concatenation of two).
    preEquilibrationConditionId::Vector{Symbol} = Vector{Symbol}(undef, 0)
    simulationConditionId::Vector{Symbol} = Vector{Symbol}(undef, 0)
    experimentalConditionId::Vector{Symbol} = Vector{Symbol}(undef, 0)
    for i in eachindex(measurementInfo.preEquilibrationConditionId)
        if measurementInfo.preEquilibrationConditionId[i] == :None
            if measurementInfo.simulationConditionId[i] ∉ experimentalConditionId
                preEquilibrationConditionId = vcat(preEquilibrationConditionId, :None)
                simulationConditionId = vcat(simulationConditionId, measurementInfo.simulationConditionId[i])
                experimentalConditionId = vcat(experimentalConditionId, measurementInfo.simulationConditionId[i])
            end
        else
            _experimentalConditionId = Symbol(string(measurementInfo.preEquilibrationConditionId[i]) * string(measurementInfo.simulationConditionId[i]))
            if _experimentalConditionId ∉ experimentalConditionId
                preEquilibrationConditionId = vcat(preEquilibrationConditionId, measurementInfo.preEquilibrationConditionId[i])
                simulationConditionId = vcat(simulationConditionId, measurementInfo.simulationConditionId[i])
                experimentalConditionId = vcat(experimentalConditionId, _experimentalConditionId)
            end
        end
    end

    haspreEquilibrationConditionId::Bool = all(preEquilibrationConditionId.== :None) ? false : true
    nExperimentalConditionId = length(experimentalConditionId)

    # When computing the gradient and hessian the ODE-system needs to be resolved to compute the gradient
    # of the dynamic parameters, while for the observable/sd parameters the system should not be resolved.
    # Hence we need a specific dictionary with ODE solutions when compuating derivatives.
    odeSolutions::Dict{Symbol, Union{Nothing, ODESolution}} = Dict{Symbol, ODESolution}()
    odePreEqulibriumSolutions::Dict{Symbol, Union{Nothing, ODESolution}} = Dict{Symbol, ODESolution}()
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
    timeMax::NamedTuple = NamedTuple{Tuple(name for name in experimentalConditionId)}(_timeMax)

    # Precompute which time-points we have observed data at for experimentalConditionId (used in saveat for ODE solution)
    _timeObserved = Tuple(computeTimeObserved(preEquilibrationConditionId[i], simulationConditionId[i], measurementInfo) for i in eachindex(preEquilibrationConditionId))
    timeObserved::NamedTuple = NamedTuple{Tuple(name for name in experimentalConditionId)}(_timeObserved)

    # Precompute indices in measurementInfo (iMeasurement) for each experimentalConditionId
    _iMeasurementsObserved = Tuple(_computeTimeIndices(preEquilibrationConditionId[i], simulationConditionId[i], measurementInfo) for i in eachindex(preEquilibrationConditionId))
    iMeasurementsObserved::NamedTuple = NamedTuple{Tuple(name for name in experimentalConditionId)}(_iMeasurementsObserved)

    # Precompute for each measurement (entry in iMeasurement) a vector which holds the corresponding index in odeSolution.t
    # accounting for experimentalConditionId
    iTimeODESolution = computeIndexTimeODESolution(preEquilibrationConditionId, simulationConditionId, measurementInfo)

    # Precompute a vector of vector where vec[i] gives the indices for time-point ti in measurementInfo for an
    # experimentalConditionId. Needed for the lower level adjoint interface where we must track the number of
    # repats per time-point (when using dgdu_discrete and dgdp_discrete)
    _iPerTimePoint = Tuple(computeTimeIndices(preEquilibrationConditionId[i], simulationConditionId[i], measurementInfo) for i in eachindex(preEquilibrationConditionId))
    iPerTimePoint::NamedTuple = NamedTuple{Tuple(name for name in experimentalConditionId)}(_iPerTimePoint)

    # When computing the gradients via forward sensitivity equations we need to track where, in the concatanated
    # odeSolution.t (accross all condition) the time-points for an experimental conditions start as we can only
    # compute the sensitivity matrix accross all conditions.
    timePositionInODESolutions::NamedTuple = getTimePositionInODESolutions(experimentalConditionId, timeObserved)

    # Some models, e.g those with time dependent piecewise statements, have callbacks encoded. When doing adjoint
    # sensitivity analysis we need to track these callbacks, hence they must be stored in simulationInfo.
    callbacks = Dict{Symbol, SciMLBase.DECallback}()
    for name in experimentalConditionId
        callbacks[name] = deepcopy(petabModel.modelCallbackSet)
    end

    # Ger terminate SS callbacks
    if terminateSSMethod == :Norm
        callbackSS = TerminateSteadyState(absTolSS, relTolSS)
    elseif terminateSSMethod == :NewtonNorm
        callbackSS = createSSTerminateSteadyState(petabModel.odeSystem, absTolSS, relTolSS, checkNewton=false)
    end

    # In case the sensitivites are computed via automatic differentitation we need to pre-allocate an
    # sensitivity matrix all experimental conditions (to efficiently levarage autodiff and handle scenarios are
    # pre-equlibrita model). Here we pre-allocate said matrix, or leave it empty.
    if sensealgForwardEquations == :AutoDiffForward
        experimentalConditionsFile = CSV.read(petabModel.pathConditions, DataFrame)
        tmp1, tmp2, tmp3, θ_dynamicNames = computeθNames(parameterInfo, measurementInfo,
                                                         petabModel.odeSystem, experimentalConditionsFile)
        nModelStates = length(states(petabModel.odeSystem))
        nTimePointsSaveAt = sum(length(timeObserved[experimentalConditionId]) for experimentalConditionId in experimentalConditionId)
        S = zeros(Float64, (nTimePointsSaveAt*nModelStates, length(θ_dynamicNames)))
    else
        S = zeros(Float64, (0, 0))
    end

    simulationInfo = SimulationInfo(preEquilibrationConditionId,
                                    simulationConditionId,
                                    experimentalConditionId,
                                    haspreEquilibrationConditionId,
                                    odeSolutions,
                                    odeSolutionsDerivatives,
                                    odePreEqulibriumSolutions,
                                    S,
                                    timeMax,
                                    timeObserved,
                                    iMeasurementsObserved,
                                    iTimeODESolution,
                                    iPerTimePoint,
                                    timePositionInODESolutions,
                                    absTolSS,
                                    relTolSS,
                                    callbacks,
                                    sensealg,
                                    callbackSS)
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
                                       timeObserved::NamedTuple)::NamedTuple

    _timePositionInODESolutions = Vector{UnitRange{Int64}}(undef, length(experimentalConditionId))
    iStart = 1
    for i in eachindex(experimentalConditionId)
        timeObservedCondition = timeObserved[experimentalConditionId[i]]
        _timePositionInODESolutions[i] = iStart:(iStart-1+length(timeObservedCondition))
        iStart = _timePositionInODESolutions[i][end] + 1
    end

    ___timePositionInODESolutions = Tuple(element for element in _timePositionInODESolutions)
    return NamedTuple{Tuple(name for name in experimentalConditionId)}(___timePositionInODESolutions)
end


# For creating terminateSS steady state where we do a Newton step to check if the model is in a steady state
function conditionTerminateSS(u, t, integrator, computeJacobian::Function,
                              absTolSS::Float64, relTolSS::Float64, checkNewton::Bool)

    testval = first(get_tmp_cache(integrator))
    DiffEqBase.get_du!(testval, integrator)

    wrms = sqrt(sum((testval ./ (relTolSS * integrator.u .+ absTolSS)).^2) / length(u))
    if wrms ≤ 1
        checkNewton == false && return true

        J = computeJacobian(dualToFloat.(u), dualToFloat.(integrator.p), t)
        local Δu
        try
            Δu = J \ DualToFloat.(testval)
        catch
            Δu = pinv(J) * dualToFloat.(testval)
        end
        wrmsΔu = sqrt(sum((Δu / (relTolSS * integrator.u .+ absTolSS)).^2) / length(u))
        wrmsΔu ≤ 1 && return true
    end

    return false
end
function affectTerminateSS!(integrator)
    terminate!(integrator)
end
function createSSTerminateSteadyState(odeSystem::ODESystem, absTolSS::Float64, relTolSS::Float64; checkNewton::Bool=false)

    j_func = generate_jacobian(odeSystem)[1] # second is in-place
    computeJacobian = eval(j_func)
    fTerminate = (u, t, integrator) -> conditionTerminateSS(u, t, integrator, computeJacobian, absTolSS, relTolSS, checkNewton)

    return DiscreteCallback(fTerminate, affectTerminateSS!, save_positions=(false, true))
end
