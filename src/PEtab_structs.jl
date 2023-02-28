"""
    PEtabModel

    Struct storing information about a PeTab-model. Create by the `setUpCostFunc` function.

    # Args
    `modelName`: PeTab model name (must match the xml-file name)
    `evalYmod`: Function to evaluate yMod for the log-likelhood.
    `evalU0!`: Function that computes the initial u0 value for the ODE-system.
    `evalSd!`: Function that computes the standard deviation value for the log-likelhood.
    `odeSystem`: ModellingToolkit ODE-system for the PeTab model.
    `paramMap`: A map to correctly map model parameters to the ODE-system.
    `stateMap`: A map to correctly mapping the parameters to the u0 values.
    `paramNames`: Names of the model parameters (both fixed and those to be estimated).
    `stateNames`: Names of the model states.
    `dirModel`: Directory where the model.xml and PeTab files are stored.
    `pathMeasurementData`: Path to the measurementData PeTab file.
    `pathMeasurementData`: Path to the experimentaCondition PeTab file
    `pathMeasurementData`: Path to the observables PeTab file
    `pathMeasurementData`: Path to the parameters PeTab file

    See also: [`setUpCostFunc`]
"""
struct PEtabModel{F1<:Function,
                  F2<:Function,
                  F3<:Function,
                  F4<:Function,
                  F5<:Function,
                  F6<:Function,
                  F7<:Function,
                  F8<:Function,
                  F9<:Function,
                  S<:ODESystem,
                  T1<:Vector{<:Pair{Num, <:Union{AbstractFloat, Num}}},
                  T2<:Vector{<:Pair{Num, <:Union{AbstractFloat, Num}}},
                  T3<:Vector{<:Any},
                  T4<:Vector{<:Any},
                  C<:SciMLBase.DECallback,
                  FA<:Vector{<:Function}}
    modelName::String
    compute_h::F1
    compute_u0!::F2
    compute_u0::F3
    compute_σ::F4
    compute_∂h∂u!::F5
    compute_∂σ∂u!::F6
    compute_∂h∂p!::F7
    compute_∂σ∂p!::F8
    computeTStops::F9
    convertTspan::Bool
    odeSystem::S
    parameterMap::T1
    stateMap::T2
    parameterNames::T3
    stateNames::T4
    dirModel::String
    dirJulia::String
    pathMeasurements::String
    pathConditions::String
    pathObservables::String
    pathParameters::String
    pathSBML::String
    pathYAML::String
    modelCallbackSet::C
    checkIfCallbackIsActive::FA
end


struct PEtabODEProblem{F1<:Function,
                       F2<:Function,
                       F3<:Function,
                       T1<:PEtabModel}

    computeCost::F1
    computeGradient!::F2
    computeHessian!::F3
    costMethod::Symbol
    gradientMethod::Symbol
    hessianMethod::Symbol
    nParametersToEstimate::Int64
    θ_estNames::Vector{Symbol}
    θ_nominal::Vector{Float64}
    θ_nominalT::Vector{Float64}
    lowerBounds::Vector{Float64}
    upperBounds::Vector{Float64}
    pathCube::String
    petabModel::T1
end


struct PEtabODEProblemCache{T1 <: AbstractVector, 
                            T2 <: DiffCache, 
                            T3 <: AbstractVector, 
                            T4 <: AbstractMatrix}
    θ_dynamic::T1
    θ_sd::T1
    θ_observable::T1
    θ_nonDynamic::T1
    θ_dynamicT::T2 # T = transformed vector
    θ_sdT::T2
    θ_observableT::T2
    θ_nonDynamicT::T2
    gradientDyanmicθ::T1
    gradientNotODESystemθ::T1
    jacobianGN::T4
    residualsGN::T1
    _gradient::T1
    _gradientAdjoint::T1
    St0::T4
    ∂h∂u::T3
    ∂σ∂u::T3
    ∂h∂p::T3
    ∂σ∂p::T3
    ∂G∂p::T3
    ∂G∂p_::T3
    ∂G∂u::T3
    dp::T1 
    du::T1
    p::T3 
    u::T3
    S::T4
    odeSolutionValues::T4
end


struct PEtabODESolverCache{T1 <: NamedTuple, 
                           T2 <: NamedTuple}
    pODEProblemCache::T1
    u0Cache::T2
end


"""
    ParameterInfo

    Struct storing the data in the PeTab parameter-file in type-stable manner.

    Currently logScale notices whether or not parameters are estimated on the
    log10 scale or not.

    See also: [`processParameterData`]
"""
struct ParametersInfo
    nominalValue::Vector{Float64}
    lowerBound::Vector{Float64}
    upperBound::Vector{Float64}
    parameterId::Vector{Symbol}
    parameterScale::Vector{Symbol}
    estimate::Vector{Bool}
    nParametersToEstimate::Int64
end


"""
    MeasurementData

    Struct storing the data in the PeTab measurementData-file in type-stable manner.

    See also: [`processMeasurementData`]
"""
struct MeasurementsInfo{T<:Vector{<:Union{<:String, <:AbstractFloat}}}

    measurement::Vector{Float64}
    measurementT::Vector{Float64}
    measurementTransformation::Vector{Symbol}
    time::Vector{Float64}
    observableId::Vector{Symbol}
    preEquilibrationConditionId::Vector{Symbol}
    simulationConditionId::Vector{Symbol}
    noiseParameters::T
    observableParameters::Vector{String}
end


struct SimulationInfo{T1<:NamedTuple,
                      T2<:NamedTuple,
                      T3<:NamedTuple,
                      T4<:NamedTuple,
                      T5<:NamedTuple,
                      T6<:Dict{<:Symbol, <:SciMLBase.DECallback},
                      T7<:Union{<:SciMLSensitivity.AbstractForwardSensitivityAlgorithm, <:SciMLSensitivity.AbstractAdjointSensitivityAlgorithm},
                      T8<:SciMLBase.DECallback, 
                      T9<:Dict{<:Symbol, <:SciMLBase.DECallback}}

    preEquilibrationConditionId::Vector{Symbol}
    simulationConditionId::Vector{Symbol}
    experimentalConditionId::Vector{Symbol}
    haspreEquilibrationConditionId::Bool
    odeSolutions::Dict{Symbol, Union{Nothing, ODESolution}}
    odeSolutionsDerivatives::Dict{Symbol, Union{Nothing, ODESolution}}
    odePreEqulibriumSolutions::Dict{Symbol, Union{Nothing, ODESolution}}
    timeMax::T1
    timeObserved::T2
    iMeasurements::T3
    iTimeODESolution::Vector{Int64}
    iPerTimePoint::T4
    timePositionInODESolutions::T5
    absTolSS::Float64
    relTolSS::Float64
    callbacks::T6
    trackedCallbacks::T9
    sensealg::T7 # sensealg for potential callbacks
    callbackSS::T8
end


struct θObsOrSdParameterMap
    shouldEstimate::Array{Bool, 1}
    indexInθ::Array{Int64, 1}
    constantValues::Vector{Float64}
    nParameters::Int64
    isSingleConstant::Bool
end


struct MapConditionId
    constantParameters::Vector{Float64}
    iODEProblemConstantParameters::Vector{Int64}
    constantsStates::Vector{Float64}
    iODEProblemConstantStates::Vector{Int64}
    iθDynamic::Vector{Int64}
    iODEProblemθDynamic::Vector{Int64}
end


struct MapODEProblem
    iθDynamic::Vector{Int64}
    iODEProblemθDynamic::Vector{Int64}
end


"""
    ParameterIndices

    Struct storing names and mapping indices for mapping the parameter provided
    to the optimizers correctly.

    Optimizers require a single vector input of parameters (pVecEst). However, the PeTab
    model has three kind of parameters, Dynmaic (part of the ODE-system),
    Observable (only part of the observation model) and Standard-deviation
    (only part of the standard deviation expression in the log-likelhood). This
    struct stores mapping indices (starting with i) to map pVecEst
    correctly when computing the likelihood (e.g map the SD-parameters in pVecEst
    correctly to a vector of SD-vals). Also stores the name of each parameter.

    Furthermore, when computing yMod or SD the correct observable and sd parameters
    has to be used for each observation. The mapArrays effectively contains precomputed
    maps allowing said parameter to be effectively be extracted by the getObsOrSdParam
    function.

    See also: [`getIndicesParam`, `ParamMap`]
"""
struct ParameterIndices{T4<:Vector{<:θObsOrSdParameterMap},
                        T5<:MapODEProblem,
                        T6<:NamedTuple,
                        T7<:NamedTuple}

    iθ_dynamic::Vector{Int64}
    iθ_observable::Vector{Int64}
    iθ_sd::Vector{Int64}
    iθ_nonDynamic::Vector{Int64}
    iθ_notOdeSystem::Vector{Int64}
    θ_dynamicNames::Vector{Symbol}
    θ_observableNames::Vector{Symbol}
    θ_sdNames::Vector{Symbol}
    θ_nonDynamicNames::Vector{Symbol}
    θ_notOdeSystemNames::Vector{Symbol}
    θ_estNames::Vector{Symbol}
    θ_scale::T7
    mapθ_observable::T4
    mapθ_sd::T4
    mapODEProblem::T5
    mapsConiditionId::T6
end


struct PriorInfo{T1 <: NamedTuple,
                 T2 <: NamedTuple}
    logpdf::T1
    priorOnParameterScale::T2
    hasPriors::Bool
end


struct PEtabFileError <: Exception
    var::String
end
