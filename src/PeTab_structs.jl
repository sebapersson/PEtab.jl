"""
    PeTabModel

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
struct PeTabModel{T1<:Vector{<:Pair{Num, <:Union{AbstractFloat, Num}}},
                  T2<:Vector{<:Pair{Num, <:Union{AbstractFloat, Num}}},
                  T3<:Vector{<:Sym},
                  T4<:Vector{<:Term}}
    modelName::String
    evalYmod::Function
    evalU0!::Function
    evalU0::Function
    evalSd!::Function
    odeSystem::ODESystem
    paramMap::T1
    stateMap::T2
    paramNames::T3
    stateNames::T4
    dirModel::String
    pathMeasurementData::String
    pathExperimentalConditions::String
    pathObservables::String
    pathParameters::String
end


struct PeTabOpt{T1 <: Integer,
                T2 <: AbstractVector{<:Number}}
    evalF::Function
    evalFZygote::Function
    evalGradF::Function
    evalGradFZygote::Function
    evalHess::Function
    evalHessApprox::Function
    nParamEst::T1
    namesParam::AbstractVector{<:String}
    paramVecNotTransformed::T2
    paramVecTransformed::T2
    lowerBounds::T2
    upperBounds::T2
    pathCube::String
    peTabModel::PeTabModel
end


"""
    ParamData

Struct storing the data in the PeTab parameter-file in type-stable manner.

Currently logScale notices whether or not parameters are estimated on the
log10 scale or not.

See also: [`processParameterData`]
"""
struct ParamData{T1<:AbstractArray{<:Number},
                 T2<:AbstractVector{<:String},
                 T3<:AbstractVector{<:Bool},
                 T4<:Signed}

    # TODO: logScale make symbol to support more transformations
    paramVal::T1
    lowerBounds::T1
    upperBounds::T1
    parameterID::T2
    logScale::T3
    shouldEst::T3
    nParamEst::T4
end


"""
    MeasurementData

Struct storing the data in the PeTab measurementData-file in type-stable manner.

Transform data supports log and log10 transformations of the data.

See also: [`processMeasurementData`]
"""
struct MeasurementData{T1<:AbstractVector{<:Number},
                       T2<:AbstractVector{<:String},
                       T3<:AbstractVector{<:Symbol},
                       T4<:Dict,
                       T5<:AbstractVector{<:Integer},
                       T6<:AbstractVector{<:Union{<:String, <:AbstractFloat}}}

    yObsNotTransformed::T1
    yObsTransformed::T1
    tObs::T1
    observableID::T2
    conditionId::T2  # Sum of pre-eq + simulation-cond id
    sdParams::T6
    transformData::T3 # Only done once
    obsParam::T2
    tVecSave::T4
    iTObs::T5
    iPerConditionId::T4
    preEqCond::T2
    simCond::T2
end


"""
    SimulationInfo

Struct storing simulation (forward ODE-solution) information. Specifcially
stores the experimental ID:s from the experimentalCondition - PeTab file;
firstExpIds (preequilibration ID:s), the shiftExpIds (postequilibration), and
simulateSS (whether or not to simulate ODE-model to steady state). Further
stores a solArray with the ODE solution where conditionIdSol of the ID for
each forward solution. It also stores for each experimental condition which
time-points we have observed data at

See also: [`getSimulationInfo`]
"""
struct SimulationInfo{T1<:AbstractVector{<:String},
                      T2<:AbstractVector{<:AbstractVector{<:String}},
                      T3<:Bool,
                      T4<:AbstractVector{<:SciMLBase.AbstractODESolution},
                      T5<:Dict,
                      T6<:Array{<:AbstractFloat, 1}}
    firstExpIds::T1
    shiftExpIds::T2
    conditionIdSol::T1
    tMaxForwardSim::T6
    simulateSS::T3
    solArray::T4
    solArrayGrad::T4
    absTolSS::Float64
    relTolSS::Float64
    tVecSave::T5
end


"""
    ParamMap

Struct which makes out a map to correctly for an observation extract the correct observable
or sd-param via the getObsOrSdParam function when computing the likelihood. Correctly built
by `buildMapParameters`, and is part of the ParameterIndices-struct.

For noise or observable parameters belong to an observation, e.g (obsParam1, obsParam2),
this struct stores which parameters should be estimtated, and for those parameters which
index they correspond to in the parameter estimation vector. For constant parameters
the struct stores the values.

See also: [`getIndicesParam`, `buildMapParameters`]
"""
struct ParamMap{T1<:AbstractVector{<:Number}}
    shouldEst::AbstractVector{<:Bool}
    indexUse::AbstractVector{UInt32}
    valuesConst::T1
    nParam::UInt32
end


struct MapExpCond{T1 <: Vector{<:AbstractFloat},
                  T2 <: Vector{<:Integer}}
    condID::String
    expCondParamConstVal::T1
    iOdeProbParamConstVal::T2
    expCondStateConstVal::T1
    iOdeProbStateConstVal::T2
    iDynEstVec::T2
    iOdeProbDynParam::T2
end


struct MapDynParEst{T1 <: Vector{<:Integer}}
    iDynParamInSys::T1
    iDynParamInVecEst::T1
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
struct ParameterIndices{T1<:AbstractVector{<:Integer},
                        T2<:AbstractVector{<:String},
                        T3<:AbstractVector{UInt32},
                        T4<:AbstractVector{<:ParamMap},
                        T5<:MapDynParEst,
                        T6<:AbstractVector{<:MapExpCond},
                        T7<:AbstractVector{<:AbstractVector{<:AbstractFloat}}}

    iDynParam::T1
    iObsParam::T1
    iSdParam::T1
    iSdObsNonDynPar::T1
    iNonDynParam::T1
    namesDynParam::T2
    namesObsParam::T2
    namesSdParam::T2
    namesSdObsNonDynPar::T2
    namesNonDynParam::T2
    namesParamEst::T2
    indexObsParamMap::T3
    indexSdParamMap::T3
    mapArrayObsParam::T4
    mapArraySdParam::T4
    mapDynParEst::T5
    mapExpCond::T6
    constParamPerCond::T7
end


struct PriorInfo
    logpdf::Vector{Function}
    priorOnParamScale::Vector{Bool}
    hasPriors::Bool
end
