"""
    PEtabModel

A Julia-compatible representation of a PEtab-specified problem.

Created by `readPEtabModel`, this object contains helper functions for setting up cost, gradient, and Hessian computations, as well as handling potential model events (callbacks). 

**Note1:** Several of the functions in `PEtabModel` are not intended to be accessed by the user. For example, `compute_h` (and similar functions) require indices that are built in the background to efficiently map parameters between experimental (simulation) conditions. Rather, `PEtabModel` holds all information needed to create a `PEtabODEProblem`, and in the future, `PEtabSDEProblem`, etc.

**Note2:** `ODEProblem.p` refers to the parameters for the underlying `DifferentialEquations.jl` `ODEProblem`.

# Fields
- `modelName`: The model name extracted from the PEtab YAML file.
- `compute_h`: Computes the observable `h` for a specific time point and simulation condition.
- `compute_u0!`: Computes in-place initial values using `ODEProblem.p` for a simulation condition; `compute_u0!(u0, p)`.
- `compute_u0`: Computes initial values as above, but not in-place; `u0 = compute_u0(p)`.
- `compute_σ`: Computes the noise parameter `σ` for a specific time point and simulation condition.
- `compute_∂h∂u!`: Computes the gradient of `h` with respect to `ODEModel` states (`u`) for a specific time point and simulation condition.
- `compute_∂σ∂u!`: Computes the gradient of `σ` with respect to `ODEModel` states (`u`) for a specific time point and simulation condition.
- `compute_∂h∂p!`: Computes the gradient of `h` with respect to `ODEProblem.p`.
- `compute_∂σ∂p!`: Computes the gradient of `σ` with respect to `ODEProblem.p`.
- `computeTStops`: Computes the event times in case the model has `DiscreteCallbacks` (events).
- `convertTspan::Bool`: Tracks whether the time span should be converted to `Dual` numbers for `ForwardDiff.jl` gradients, in case the model has `DiscreteCallbacks` and the trigger time is a parameter set to be estimated.
- `dirModel`: The directory where the model.xml and PEtab files are stored.
- `dirJulia`: The directory where the Julia-model files created by parsing the PEtab files (e.g., SBML file) are stored. 
- `odeSystem`: A `ModellingToolkit.jl` ODE system obtained from parsing the model SBML file.
- `parameterMap`: A `ModellingToolkit.jl` parameter map for the ODE system.
- `stateMap`: A `ModellingToolkit.jl` state map for the ODE system describing how the initial values are computed, e.g., whether or not certain initial values are computed from parameters in the `parameterMap`.
- `parameterNames`: The names of the parameters in the `odeSystem`.
- `stateNames`: The names of the states in the `odeSystem`.
- `pathMeasurements`: The path to the PEtab measurements file.
- `pathConditions`: The path to the PEtab conditions file.
- `pathObservables`: The path to the PEtab observables file.
- `pathParameters`: The path to the PEtab parameters file.
- `pathSBML`: The path to the PEtab SBML file.
- `pathYAML`: The path to the PEtab YAML file.
- `modelCallbackSet`: This stores potential model callbacks or events.
- `checkIfCallbackIsActive`: Piecewise SBML statements are transformed to DiscreteCallbacks that are activated at a specific time-point. The piecewise callback has a default value at t0 and is only triggered when reaching t_activation. If t_activation ≤ 0 (never reached when solving the model), this function checks whether the callback should be triggered before solving the model.
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


mutable struct ODESolverOptions{T2 <: Union{Float64, Nothing}}
    solver
    abstol::Float64
    reltol::Float64
    force_dtmin::Bool
    dtmin::T2
    maxiters::Int64    
end


struct SteadyStateSolverOptions{T1 <: Union{Nothing, NonlinearSolve.AbstractNonlinearSolveAlgorithm}, 
                                T2 <: Union{Nothing, AbstractFloat},
                                T3 <: Union{Nothing, NonlinearProblem}, 
                                CA <: Union{Nothing, SciMLBase.DECallback},
                                T4 <: Union{Nothing, Integer}}
    method::Symbol
    rootfindingAlgorithm::T1
    howCheckSimulationReachedSteadyState::Symbol
    abstol::T2
    reltol::T2
    maxiters::T4
    callbackSS::CA
    nonlinearSolveProblem::T3
end


struct SimulationInfo{T1<:NamedTuple,
                      T2<:NamedTuple,
                      T3<:NamedTuple,
                      T4<:NamedTuple,
                      T5<:NamedTuple,
                      T6<:Dict{<:Symbol, <:SciMLBase.DECallback},
                      T7<:Union{<:SciMLSensitivity.AbstractForwardSensitivityAlgorithm, <:SciMLSensitivity.AbstractAdjointSensitivityAlgorithm},
                      T9<:Dict{<:Symbol, <:SciMLBase.DECallback}}

    preEquilibrationConditionId::Vector{Symbol}
    simulationConditionId::Vector{Symbol}
    experimentalConditionId::Vector{Symbol}
    haspreEquilibrationConditionId::Bool
    odeSolutions::Dict{Symbol, Union{Nothing, ODESolution}}
    odeSolutionsDerivatives::Dict{Symbol, Union{Nothing, ODESolution}}
    odePreEqulibriumSolutions::Dict{Symbol, Union{Nothing, ODESolution, SciMLBase.NonlinearSolution}}
    timeMax::T1
    timeObserved::T2
    iMeasurements::T3
    iTimeODESolution::Vector{Int64}
    iPerTimePoint::T4
    timePositionInODESolutions::T5
    callbacks::T6
    trackedCallbacks::T9
    sensealg::T7 # sensealg for potential callbacks
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


"""
PEtabODEProblem

Everything needed to setup an optimization problem (compute cost, gradient, hessian and parameter bounds) for a PEtab model.

!!! note 
    The parameter vector θ is always assumed to be on the parameter scale specified in the PEtab parameters file. If needed, θ is transformed to the linear scale inside the function call.

# Fields
- `computeCost`: For θ computes the negative likelihood (objective to minimize)
- `computeChi2`: For θ compute χ2 value
- `computeGradient!`: For θ computes in-place gradient computeGradient!(gradient, θ)
- `computeGradient`: For θ computes out-place gradient gradient = computeGradient(θ)
- `computeHessian!`: For θ computes in-place hessian-(approximation) computeHessian!(hessian, θ)
- `computeHessian`: For θ computes out-place hessian-(approximation) hessian = computeHessian(θ)
- `computeSimulatedValues`: For θ compute the corresponding model (simulated) values to the measurements in the same order as in the Measurements PEtab table
- `computeResiduals`: For θ compute the residuals (h_model - h_observed)^2 / σ^2 in the same order as in the Measurements PEtab table
- `gradientMethod`: The method used to compute the gradient (either :ForwardDiff, :ForwardEquations, :Adjoint, or :Zygote).
- `hessianMethod`: The method used to compute or approximate the Hessian (either :ForwardDiff, :BlocForwardDiff, or :GaussNewton).
- `nParametersToEstimate`: The number of parameters to estimate.
- `θ_estNames`: The names of the parameters in θ.
- `θ_nominal`: The nominal values of θ as specified in the PEtab parameters file.
- `θ_nominalT`: The nominal values of θ on the parameter scale (e.g., log) as specified in the PEtab parameters file.
- `lowerBounds`: The lower parameter bounds on the parameter scale for θ as specified in the PEtab parameters file.
- `upperBounds`: The upper parameter bounds on the parameter scale for θ as specified in the PEtab parameters file.
- `petabModel`: The PEtabModel used to construct the PEtabODEProblem.
- `odeSolverOptions`: The options for the ODE solver specified when creating the PEtabODEProblem.
- `odeSolverGradientOptions`: The options for the ODE solver gradient specified when creating the PEtabODEProblem.
"""
struct PEtabODEProblem{F1<:Function,
                       F2<:Function,
                       F3<:Function,
                       F4<:Function,
                       F5<:Union{Function, Nothing},
                       F6<:Union{Function, Nothing}, 
                       F7<:Function,
                       F8<:Function,
                       T1<:PEtabModel, 
                       T2<:ODESolverOptions, 
                       T3<:ODESolverOptions, 
                       T4<:SteadyStateSolverOptions, 
                       T5<:SteadyStateSolverOptions, 
                       T6<:ParameterIndices, 
                       T7<:SimulationInfo}

    computeCost::F1
    computeChi2::F2
    computeGradient!::F3
    computeGradient::F4
    computeHessian!::F5
    computeHessian::F6
    computeSimulatedValues::F7
    computeResiduals::F8
    costMethod::Symbol
    gradientMethod::Symbol
    hessianMethod::Union{Symbol, Nothing}
    nParametersToEstimate::Int64
    θ_estNames::Vector{Symbol}
    θ_nominal::Vector{Float64}
    θ_nominalT::Vector{Float64}
    lowerBounds::Vector{Float64}
    upperBounds::Vector{Float64}
    pathCube::String
    petabModel::T1
    odeSolverOptions::T2
    odeSolverGradientOptions::T3
    ssSolverOptions::T4
    ssSolverGradientOptions::T5
    θ_indices::T6
    simulationInfo::T7
    splitOverConditions::Bool
end


struct ParametersInfo
    nominalValue::Vector{Float64}
    lowerBound::Vector{Float64}
    upperBound::Vector{Float64}
    parameterId::Vector{Symbol}
    parameterScale::Vector{Symbol}
    estimate::Vector{Bool}
    nParametersToEstimate::Int64
end


struct MeasurementsInfo{T<:Vector{<:Union{<:String, <:AbstractFloat}}}

    measurement::Vector{Float64}
    measurementT::Vector{Float64}
    simulatedValues::Vector{Float64}
    chi2Values::Vector{Float64}
    residuals::Vector{Float64}
    measurementTransformation::Vector{Symbol}
    time::Vector{Float64}
    observableId::Vector{Symbol}
    preEquilibrationConditionId::Vector{Symbol}
    simulationConditionId::Vector{Symbol}
    noiseParameters::T
    observableParameters::Vector{String}
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
    θ_dynamicInputOrder::Vector{Int64}
    θ_dynamicOutputOrder::Vector{Int64}
    nθ_dynamicEst::Vector{Int64}
end


struct PEtabODESolverCache{T1 <: NamedTuple, 
                           T2 <: NamedTuple}
    pODEProblemCache::T1
    u0Cache::T2
end


struct PriorInfo{T1 <: NamedTuple,
                 T2 <: NamedTuple}
    logpdf::T1
    priorOnParameterScale::T2
    hasPriors::Bool
end


"""
    Fides

[Fides](https://github.com/fides-dev/fides) is a Python Newton-trust region optimizer for box-bounded optimization problems.

It is particularly effective when the full Hessian cannot be computed, but the Gauss-Newton Hessian approximation can be 
computed. If constructed with `Fides(verbose=true)`, it prints optimization progress during the process.
"""
struct Fides
    hessianApproximation
    verbose
end
function Fides(; verbose::Bool=false)
    verboseArg = verbose == true ? 1 : 0
    return Fides(nothing, verboseArg)
end


struct PEtabFileError <: Exception
    var::String
end
