#=
    Functions specific to gradient compuations via forward sensitivity equations. Notice that we can solve the 
    forward system either via i) solving the expanded ODE-system or ii) by using AutoDiff to obtain the sensitivites, 
    which efficiently are the jacobian of the ODESolution.

    There are two cases. When we compute the Jacobian of the ODE-solution via autodiff (sensealg=:AutoDiff) we compute 
    a big Jacobian matrix (sensitivity matrix) across all experimental condition, while using one of the Julia forward 
    algorithms we compute a "small" Jacobian for each experimental condition.
=#


function computeGradientForwardEqDynamicθ!(gradient::Vector{Float64},
                                           θ_dynamic::Vector{Float64},
                                           θ_sd::Vector{Float64},
                                           θ_observable::Vector{Float64},
                                           θ_nonDynamic::Vector{Float64},
                                           S::Matrix{Float64},
                                           petabModel::PEtabModel,
                                           sensealg::Union{Symbol, SciMLSensitivity.AbstractForwardSensitivityAlgorithm},
                                           odeProblem::ODEProblem,
                                           simulationInfo::SimulationInfo,
                                           θ_indices::ParameterIndices,
                                           measurementInfo ::MeasurementsInfo, 
                                           parameterInfo::ParametersInfo, 
                                           changeODEProblemParameters!::Function,
                                           solveOdeModelAllConditions!::Function;
                                           expIDSolve::Vector{Symbol} = [:all])

    θ_dynamicT = transformθ(θ_dynamic, θ_indices.θ_dynamicNames, θ_indices)
    θ_sdT = transformθ(θ_sd, θ_indices.θ_sdNames, θ_indices)
    θ_observableT = transformθ(θ_observable, θ_indices.θ_observableNames, θ_indices)
    θ_nonDynamicT = transformθ(θ_nonDynamic, θ_indices.θ_nonDynamicNames, θ_indices)

    # Solve the expanded ODE system for the sensitivites
    success = solveForSensitivites(S, odeProblem, simulationInfo, petabModel, sensealg, θ_dynamicT, 
                                   solveOdeModelAllConditions!, changeODEProblemParameters!, expIDSolve)
    if success != true
        println("Failed to solve sensitivity equations")
        gradient .= 1e8
        return
    end

    gradient .= 0.0
    for i in eachindex(simulationInfo.experimentalConditionId)
        experimentalConditionId = simulationInfo.experimentalConditionId[i]
        simulationConditionId = simulationInfo.simulationConditionId[i]
        
        if expIDSolve[1] != :all && experimentalConditionId ∉ expIDSolve
            continue
        end

        sol = simulationInfo.odeSolutionsDerivatives[experimentalConditionId]
    
        # If we have a callback it needs to be properly handled 
        computeGradientForwardExpCond!(gradient, sol, S, sensealg, θ_dynamicT, θ_sdT, θ_observableT, θ_nonDynamicT,
                                       experimentalConditionId, simulationConditionId, simulationInfo, petabModel, 
                                       θ_indices, measurementInfo, parameterInfo)
    end
end


function solveForSensitivites(S::Matrix{Float64},
                              odeProblem::ODEProblem, 
                              simulationInfo::SimulationInfo,
                              petabModel::PEtabModel,
                              sensealg::SciMLSensitivity.AbstractForwardSensitivityAlgorithm,
                              θ_dynamic::AbstractVector, 
                              solveOdeModelAllConditions!::Function,
                              changeODEProblemParameters!::Function,
                              expIDSolve::Vector{Symbol})

    nModelStates = length(petabModel.stateNames)
    _odeProblem = remake(odeProblem, p = convert.(eltype(θ_dynamic), odeProblem.p), u0 = convert.(eltype(θ_dynamic), odeProblem.u0))
    changeODEProblemParameters!(_odeProblem.p, (@view _odeProblem.u0[1:nModelStates]), θ_dynamic)
    success = solveOdeModelAllConditions!(simulationInfo.odeSolutionsDerivatives, _odeProblem, θ_dynamic, expIDSolve)
    return success
end
function solveForSensitivites(S::Matrix{Float64},
                              odeProblem::ODEProblem, 
                              simulationInfo::SimulationInfo,
                              petabModel::PEtabModel,
                              sensealg::Symbol,
                              θ_dynamic::AbstractVector, 
                              solveOdeModelAllConditions!::Function,
                              changeODEProblemParameters!::Function,
                              expIDSolve::Vector{Symbol})

    _odeProblem = remake(odeProblem, p = convert.(eltype(θ_dynamic), odeProblem.p), u0 = convert.(eltype(θ_dynamic), odeProblem.u0))
    success = solveOdeModelAllConditions!(simulationInfo.odeSolutionsDerivatives, S, _odeProblem, θ_dynamic, expIDSolve)

    return success
end


function computeGradientForwardExpCond!(gradient::Vector{Float64},
                                        sol::ODESolution,
                                        S::Matrix{Float64},
                                        sensealg::SciMLSensitivity.AbstractForwardSensitivityAlgorithm,
                                        θ_dynamic::Vector{Float64},
                                        θ_sd::Vector{Float64}, 
                                        θ_observable::Vector{Float64}, 
                                        θ_nonDynamic::Vector{Float64},
                                        experimentalConditionId::Symbol,
                                        simulationConditionId::Symbol,
                                        simulationInfo::SimulationInfo,
                                        petabModel::PEtabModel,
                                        θ_indices::ParameterIndices,
                                        measurementInfo::MeasurementsInfo, 
                                        parameterInfo::ParametersInfo)

    iPerTimePoint = simulationInfo.iPerTimePoint[experimentalConditionId]                                        
    timeObserved = simulationInfo.timeObserved[experimentalConditionId]      

    # Pre allcoate vectors needed for computations 
    ∂h∂u, ∂σ∂u, ∂h∂p, ∂σ∂p = allocateObservableFunctionDerivatives(sol, petabModel) 
    
    # To compute 
    compute∂G∂u = (out, u, p, t, i) -> begin compute∂G∂_(out, u, p, t, i, iPerTimePoint, 
                                                         measurementInfo, parameterInfo, 
                                                         θ_indices, petabModel, 
                                                         θ_dynamic, θ_sd, θ_observable, θ_nonDynamic, 
                                                         ∂h∂u, ∂σ∂u, compute∂G∂U=true)
                                            end
    compute∂G∂p = (out, u, p, t, i) -> begin compute∂G∂_(out, u, p, t, i, iPerTimePoint, 
                                                         measurementInfo, parameterInfo, 
                                                         θ_indices, petabModel, 
                                                         θ_dynamic, θ_sd, θ_observable, θ_nonDynamic, 
                                                         ∂h∂p, ∂σ∂p, compute∂G∂U=false)
                                        end
      
    # Loop through solution and extract sensitivites                                                 
    p = sol.prob.p
    ∂G∂p, ∂G∂p_ = zeros(Float64, length(p)), zeros(Float64, length(p)) 
    ∂G∂u = zeros(Float64, length(petabModel.stateNames))
    _gradient = zeros(Float64, length(p))
    for i in eachindex(timeObserved)     
        u, _S = extract_local_sensitivities(sol, i, true)
        compute∂G∂u(∂G∂u, u, p, timeObserved[i], i)
        compute∂G∂p(∂G∂p_, u, p, timeObserved[i], i)
        _gradient .+= _S'*∂G∂u 
        ∂G∂p .+= ∂G∂p_
    end

    # Thus far have have computed dY/dθ, but for parameters on the log-scale we want dY/dθ_log. We can adjust via;
    # dY/dθ_log = log(10) * θ * dY/dθ
    adjustGradientTransformedParameters!(gradient, _gradient, ∂G∂p, θ_dynamic, θ_indices, 
                                         simulationConditionId, autoDiffSensitivites=false)
end
function computeGradientForwardExpCond!(gradient::Vector{Float64},
                                        sol::ODESolution,
                                        S::Matrix{Float64},
                                        sensealg::Symbol,
                                        θ_dynamic::Vector{Float64},
                                        θ_sd::Vector{Float64}, 
                                        θ_observable::Vector{Float64}, 
                                        θ_nonDynamic::Vector{Float64},
                                        experimentalConditionId::Symbol,
                                        simulationConditionId::Symbol,
                                        simulationInfo::SimulationInfo,
                                        petabModel::PEtabModel,
                                        θ_indices::ParameterIndices,
                                        measurementInfo::MeasurementsInfo, 
                                        parameterInfo::ParametersInfo)

    iPerTimePoint = simulationInfo.iPerTimePoint[experimentalConditionId]                                        
    timeObserved = simulationInfo.timeObserved[experimentalConditionId]      
    timePositionInODESolutions = simulationInfo.timePositionInODESolutions[experimentalConditionId]                                                                          
    
    # Pre allcoate vectors needed for computations 
    ∂h∂u, ∂σ∂u, ∂h∂p, ∂σ∂p = allocateObservableFunctionDerivatives(sol, petabModel) 
    
    # To compute 
    compute∂G∂u = (out, u, p, t, i) -> begin compute∂G∂_(out, u, p, t, i, iPerTimePoint, 
                                                         measurementInfo, parameterInfo, 
                                                         θ_indices, petabModel, 
                                                         θ_dynamic, θ_sd, θ_observable, θ_nonDynamic, 
                                                         ∂h∂u, ∂σ∂u, compute∂G∂U=true)
                                            end
    compute∂G∂p = (out, u, p, t, i) -> begin compute∂G∂_(out, u, p, t, i, iPerTimePoint, 
                                                         measurementInfo, parameterInfo, 
                                                         θ_indices, petabModel, 
                                                         θ_dynamic, θ_sd, θ_observable, θ_nonDynamic, 
                                                         ∂h∂p, ∂σ∂p, compute∂G∂U=false)
                                        end
      
    # Extract which parameters we compute gradient for in this specific experimental condition 
    mapConditionId = θ_indices.mapsConiditionId[simulationConditionId]  
    # Unique is needed to account for condition specific parameters which maps to potentially several 
    # parameters in ODEProblem.p                               
    iθ_experimentalCondition = unique(vcat(θ_indices.mapODEProblem.iθDynamic, mapConditionId.iθDynamic))
    
    # Loop through solution and extract sensitivites                
    p = dualToFloat.(sol.prob.p)
    nModelStates = length(petabModel.stateNames)
    ∂G∂p, ∂G∂p_ = zeros(Float64, length(p)), zeros(Float64, length(p)) 
    ∂G∂u = zeros(Float64, nModelStates)
    _gradient = zeros(Float64, length(θ_indices.iθ_dynamic))
    for i in eachindex(timeObserved)     
        u = dualToFloat.(sol[:, i])
        compute∂G∂u(∂G∂u, u, p, timeObserved[i], i)
        compute∂G∂p(∂G∂p_, u, p, timeObserved[i], i)
        # We need to extract the correct indices from the big sensitivity matrix (row is observation at specific time
        # point). Overall, positions are precomputed in timePositionInODESolutions
        iStart, iEnd = (timePositionInODESolutions[i]-1)*nModelStates+1, (timePositionInODESolutions[i]-1)*nModelStates + nModelStates
        _S = @view S[iStart:iEnd, iθ_experimentalCondition]
        @views _gradient[iθ_experimentalCondition] .+= _S'*∂G∂u 
        ∂G∂p .+= ∂G∂p_
    end

    # Thus far have have computed dY/dθ, but for parameters on the log-scale we want dY/dθ_log. We can adjust via;
    # dY/dθ_log = log(10) * θ * dY/dθ
    adjustGradientTransformedParameters!(gradient, _gradient, ∂G∂p, θ_dynamic, θ_indices, 
                                         simulationConditionId, autoDiffSensitivites=true)
end
