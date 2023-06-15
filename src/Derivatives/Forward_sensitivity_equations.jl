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
                                           petabModel::PEtabModel,
                                           sensealg,
                                           odeProblem::ODEProblem,
                                           simulationInfo::SimulationInfo,
                                           θ_indices::ParameterIndices,
                                           measurementInfo ::MeasurementsInfo,
                                           parameterInfo::ParametersInfo,
                                           solveOdeModelAllConditions!::Function,
                                           cfg::Union{ForwardDiff.JacobianConfig, Nothing},
                                           petabODECache::PEtabODEProblemCache;
                                           expIDSolve::Vector{Symbol} = [:all],
                                           splitOverConditions::Bool=false,
                                           isRemade::Bool=false)

    θ_dynamicT = transformθ(θ_dynamic, θ_indices.θ_dynamicNames, θ_indices, :θ_dynamic, petabODECache)
    θ_sdT = transformθ(θ_sd, θ_indices.θ_sdNames, θ_indices, :θ_sd, petabODECache)
    θ_observableT = transformθ(θ_observable, θ_indices.θ_observableNames, θ_indices, :θ_observable, petabODECache)
    θ_nonDynamicT = transformθ(θ_nonDynamic, θ_indices.θ_nonDynamicNames, θ_indices, :θ_nonDynamic, petabODECache)

    # Solve the expanded ODE system for the sensitivites
    success = solveForSensitivites(odeProblem, simulationInfo, θ_indices, petabModel, sensealg, θ_dynamicT,
                                   solveOdeModelAllConditions!, cfg, petabODECache, expIDSolve, splitOverConditions,
                                   isRemade)
    if success != true
        @warn "Failed to solve sensitivity equations"
        gradient .= 1e8
        return
    end
    if isempty(θ_dynamic)
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
        computeGradientForwardExpCond!(gradient, sol, petabODECache, sensealg, θ_dynamicT, θ_sdT, θ_observableT, θ_nonDynamicT,
                                       experimentalConditionId, simulationConditionId, simulationInfo, petabModel,
                                       θ_indices, measurementInfo, parameterInfo)
    end
end


function solveForSensitivites(odeProblem::ODEProblem,
                              simulationInfo::SimulationInfo,
                              θ_indices::ParameterIndices,
                              petabModel::PEtabModel,
                              sensealg::Symbol,
                              θ_dynamic::AbstractVector,
                              solveOdeModelAllConditions!::Function,
                              cfg::ForwardDiff.JacobianConfig,
                              petabODECache::PEtabODEProblemCache,
                              expIDSolve::Vector{Symbol},
                              splitOverConditions::Bool,
                              isRemade::Bool=false)

    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough. 
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final 
    # retcode we cannot catch these cases                               
    simulationInfo.couldSolve[1] = true                              
    petabODECache.S .= 0.0
    if splitOverConditions == false

        # Case where based on the original PEtab file read into Julia we do not have any parameter vectors fixated.
        if isRemade == false || length(petabODECache.gradientDyanmicθ) == petabODECache.nθ_dynamicEst[1]

            # Allow correct mapping to sensitivity matrix
            tmp = petabODECache.nθ_dynamicEst[1]
            petabODECache.nθ_dynamicEst[1] = length(θ_dynamic)

            if isempty(θ_dynamic)
                solveOdeModelAllConditions!(petabODECache.odeSolutionValues, θ_dynamic)
                petabODECache.S .= 0.0
            end

            if !isempty(θ_dynamic)
                ForwardDiff.jacobian!(petabODECache.S, solveOdeModelAllConditions!, petabODECache.odeSolutionValues, θ_dynamic, cfg)
            end

            petabODECache.nθ_dynamicEst[1] = tmp
        end

        # Case when we have dynamic parameters fixed. Here it is not always worth to move accross all chunks
        if !(isRemade == false || length(petabODECache.gradientDyanmicθ) == petabODECache.nθ_dynamicEst[1])
            if petabODECache.nθ_dynamicEst[1] != 0
                C = length(cfg.seeds)
                nForwardPasses = Int64(ceil(petabODECache.nθ_dynamicEst[1] / C))
                __θ_dynamic = θ_dynamic[petabODECache.θ_dynamicInputOrder]
                forwardDiffJacobianChunks(solveOdeModelAllConditions!, petabODECache.odeSolutionValues, petabODECache.S, __θ_dynamic, ForwardDiff.Chunk(C); nForwardPasses=nForwardPasses)
                @views petabODECache.S .= petabODECache.S[:, petabODECache.θ_dynamicOutputOrder]
            end

            if petabODECache.nθ_dynamicEst[1] == 0
                solveOdeModelAllConditions!(petabODECache.odeSolutionValues, θ_dynamic)
                petabODECache.S .= 0.0
            end
        end
    end

    # Slower option, but more efficient if there are several parameters unique to an experimental condition
    if splitOverConditions == true

        petabODECache.S .= 0.0
        Stmp = similar(petabODECache.S)
        for conditionId in simulationInfo.experimentalConditionId
            mapConditionId = θ_indices.mapsConiditionId[conditionId]
            iθ_experimentalCondition = unique(vcat(θ_indices.mapODEProblem.iθDynamic, mapConditionId.iθDynamic))
            θ_input = θ_dynamic[iθ_experimentalCondition]
            computeSensitivityMatrixExpCond! = (odeSolutionValues, θ_arg) ->    begin
                                                                                    _θ_dynamic = convert.(eltype(θ_arg), θ_dynamic)
                                                                                    _θ_dynamic[iθ_experimentalCondition] .= θ_arg
                                                                                    solveOdeModelAllConditions!(odeSolutionValues, _θ_dynamic, [conditionId])
                                                                                end
            @views ForwardDiff.jacobian!(Stmp[:, iθ_experimentalCondition], computeSensitivityMatrixExpCond!, petabODECache.odeSolutionValues, θ_input)
            @views petabODECache.S[:, iθ_experimentalCondition] .+= Stmp[:, iθ_experimentalCondition]
        end
    end

    return simulationInfo.couldSolve[1]
end


function computeGradientForwardExpCond!(gradient::Vector{Float64},
                                        sol::ODESolution,
                                        petabODECache::PEtabODEProblemCache,
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

    # To compute
    compute∂G∂u! = (out, u, p, t, i) -> begin compute∂G∂_(out, u, p, t, i, iPerTimePoint,
                                                         measurementInfo, parameterInfo,
                                                         θ_indices, petabModel,
                                                         θ_sd, θ_observable, θ_nonDynamic,
                                                         petabODECache.∂h∂u, petabODECache.∂σ∂u, compute∂G∂U=true)
                                            end
    compute∂G∂p! = (out, u, p, t, i) -> begin compute∂G∂_(out, u, p, t, i, iPerTimePoint,
                                                         measurementInfo, parameterInfo,
                                                         θ_indices, petabModel,
                                                         θ_sd, θ_observable, θ_nonDynamic,
                                                         petabODECache.∂h∂p, petabODECache.∂σ∂p, compute∂G∂U=false)
                                        end

    # Extract which parameters we compute gradient for in this specific experimental condition
    mapConditionId = θ_indices.mapsConiditionId[simulationConditionId]
    # Unique is needed to account for condition specific parameters which maps to potentially several
    # parameters in ODEProblem.p
    iθ_experimentalCondition = unique(vcat(θ_indices.mapODEProblem.iθDynamic, mapConditionId.iθDynamic))

    # Loop through solution and extract sensitivites
    nModelStates = length(petabModel.stateNames)
    petabODECache.p .= dualToFloat.(sol.prob.p)
    p = petabODECache.p
    u = petabODECache.u
    ∂G∂p, ∂G∂p_, ∂G∂u = petabODECache.∂G∂p, petabODECache.∂G∂p_, petabODECache.∂G∂u
    _gradient = petabODECache._gradient
    fill!(_gradient, 0.0)
    fill!(∂G∂p, 0.0)
    for i in eachindex(timeObserved)
        u .= dualToFloat.((@view sol[:, i]))
        compute∂G∂u!(∂G∂u, u, p, timeObserved[i], i)
        compute∂G∂p!(∂G∂p_, u, p, timeObserved[i], i)
        # We need to extract the correct indices from the big sensitivity matrix (row is observation at specific time
        # point). Overall, positions are precomputed in timePositionInODESolutions
        iStart, iEnd = (timePositionInODESolutions[i]-1)*nModelStates+1, (timePositionInODESolutions[i]-1)*nModelStates + nModelStates
        _S = @view petabODECache.S[iStart:iEnd, iθ_experimentalCondition]
        @views _gradient[iθ_experimentalCondition] .+= transpose(_S)*∂G∂u
        ∂G∂p .+= ∂G∂p_
    end

    # Thus far have have computed dY/dθ, but for parameters on the log-scale we want dY/dθ_log. We can adjust via;
    # dY/dθ_log = log(10) * θ * dY/dθ
    adjustGradientTransformedParameters!(gradient, _gradient, ∂G∂p, θ_dynamic, θ_indices,
                                         simulationConditionId, autoDiffSensitivites=true)
end
