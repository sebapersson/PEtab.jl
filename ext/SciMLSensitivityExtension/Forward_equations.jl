function PEtab.getODEProblemForwardEquations(odeProblem::ODEProblem,
                                       sensealgForwardEquations::SciMLSensitivity.AbstractForwardSensitivityAlgorithm)::ODEProblem
    return ODEForwardSensitivityProblem(odeProblem.f, odeProblem.u0, odeProblem.tspan, odeProblem.p, sensealg=sensealgForwardEquations)
end


function PEtab.solveForSensitivites(odeProblem::ODEProblem,
                                    simulationInfo::PEtab.SimulationInfo,
                                    θ_indices::PEtab.ParameterIndices,
                                    petabModel::PEtabModel,
                                    sensealg::SciMLSensitivity.AbstractForwardSensitivityAlgorithm,
                                    θ_dynamic::AbstractVector,
                                    solveOdeModelAllConditions!::Function,
                                    cfg::Nothing,
                                    petabODECache::PEtab.PEtabODEProblemCache,
                                    expIDSolve::Vector{Symbol},
                                    splitOverConditions::Bool,
                                    isRemade::Bool=false)

    nModelStates = length(petabModel.stateNames)
    _odeProblem = remake(odeProblem, p = convert.(eltype(θ_dynamic), odeProblem.p), u0 = convert.(eltype(θ_dynamic), odeProblem.u0))
    PEtab.changeODEProblemParameters!(_odeProblem.p, (@view _odeProblem.u0[1:nModelStates]), θ_dynamic, θ_indices, petabModel)
    success = solveOdeModelAllConditions!(simulationInfo.odeSolutionsDerivatives, _odeProblem, θ_dynamic, expIDSolve)
    return success
end


function PEtab.computeGradientForwardExpCond!(gradient::Vector{Float64},
                                              sol::ODESolution,
                                              petabODECache::PEtab.PEtabODEProblemCache,
                                              sensealg::SciMLSensitivity.AbstractForwardSensitivityAlgorithm,
                                              θ_dynamic::Vector{Float64},
                                              θ_sd::Vector{Float64},
                                              θ_observable::Vector{Float64},
                                              θ_nonDynamic::Vector{Float64},
                                              experimentalConditionId::Symbol,
                                              simulationConditionId::Symbol,
                                              simulationInfo::PEtab.SimulationInfo,
                                              petabModel::PEtabModel,
                                              θ_indices::PEtab.ParameterIndices,
                                              measurementInfo::PEtab.MeasurementsInfo,
                                              parameterInfo::PEtab.ParametersInfo)

    iPerTimePoint = simulationInfo.iPerTimePoint[experimentalConditionId]
    timeObserved = simulationInfo.timeObserved[experimentalConditionId]

    # To compute
    compute∂G∂u = (out, u, p, t, i) -> begin PEtab.compute∂G∂_(out, u, p, t, i, iPerTimePoint,
                                                               measurementInfo, parameterInfo,
                                                               θ_indices, petabModel,
                                                               θ_sd, θ_observable, θ_nonDynamic,
                                                               petabODECache.∂h∂u, petabODECache.∂σ∂u, compute∂G∂U=true)
                                            end
    compute∂G∂p = (out, u, p, t, i) -> begin PEtab.compute∂G∂_(out, u, p, t, i, iPerTimePoint,
                                                               measurementInfo, parameterInfo,
                                                               θ_indices, petabModel,
                                                               θ_sd, θ_observable, θ_nonDynamic,
                                                               petabODECache.∂h∂p, petabODECache.∂σ∂p, compute∂G∂U=false)
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
        _gradient .+= transpose(_S)*∂G∂u
        ∂G∂p .+= ∂G∂p_
    end

    # Thus far have have computed dY/dθ, but for parameters on the log-scale we want dY/dθ_log. We can adjust via;
    # dY/dθ_log = log(10) * θ * dY/dθ
    PEtab.adjustGradientTransformedParameters!(gradient, _gradient, ∂G∂p, θ_dynamic, θ_indices,
                                               simulationConditionId, autoDiffSensitivites=false)
end