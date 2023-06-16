# Transform parameter from log10 scale to normal scale, or reverse transform
function transformθZygote(θ::AbstractVector,
                          θ_names::Vector{Symbol},
                          parameterInfo::PEtab.ParametersInfo;
                          reverseTransform::Bool=false)::AbstractVector

    iθ = [findfirst(x -> x == θ_names[i], parameterInfo.parameterId) for i in eachindex(θ_names)]
    shouldTransform = [parameterInfo.parameterScale[i] == :log10 ? true : false for i in iθ]
    shouldNotTransform = .!shouldTransform

    if reverseTransform == false
        out = exp10.(θ) .* shouldTransform .+ θ .* shouldNotTransform
    else
        out = log10.(θ) .* shouldTransform .+ θ .* shouldNotTransform
    end
    return out
end


function PEtab.setUpGradient(::Val{:Zygote},
                             odeProblem::ODEProblem,
                             odeSolverOptions::ODESolverOptions,
                             ssSolverOptions::SteadyStateSolverOptions,
                             petabODECache::PEtab.PEtabODEProblemCache,
                             petabODESolverCache::PEtab.PEtabODESolverCache,
                             petabModel::PEtabModel,
                             simulationInfo::PEtab.SimulationInfo,
                             θ_indices::PEtab.ParameterIndices,
                             measurementInfo::PEtab.MeasurementsInfo,
                             parameterInfo::PEtab.ParametersInfo,
                             sensealg::SciMLBase.AbstractSensitivityAlgorithm,
                             priorInfo::PEtab.PriorInfo;
                             chunkSize::Union{Nothing, Int64}=nothing,
                             sensealgSS=nothing,
                             numberOfprocesses::Int64=1,
                             jobs=nothing,
                             results=nothing,
                             splitOverConditions::Bool=false)

    changeExperimentalCondition = (pODEProblem, u0, conditionId, θ_dynamic) -> PEtab._changeExperimentalCondition(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
    _changeODEProblemParameters = (pODEProblem, θ_est) -> PEtab.changeODEProblemParameters(pODEProblem, θ_est, θ_indices, petabModel)
    solveODEExperimentalCondition = (odeProblem, conditionId, θ_dynamic, tMax) -> solveOdeModelAtExperimentalCondZygote(odeProblem, conditionId, θ_dynamic, tMax, changeExperimentalCondition, measurementInfo, simulationInfo, odeSolverOptions.solver, odeSolverOptions.abstol, odeSolverOptions.reltol, ssSolverOptions.abstol, ssSolverOptions.reltol, sensealg, petabModel.computeTStops)
    _computeGradient! = (gradient, θ_est) -> computeGradientZygote(gradient,
                                                                   θ_est,
                                                                   odeProblem,
                                                                   petabModel,
                                                                   simulationInfo,
                                                                   θ_indices,
                                                                   measurementInfo,
                                                                   parameterInfo,
                                                                   _changeODEProblemParameters,
                                                                   solveODEExperimentalCondition,
                                                                   priorInfo,
                                                                   petabODECache)
    
    return _computeGradient!
end


function PEtab.setUpCost(::Val{:Zygote},
                         odeProblem::ODEProblem,
                         odeSolverOptions::ODESolverOptions,
                         ssSolverOptions::SteadyStateSolverOptions,
                         petabODECache::PEtab.PEtabODEProblemCache,
                         petabODESolverCache::PEtab.PEtabODESolverCache,
                         petabModel::PEtabModel,
                         simulationInfo::PEtab.SimulationInfo,
                         θ_indices::PEtab.ParameterIndices,
                         measurementInfo::PEtab.MeasurementsInfo,
                         parameterInfo::PEtab.ParametersInfo,
                         priorInfo::PEtab.PriorInfo,
                         sensealg,
                         numberOfprocesses,
                         jobs,
                         results,
                         computeResiduals)

    changeExperimentalCondition = (pODEProblem, u0, conditionId, θ_dynamic) -> PEtab._changeExperimentalCondition(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
    _changeODEProblemParameters = (pODEProblem, θ_est) -> PEtab.changeODEProblemParameters(pODEProblem, θ_est, θ_indices, petabModel)
    solveODEExperimentalCondition = (odeProblem, conditionId, θ_dynamic, tMax) -> solveOdeModelAtExperimentalCondZygote(odeProblem, conditionId, θ_dynamic, tMax, changeExperimentalCondition, measurementInfo, simulationInfo, odeSolverOptions.solver, odeSolverOptions.abstol, odeSolverOptions.reltol, ssSolverOptions.abstol, ssSolverOptions.reltol, sensealg, petabModel.computeTStops)
    __computeCost = (θ_est) -> computeCostZygote(θ_est,
                                                 odeProblem,
                                                 petabModel,
                                                 simulationInfo,
                                                 θ_indices,
                                                 measurementInfo,
                                                 parameterInfo,
                                                 _changeODEProblemParameters,
                                                 solveODEExperimentalCondition,
                                                 priorInfo)

    return __computeCost
end


function PEtab.setSensealg(sensealg, ::Val{:Zygote})

    if !isnothing(sensealg)
        @assert (typeof(sensealg) <: SciMLSensitivity.AbstractSensitivityAlgorithm) "For Zygote an abstract sensitivity algorithm from SciMLSensitivity must be used"
        return sensealg
    end

    return SciMLSensitivity.ForwardSensitivity()
end