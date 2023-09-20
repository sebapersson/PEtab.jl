# Transform parameter from log10 scale to normal scale, or reverse transform
function transformθZygote(θ::AbstractVector,
                          n_parameters_estimate::Vector{Symbol},
                          parameterInfo::PEtab.ParametersInfo;
                          reverseTransform::Bool=false)::AbstractVector

    iθ = [findfirst(x -> x == n_parameters_estimate[i], parameterInfo.parameterId) for i in eachindex(n_parameters_estimate)]
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
                             ode_solver::ODESolver,
                             ss_solver::SteadyStateSolver,
                             petabODECache::PEtab.PEtabODEProblemCache,
                             petabODESolverCache::PEtab.PEtabODESolverCache,
                             petab_model::PEtabModel,
                             simulation_info::PEtab.SimulationInfo,
                             θ_indices::PEtab.ParameterIndices,
                             measurementInfo::PEtab.MeasurementsInfo,
                             parameterInfo::PEtab.ParametersInfo,
                             sensealg::SciMLBase.AbstractSensitivityAlgorithm,
                             priorInfo::PEtab.PriorInfo;
                             chunksize::Union{Nothing, Int64}=nothing,
                             sensealg_ss=nothing,
                             n_processes::Int64=1,
                             jobs=nothing,
                             results=nothing,
                             split_over_conditions::Bool=false)

    changeExperimentalCondition = (pODEProblem, u0, conditionId, θ_dynamic) -> PEtab._changeExperimentalCondition(pODEProblem, u0, conditionId, θ_dynamic, petab_model, θ_indices)
    solveODEExperimentalCondition = (odeProblem, conditionId, θ_dynamic, tMax) -> solveOdeModelAtExperimentalCondZygote(odeProblem, conditionId, θ_dynamic, tMax, changeExperimentalCondition, measurementInfo, simulation_info, ode_solver.solver, ode_solver.abstol, ode_solver.reltol, ss_solver.abstol, ss_solver.reltol, sensealg, petab_model.compute_tstops)
    _compute_gradient! = (gradient, θ_est) -> compute_gradientZygote(gradient,
                                                                   θ_est,
                                                                   odeProblem,
                                                                   petab_model,
                                                                   simulation_info,
                                                                   θ_indices,
                                                                   measurementInfo,
                                                                   parameterInfo,
                                                                   solveODEExperimentalCondition,
                                                                   priorInfo,
                                                                   petabODECache)
    
    return _compute_gradient!
end


function PEtab.setUpCost(::Val{:Zygote},
                         odeProblem::ODEProblem,
                         ode_solver::ODESolver,
                         ss_solver::SteadyStateSolver,
                         petabODECache::PEtab.PEtabODEProblemCache,
                         petabODESolverCache::PEtab.PEtabODESolverCache,
                         petab_model::PEtabModel,
                         simulation_info::PEtab.SimulationInfo,
                         θ_indices::PEtab.ParameterIndices,
                         measurementInfo::PEtab.MeasurementsInfo,
                         parameterInfo::PEtab.ParametersInfo,
                         priorInfo::PEtab.PriorInfo,
                         sensealg,
                         n_processes,
                         jobs,
                         results,
                         compute_residuals)

    changeExperimentalCondition = (pODEProblem, u0, conditionId, θ_dynamic) -> PEtab._changeExperimentalCondition(pODEProblem, u0, conditionId, θ_dynamic, petab_model, θ_indices)
    solveODEExperimentalCondition = (odeProblem, conditionId, θ_dynamic, tMax) -> solveOdeModelAtExperimentalCondZygote(odeProblem, conditionId, θ_dynamic, tMax, changeExperimentalCondition, measurementInfo, simulation_info, ode_solver.solver, ode_solver.abstol, ode_solver.reltol, ss_solver.abstol, ss_solver.reltol, sensealg, petab_model.compute_tstops)
    __compute_cost = (θ_est) -> compute_costZygote(θ_est,
                                                 odeProblem,
                                                 petab_model,
                                                 simulation_info,
                                                 θ_indices,
                                                 measurementInfo,
                                                 parameterInfo,
                                                 solveODEExperimentalCondition,
                                                 priorInfo)

    return __compute_cost
end


function PEtab.setSensealg(sensealg, ::Val{:Zygote})

    if !isnothing(sensealg)
        @assert (typeof(sensealg) <: SciMLSensitivity.AbstractSensitivityAlgorithm) "For Zygote an abstract sensitivity algorithm from SciMLSensitivity must be used"
        return sensealg
    end

    return SciMLSensitivity.ForwardSensitivity()
end