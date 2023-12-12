# Transform parameter from log10 scale to normal scale, or reverse transform
function transformθ_zygote(θ::AbstractVector,
                           n_parameters_estimate::Vector{Symbol},
                           parameter_info::PEtab.ParametersInfo;
                           reverse_transform::Bool=false)::AbstractVector

    iθ = [findfirst(x -> x == n_parameters_estimate[i], parameter_info.parameter_id) for i in eachindex(n_parameters_estimate)]
    should_transform = [parameter_info.parameter_scale[i] == :log10 ? true : false for i in iθ]
    should_not_transform = .!should_transform

    if reverse_transform == false
        out = exp10.(θ) .* should_transform .+ θ .* should_not_transform
    else
        out = log10.(θ) .* should_transform .+ θ .* should_not_transform
    end
    return out
end


function PEtab.create_gradient_function(::Val{:Zygote},
                                        ode_problem::ODEProblem,
                                        ode_solver::ODESolver,
                                        ss_solver::SteadyStateSolver,
                                        petab_ODE_cache::PEtab.PEtabODEProblemCache,
                                        petab_ODESolver_cache::PEtab.PEtabODESolverCache,
                                        petab_model::PEtabModel,
                                        simulation_info::PEtab.SimulationInfo,
                                        θ_indices::PEtab.ParameterIndices,
                                        measurement_info::PEtab.MeasurementsInfo,
                                        parameter_info::PEtab.ParametersInfo,
                                        sensealg::SciMLBase.AbstractSensitivityAlgorithm,
                                        prior_info::PEtab.PriorInfo;
                                        chunksize::Union{Nothing, Int64}=nothing,
                                        sensealg_ss=nothing,
                                        n_processes::Int64=1,
                                        jobs=nothing,
                                        results=nothing,
                                        split_over_conditions::Bool=false)

    change_simulation_condition = (p_ode_problem, u0, conditionId, θ_dynamic) -> PEtab._change_simulation_condition(p_ode_problem, u0, conditionId, θ_dynamic, petab_model, θ_indices)
    solve_ode_condition = (ode_problem, conditionId, θ_dynamic, tmax) -> solve_ode_condition_zygote(ode_problem, conditionId, θ_dynamic, tmax, change_simulation_condition, measurement_info, simulation_info, ode_solver.solver, ode_solver.abstol, ode_solver.reltol, ss_solver.abstol, ss_solver.reltol, sensealg, petab_model.compute_tstops)
    _compute_gradient! = (gradient, θ_est) -> compute_gradient_zygote!(gradient,
                                                                       θ_est,
                                                                       ode_problem,
                                                                       petab_model,
                                                                       simulation_info,
                                                                       θ_indices,
                                                                       measurement_info,
                                                                       parameter_info,
                                                                       solve_ode_condition,
                                                                       prior_info,
                                                                       petab_ODE_cache)

    compute_gradient = let _compute_gradient! =_compute_gradient!
        (θ) -> begin
            gradient = zeros(Float64, length(θ))
            _compute_gradient!(gradient, θ)
            return gradient
        end
    end                                                                       
    
    return _compute_gradient!, compute_gradient
end


function PEtab.create_cost_function(::Val{:Zygote},
                                    ode_problem::ODEProblem,
                                    ode_solver::ODESolver,
                                    ss_solver::SteadyStateSolver,
                                    petab_ODE_cache::PEtab.PEtabODEProblemCache,
                                    petab_ODESolver_cache::PEtab.PEtabODESolverCache,
                                    petab_model::PEtabModel,
                                    simulation_info::PEtab.SimulationInfo,
                                    θ_indices::PEtab.ParameterIndices,
                                    measurement_info::PEtab.MeasurementsInfo,
                                    parameter_info::PEtab.ParametersInfo,
                                    prior_info::PEtab.PriorInfo,
                                    sensealg,
                                    n_processes,
                                    jobs,
                                    results,
                                    compute_residuals)

    change_simulation_condition = (p_ode_problem, u0, conditionId, θ_dynamic) -> PEtab._change_simulation_condition(p_ode_problem, u0, conditionId, θ_dynamic, petab_model, θ_indices)
    solve_ode_condition = (ode_problem, conditionId, θ_dynamic, tmax) -> solve_ode_condition_zygote(ode_problem, conditionId, θ_dynamic, tmax, change_simulation_condition, measurement_info, simulation_info, ode_solver.solver, ode_solver.abstol, ode_solver.reltol, ss_solver.abstol, ss_solver.reltol, sensealg, petab_model.compute_tstops)
    __compute_cost = (θ_est) -> compute_cost_zygote(θ_est,
                                                 ode_problem,
                                                 petab_model,
                                                 simulation_info,
                                                 θ_indices,
                                                 measurement_info,
                                                 parameter_info,
                                                 solve_ode_condition,
                                                 prior_info)

    return __compute_cost
end


function PEtab.set_sensealg(sensealg, ::Val{:Zygote})

    if !isnothing(sensealg)
        @assert (typeof(sensealg) <: SciMLSensitivity.AbstractSensitivityAlgorithm) "For Zygote an abstract sensitivity algorithm from SciMLSensitivity must be used"
        return sensealg
    end

    return SciMLSensitivity.ForwardSensitivity()
end