function PEtab.create_gradient_function(which_method::Symbol,
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
                                        sensealg::Union{InterpolatingAdjoint, QuadratureAdjoint},
                                        prior_info::PEtab.PriorInfo;
                                        chunksize::Union{Nothing, Int64}=nothing,
                                        sensealg_ss=nothing,
                                        n_processes::Int64=1,
                                        jobs=nothing,
                                        results=nothing,
                                        split_over_conditions::Bool=false)::Function

    _sensealg_ss = isnothing(sensealg_ss) ? InterpolatingAdjoint(autojacvec=ReverseDiffVJP()) : sensealg_ss
    # Fast but numerically unstable method
    if simulation_info.has_pre_equilibration_condition_id == true && typeof(_sensealg_ss) <: SteadyStateAdjoint
        @warn "If using adjoint sensitivity analysis for a model with PreEq-criteria the most the most efficient sensealg_ss is as provided SteadyStateAdjoint. However, SteadyStateAdjoint fails if the Jacobian is singular hence we recomend you check that the Jacobian is non-singular."
    end

    iθ_sd, iθ_observable, iθ_non_dynamic, iθ_not_ode = PEtab.get_index_parameters_not_ODE(θ_indices)
    compute_cost_θ_not_ODE = let iθ_sd=iθ_sd, iθ_observable=iθ_observable, iθ_non_dynamic=iθ_non_dynamic,
                                 petab_model=petab_model, simulation_info=simulation_info, θ_indices=θ_indices,
                                 measurement_info=measurement_info, parameter_info=parameter_info, petab_ODE_cache=petab_ODE_cache

                                (x) -> PEtab.compute_cost_not_solve_ODE(x[iθ_sd],
                                                                        x[iθ_observable],
                                                                        x[iθ_non_dynamic],
                                                                        petab_model,
                                                                        simulation_info,
                                                                        θ_indices,
                                                                        measurement_info,
                                                                        parameter_info,
                                                                        petab_ODE_cache,
                                                                        exp_id_solve=[:all],
                                                                        compute_gradient_not_solve_adjoint=true)
                             end

    _compute_gradient! = let ode_solver=ode_solver, ss_solver=ss_solver, compute_cost_θ_not_ODE=compute_cost_θ_not_ODE, 
                             sensealg=sensealg, sensealg_ss=sensealg_ss, ode_problem=ode_problem, petab_model=petab_model, 
                             simulation_info=simulation_info, θ_indices=θ_indices, measurement_info=measurement_info, 
                             parameter_info=parameter_info, prior_info=prior_info, petab_ODE_cache=petab_ODE_cache, 
                             petab_ODESolver_cache=petab_ODESolver_cache

                            (gradient, θ) -> compute_gradient_adjoint!(gradient,
                                                                       θ,
                                                                       ode_solver,
                                                                       ss_solver,
                                                                       compute_cost_θ_not_ODE,
                                                                       sensealg,
                                                                       _sensealg_ss,
                                                                       ode_problem,
                                                                       petab_model,
                                                                       simulation_info,
                                                                       θ_indices,
                                                                       measurement_info,
                                                                       parameter_info,
                                                                       prior_info,
                                                                       petab_ODE_cache,
                                                                       petab_ODESolver_cache,
                                                                       exp_id_solve=[:all])
                         end
    
    return _compute_gradient!
end


function PEtab.set_sensealg(sensealg, ::Val{:Adjoint})

    if !isnothing(sensealg)
        @assert any(typeof(sensealg) .<: [InterpolatingAdjoint, QuadratureAdjoint]) "For gradient method :Adjoint allowed sensealg args are InterpolatingAdjoint, QuadratureAdjoint not $sensealg"
        return sensealg
    end

    return InterpolatingAdjoint(autojacvec=ReverseDiffVJP())
end
function PEtab.set_sensealg(sensealg::Union{ForwardSensitivity, ForwardDiffSensitivity}, ::Val{:ForwardEquations})
    return sensealg
end


function PEtab.get_callbackset(ode_problem::ODEProblem,
                               simulation_info::PEtab.SimulationInfo,
                               simulation_condition_id::Symbol,
                               sensealg::Union{InterpolatingAdjoint, QuadratureAdjoint})::SciMLBase.DECallback

    cbset = SciMLSensitivity.track_callbacks(simulation_info.callbacks[simulation_condition_id], ode_problem.tspan[1],
                                                 ode_problem.u0, ode_problem.p, sensealg)
    simulation_info.tracked_callbacks[simulation_condition_id] = cbset
    return cbset
end