function compute_cost(θ_est::Vector{T},
                      ode_problem::ODEProblem,
                      ode_solver::ODESolver,
                      ss_solver::SteadyStateSolver,
                      petab_model::PEtabModel,
                      simulation_info::SimulationInfo,
                      θ_indices::ParameterIndices,
                      measurement_info::MeasurementsInfo,
                      parameter_info::ParametersInfo,
                      prior_info::PriorInfo,
                      petab_ODE_cache::PEtabODEProblemCache,
                      petab_ODESolver_cache::PEtabODESolverCache,
                      exp_id_solve::Vector{Symbol},
                      compute_cost::Bool,
                      compute_hessian::Bool,
                      compute_residuals::Bool)::T where T<:Real

    θ_dynamic, θ_observable, θ_sd, θ_non_dynamic = splitθ(θ_est, θ_indices)

    cost = compute_cost_solve_ODE(θ_dynamic, θ_sd, θ_observable,  θ_non_dynamic, ode_problem, ode_solver, ss_solver, petab_model,
                               simulation_info, θ_indices, measurement_info, parameter_info, petab_ODE_cache, petab_ODESolver_cache,
                               compute_cost=compute_cost,
                               compute_hessian=compute_hessian,
                               compute_residuals=compute_residuals,
                               exp_id_solve=exp_id_solve)

    if prior_info.has_priors == true && compute_hessian == false
        θ_estT = transformθ(θ_est, θ_indices.θ_names, θ_indices)
        cost -= compute_priors(θ_est, θ_estT, θ_indices.θ_names, prior_info) # We work with -loglik
    end

    return cost
end


function compute_cost_solve_ODE(θ_dynamic::T1,
                                θ_sd::T2,
                                θ_observable::T2,
                                θ_non_dynamic::T2,
                                ode_problem::ODEProblem,
                                ode_solver::ODESolver,
                                ss_solver::SteadyStateSolver,
                                petab_model::PEtabModel,
                                simulation_info::SimulationInfo,
                                θ_indices::ParameterIndices,
                                measurement_info::MeasurementsInfo,
                                parameter_info::ParametersInfo,
                                petab_ODE_cache::PEtabODEProblemCache,
                                petab_ODESolver_cache::PEtabODESolverCache;
                                compute_cost::Bool=false,
                                compute_hessian::Bool=false,
                                compute_gradient_θ_dynamic::Bool=false,
                                compute_residuals::Bool=false,
                                exp_id_solve::Vector{Symbol} = [:all])::Real where {T1<:AbstractVector, T2<:AbstractVector} 

    if compute_gradient_θ_dynamic == true && petab_ODE_cache.nθ_dynamic[1] != length(θ_dynamic)
        _θ_dynamic = θ_dynamic[petab_ODE_cache.θ_dynamic_output_order]
        θ_dynamicT = transformθ(_θ_dynamic, θ_indices.θ_dynamic_names, θ_indices, :θ_dynamic, petab_ODE_cache)
    else
        θ_dynamicT = transformθ(θ_dynamic, θ_indices.θ_dynamic_names, θ_indices, :θ_dynamic, petab_ODE_cache)
    end

    θ_sdT = transformθ(θ_sd, θ_indices.θ_sd_names, θ_indices, :θ_sd, petab_ODE_cache)
    θ_observableT = transformθ(θ_observable, θ_indices.θ_observable_names, θ_indices, :θ_observable, petab_ODE_cache)
    θ_non_dynamicT = transformθ(θ_non_dynamic, θ_indices.θ_non_dynamic_names, θ_indices, :θ_non_dynamic, petab_ODE_cache)

    _ode_problem = remake(ode_problem, p = convert.(eltype(θ_dynamicT), ode_problem.p), u0 = convert.(eltype(θ_dynamicT), ode_problem.u0))
    change_ode_parameters!(_ode_problem.p, _ode_problem.u0, θ_dynamicT, θ_indices, petab_model)

    # If computing hessian or gradient store ODE solution in arrary with dual numbers, else use
    # solution array with floats
    if compute_hessian == true || compute_gradient_θ_dynamic == true
        success = solve_ode_all_conditions!(simulation_info.ode_sols_derivatives, _ode_problem, petab_model, θ_dynamicT, petab_ODESolver_cache, simulation_info, θ_indices, ode_solver, ss_solver, exp_id_solve=exp_id_solve, dense_sol=false, save_at_observed_t=true)
    elseif compute_cost == true
        success = solve_ode_all_conditions!(simulation_info.ode_sols, _ode_problem, petab_model, θ_dynamicT, petab_ODESolver_cache, simulation_info, θ_indices, ode_solver, ss_solver, exp_id_solve=exp_id_solve, dense_sol=false, save_at_observed_t=true)
    end
    if success != true
        if ode_solver.verbose == true 
            @warn "Failed to solve ODE model"
        end
        return Inf
    end

    cost = _compute_cost(θ_sdT, θ_observableT,  θ_non_dynamicT, petab_model, simulation_info, θ_indices, measurement_info,
                         parameter_info, exp_id_solve,
                         compute_hessian=compute_hessian,
                         compute_gradient_θ_dynamic=compute_gradient_θ_dynamic,
                         compute_residuals=compute_residuals)

    return cost
end


function compute_cost_not_solve_ODE(θ_sd::T1,
                                    θ_observable::T1,
                                    θ_non_dynamic::T1,
                                    petab_model::PEtabModel,
                                    simulation_info::SimulationInfo,
                                    θ_indices::ParameterIndices,
                                    measurement_info::MeasurementsInfo,
                                    parameter_info::ParametersInfo,
                                    petab_ODE_cache::PEtabODEProblemCache;
                                    compute_gradient_not_solve_autodiff::Bool=false,
                                    compute_gradient_not_solve_adjoint::Bool=false,
                                    compute_gradient_not_solve_forward::Bool=false,
                                    exp_id_solve::Vector{Symbol} = [:all])::Real where {T1<:AbstractVector}

    # To be able to use ReverseDiff sdParamEstUse and obsParamEstUse cannot be overwritten.
    # Hence new vectors have to be created. Minimal overhead.
    θ_sdT = transformθ(θ_sd, θ_indices.θ_sd_names, θ_indices, :θ_sd, petab_ODE_cache)
    θ_observableT = transformθ(θ_observable, θ_indices.θ_observable_names, θ_indices, :θ_observable, petab_ODE_cache)
    θ_non_dynamicT = transformθ(θ_non_dynamic, θ_indices.θ_non_dynamic_names, θ_indices, :θ_non_dynamic, petab_ODE_cache)

    cost = _compute_cost(θ_sdT, θ_observableT,  θ_non_dynamicT, petab_model, simulation_info, θ_indices,
                         measurement_info, parameter_info, exp_id_solve,
                         compute_gradient_not_solve_autodiff=compute_gradient_not_solve_autodiff,
                         compute_gradient_not_solve_adjoint=compute_gradient_not_solve_adjoint,
                         compute_gradient_not_solve_forward=compute_gradient_not_solve_forward)

    return cost
end


function _compute_cost(θ_sd::T1,
                       θ_observable::T1,
                       θ_non_dynamic::T1,
                       petab_model::PEtabModel,
                       simulation_info::SimulationInfo,
                       θ_indices::ParameterIndices,
                       measurement_info::MeasurementsInfo,
                       parameter_info::ParametersInfo,
                       exp_id_solve::Vector{Symbol} = [:all];
                       compute_hessian::Bool=false,
                       compute_gradient_θ_dynamic::Bool=false,
                       compute_residuals::Bool=false,
                       compute_gradient_not_solve_adjoint::Bool=false,
                       compute_gradient_not_solve_forward::Bool=false,
                       compute_gradient_not_solve_autodiff::Bool=false)::Real where T1<:AbstractVector

    if compute_hessian == true || compute_gradient_θ_dynamic == true || compute_gradient_not_solve_adjoint == true || compute_gradient_not_solve_forward == true || compute_gradient_not_solve_autodiff == true
        ode_sols = simulation_info.ode_sols_derivatives
    else
        ode_sols = simulation_info.ode_sols
    end

    cost = 0.0
    for experimental_condition_id in simulation_info.experimental_condition_id

        if exp_id_solve[1] != :all && experimental_condition_id ∉ exp_id_solve
            continue
        end

        # Extract the ODE-solution for specific condition ID
        ode_sol = ode_sols[experimental_condition_id]
        cost += compute_cost_condition(ode_sol, Float64[], θ_sd, θ_observable,  θ_non_dynamic, petab_model,
                                       experimental_condition_id, θ_indices, measurement_info, parameter_info, 
                                       simulation_info,
                                       compute_residuals=compute_residuals,
                                       compute_gradient_not_solve_adjoint=compute_gradient_not_solve_adjoint,
                                       compute_gradient_not_solve_forward=compute_gradient_not_solve_forward,
                                       compute_gradient_not_solve_autodiff=compute_gradient_not_solve_autodiff)

        if isinf(cost)
            return Inf
        end
    end

    return cost
end


function compute_cost_condition(ode_sol::ODESolution,
                                p_ode_problem_zygote::T1,
                                θ_sd::T2,
                                θ_observable::AbstractVector,
                                θ_non_dynamic::AbstractVector,
                                petab_model::PEtabModel,
                                experimental_condition_id::Symbol,
                                θ_indices::ParameterIndices,
                                measurement_info::MeasurementsInfo,
                                parameter_info::ParametersInfo,
                                simulation_info::SimulationInfo;
                                compute_residuals::Bool=false,
                                compute_gradient_not_solve_adjoint::Bool=false,
                                compute_gradient_not_solve_forward::Bool=false,
                                compute_gradient_not_solve_autodiff::Bool=false,
                                compute_gradient_θ_dynamic_zygote::Bool=false)::Real where {T1<:AbstractVector, T2<:AbstractVector}

    if !(ode_sol.retcode == ReturnCode.Success || ode_sol.retcode == ReturnCode.Terminated)
        return Inf
    end

    cost = 0.0
    for i_measurement in simulation_info.i_measurements[experimental_condition_id]

        t = measurement_info.time[i_measurement]

        # In these cases we only save the ODE at observed time-points and we do not want
        # to extract Dual ODE solution
        if compute_gradient_not_solve_forward == true || compute_gradient_not_solve_autodiff == true
            n_model_states = length(petab_model.state_names)
            u = dual_to_float.(ode_sol[1:n_model_states, simulation_info.i_time_ode_sol[i_measurement]])
            p = dual_to_float.(ode_sol.prob.p)
        # For adjoint sensitivity analysis we have a dense-ode solution
        elseif compute_gradient_not_solve_adjoint == true
            # In case we only have sol.t = 0.0 (or similar) interpolation does not work
            u = length(ode_sol.t) > 1 ? ode_sol(t) : ode_sol[1]
            p = ode_sol.prob.p

        elseif compute_gradient_θ_dynamic_zygote == true
            u = ode_sol.u[simulation_info.i_time_ode_sol[i_measurement], :][1]
            p = p_ode_problem_zygote

        # When we want to extract dual number from the ODE solution
        else
            u = ode_sol[:, simulation_info.i_time_ode_sol[i_measurement]]
            p = ode_sol.prob.p
        end

        h = computeh(u, t, p, θ_observable,  θ_non_dynamic, petab_model, i_measurement, measurement_info, θ_indices, parameter_info)
        hT = transform_measurement_or_h(h, measurement_info.measurement_transformation[i_measurement])
        σ = computeσ(u, t, p, θ_sd,  θ_non_dynamic, petab_model, i_measurement, measurement_info, θ_indices, parameter_info)
        residual = (hT - measurement_info.measurementT[i_measurement]) / σ

        # These values might be needed by different software, e.g. PyPesto, to assess things such as parameter uncertainity. By storing them in
        # measurement_info they can easily be computed given a call to the cost function has been made.
        update_measurement_info!(measurement_info, h, hT, σ, residual, i_measurement)

        # By default a positive ODE solution is not enforced (even though the user can provide it as option).
        # In case with transformations on the data the code can crash, hence Inf is returned in case the
        # model data transformation can not be perfomred.
        if isinf(hT)
            println("Warning - transformed observable is non-finite for measurement $i_measurement")
            return Inf
        end

        # The user can provide a noise formula, e.g σ = 0.1*I(t), which technically can go below zero due to 
        # numerical errors when solving the ODE. However, having a noise formula which can go to zero is 
        # probably a bad noise formula to start with, so in case this happen throw a warning
        if σ < 0.0
            ChainRulesCore.@ignore_derivatives begin
                @warn "Measurement noise σ is smaller than 0. Consider changing the noise formula so it\ncannot go below zero. This issue likelly happens due to numerical noise when solving the ODE" maxlog=10
            end
            return Inf
        end

        # Update log-likelihood. In case of guass newton approximation we are only interested in the residuals, and here
        # we allow the residuals to be computed to test the gauss-newton implementation
        if compute_residuals == false
            if measurement_info.measurement_transformation[i_measurement] === :lin
                cost += log(σ) + 0.5*log(2*pi) + 0.5*residual^2
            elseif measurement_info.measurement_transformation[i_measurement] === :log10
                cost += log(σ) + 0.5*log(2*pi) + log(log(10)) + log(10)*measurement_info.measurementT[i_measurement] + 0.5*residual^2
            elseif measurement_info.measurement_transformation[i_measurement] === :log
                cost += log(σ) + 0.5*log(2*pi) + log(measurement_info.measurement[i_measurement]) + 0.5*residual^2
            else
                println("Transformation ", measurement_info.measurement_transformation[i_measurement], " not yet supported.")
                return Inf
            end
        elseif compute_residuals == true
            cost += residual
        end
    end
    return cost
end


function update_measurement_info!(measurement_info::MeasurementsInfo, h::T, hT::T, σ::T, residual::T, i_measurement::Integer)::Nothing where {T<:AbstractFloat}
    ChainRulesCore.@ignore_derivatives begin
        measurement_info.simulated_values[i_measurement] = h
        measurement_info.chi2_values[i_measurement] = (hT - measurement_info.measurementT[i_measurement])^2 / σ^2
        measurement_info.residuals[i_measurement] = residual
    end
    return nothing
end
function update_measurement_info!(measurement_info::MeasurementsInfo, h, hT, σ, residual, i_measurement)::Nothing
    return nothing
end