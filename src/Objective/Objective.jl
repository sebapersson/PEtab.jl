function compute_cost(θ_est::Vector{T},
                      probleminfo::PEtabODEProblemInfo,
                      model_info::ModelInfo,
                      exp_id_solve::Vector{Symbol},
                      compute_cost::Bool,
                      compute_hessian::Bool,
                      compute_residuals::Bool)::T where {T <: Real}
    @unpack θ_indices, prior_info = model_info
    xdynamic, xobservable, xnoise, xnondynamic = splitθ(θ_est, model_info.θ_indices)
    cost = compute_cost_solve_ODE(xdynamic, xnoise, xobservable, xnondynamic, probleminfo,
                                  model_info; compute_cost = compute_cost,
                                  compute_hessian = compute_hessian,
                                  compute_residuals = compute_residuals,
                                  exp_id_solve = exp_id_solve)

    if prior_info.has_priors == true && compute_hessian == false
        θ_estT = transform_x(θ_est, θ_indices.xids[:estimate], θ_indices)
        cost -= compute_priors(θ_est, θ_estT, θ_indices.xids[:estimate], prior_info) # We work with -loglik
    end
    return cost
end

function compute_cost_solve_ODE(xdynamic::T1, xnoise::T2, xobservable::T2, xnondynamic::T2,
                                probleminfo::PEtabODEProblemInfo, model_info::ModelInfo;
                                compute_cost::Bool = false, compute_hessian::Bool = false,
                                compute_gradient_xdynamic::Bool = false,
                                compute_residuals::Bool = false,
                                exp_id_solve::Vector{Symbol} = [:all])::Real where {
                                                                                    T1 <:
                                                                                    AbstractVector,
                                                                                    T2 <:
                                                                                    AbstractVector
                                                                                    }
    @unpack θ_indices, simulation_info = model_info
    @unpack cache = probleminfo
    if compute_gradient_xdynamic == true &&
       cache.nxdynamic[1] != length(xdynamic)
        _xdynamic = xdynamic[cache.xdynamic_output_order]
        xdynamic_ps = transform_x(_xdynamic, θ_indices.xids[:dynamic], θ_indices,
                                :xdynamic, cache)
    else
        xdynamic_ps = transform_x(xdynamic, θ_indices.xids[:dynamic], θ_indices, :xdynamic,
                                cache)
    end
    xnoise_ps = transform_x(xnoise, θ_indices.xids[:noise], θ_indices, :xnoise, cache)
    xobservable_ps = transform_x(xobservable, θ_indices.xids[:observable], θ_indices,
                               :xobservable, cache)
    xnondynamic_ps = transform_x(xnondynamic, θ_indices.xids[:nondynamic], θ_indices,
                                :xnondynamic, cache)

    # If computing hessian or gradient store ODE solution in arrary with dual numbers, else use
    # solution array with floats
    derivative = compute_hessian == true || compute_gradient_xdynamic == true
    success = solve_ode_all_conditions!(model_info, xdynamic_ps, probleminfo,
                                        exp_id_solve = exp_id_solve, dense_sol = false,
                                        save_at_observed_t = true, derivative = derivative)
    if success != true
        if probleminfo.solver.verbose == true
            @warn "Failed to solve ODE model"
        end
        return Inf
    end

    cost = _compute_cost(xnoise_ps, xobservable_ps, xnondynamic_ps, model_info, exp_id_solve;
                         compute_hessian = compute_hessian,
                         compute_gradient_xdynamic = compute_gradient_xdynamic,
                         compute_residuals = compute_residuals)

    return cost
end

function compute_cost_not_solve_ODE(xnoise::T1,
                                    xobservable::T1,
                                    xnondynamic::T1,
                                    probleminfo::PEtabODEProblemInfo,
                                    model_info::ModelInfo;
                                    compute_gradient_not_solve_autodiff::Bool = false,
                                    compute_gradient_not_solve_adjoint::Bool = false,
                                    compute_gradient_not_solve_forward::Bool = false,
                                    exp_id_solve::Vector{Symbol} = [:all])::Real where {T1 <:
                                                                                        AbstractVector}
    @unpack θ_indices = model_info
    @unpack cache = probleminfo
    # To be able to use ReverseDiff sdParamEstUse and obsParamEstUse cannot be overwritten.
    # Hence new vectors have to be created. Minimal overhead.
    xnoise_ps = transform_x(xnoise, θ_indices.xids[:noise], θ_indices, :xnoise, cache)
    xobservable_ps = transform_x(xobservable, θ_indices.xids[:observable], θ_indices,
                               :xobservable, cache)
    xnondynamic_ps = transform_x(xnondynamic, θ_indices.xids[:nondynamic], θ_indices,
                                :xnondynamic, cache)

    cost = _compute_cost(xnoise_ps, xobservable_ps, xnondynamic_ps, model_info, exp_id_solve,
                         compute_gradient_not_solve_autodiff = compute_gradient_not_solve_autodiff,
                         compute_gradient_not_solve_adjoint = compute_gradient_not_solve_adjoint,
                         compute_gradient_not_solve_forward = compute_gradient_not_solve_forward)

    return cost
end

function _compute_cost(xnoise::T1,
                       xobservable::T1,
                       xnondynamic::T1,
                       model_info::ModelInfo,
                       exp_id_solve::Vector{Symbol} = [:all];
                       compute_hessian::Bool = false,
                       compute_gradient_xdynamic::Bool = false,
                       compute_residuals::Bool = false,
                       compute_gradient_not_solve_adjoint::Bool = false,
                       compute_gradient_not_solve_forward::Bool = false,
                       compute_gradient_not_solve_autodiff::Bool = false)::Real where {T1 <:
                                                                                       AbstractVector}
    @unpack simulation_info = model_info
    if compute_hessian == true || compute_gradient_xdynamic == true ||
       compute_gradient_not_solve_adjoint == true ||
       compute_gradient_not_solve_forward == true ||
       compute_gradient_not_solve_autodiff == true
        ode_sols = simulation_info.odesols_derivatives
    else
        ode_sols = simulation_info.odesols
    end

    cost = 0.0
    for experimental_condition_id in simulation_info.conditionids[:experiment]
        if exp_id_solve[1] != :all && experimental_condition_id ∉ exp_id_solve
            continue
        end

        # Extract the ODE-solution for specific condition ID
        ode_sol = ode_sols[experimental_condition_id]
        cost += compute_cost_condition(ode_sol, xnoise, xobservable, xnondynamic,
                                       experimental_condition_id, model_info,
                                       compute_residuals = compute_residuals,
                                       compute_gradient_not_solve_adjoint = compute_gradient_not_solve_adjoint,
                                       compute_gradient_not_solve_forward = compute_gradient_not_solve_forward,
                                       compute_gradient_not_solve_autodiff = compute_gradient_not_solve_autodiff)

        if isinf(cost)
            return Inf
        end
    end

    return cost
end

function compute_cost_condition(ode_sol::ODESolution,
                                xnoise::AbstractVector,
                                xobservable::AbstractVector,
                                xnondynamic::AbstractVector,
                                experimental_condition_id::Symbol,
                                model_info::ModelInfo;
                                compute_residuals::Bool = false,
                                compute_gradient_not_solve_adjoint::Bool = false,
                                compute_gradient_not_solve_forward::Bool = false,
                                compute_gradient_not_solve_autodiff::Bool = false)::Real
    @unpack θ_indices, simulation_info, measurement_info, parameter_info, petab_model = model_info
    if !(ode_sol.retcode == ReturnCode.Success || ode_sol.retcode == ReturnCode.Terminated)
        return Inf
    end

    cost = 0.0
    for i_measurement in simulation_info.imeasurements[experimental_condition_id]
        t = measurement_info.time[i_measurement]

        # In these cases we only save the ODE at observed time-points and we do not want
        # to extract Dual ODE solution
        if compute_gradient_not_solve_forward == true ||
           compute_gradient_not_solve_autodiff == true
            n_model_states = length(states(petab_model.sys_mutated))
            u = dual_to_float.(ode_sol[1:n_model_states,
                                       simulation_info.imeasurements_t_sol[i_measurement]])
            p = dual_to_float.(ode_sol.prob.p)
            # For adjoint sensitivity analysis we have a dense-ode solution
        elseif compute_gradient_not_solve_adjoint == true
            # In case we only have sol.t = 0.0 (or similar) interpolation does not work
            u = length(ode_sol.t) > 1 ? ode_sol(t) : ode_sol[1]
            p = ode_sol.prob.p
        else
            u = ode_sol[:, simulation_info.imeasurements_t_sol[i_measurement]]
            p = ode_sol.prob.p
        end

        h = computeh(u, t, p, xobservable, xnondynamic, petab_model, i_measurement,
                     measurement_info, θ_indices, parameter_info)
        hT = transform_measurement_or_h(h,
                                        measurement_info.measurement_transformation[i_measurement])
        σ = computeσ(u, t, p, xnoise, xnondynamic, petab_model, i_measurement,
                     measurement_info, θ_indices, parameter_info)
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
                cost += log(σ) + 0.5 * log(2 * pi) + 0.5 * residual^2
            elseif measurement_info.measurement_transformation[i_measurement] === :log10
                cost += log(σ) + 0.5 * log(2 * pi) + log(log(10)) +
                        log(10) * measurement_info.measurementT[i_measurement] +
                        0.5 * residual^2
            elseif measurement_info.measurement_transformation[i_measurement] === :log
                cost += log(σ) + 0.5 * log(2 * pi) +
                        log(measurement_info.measurement[i_measurement]) + 0.5 * residual^2
            else
                println("Transformation ",
                        measurement_info.measurement_transformation[i_measurement],
                        " not yet supported.")
                return Inf
            end
        elseif compute_residuals == true
            cost += residual
        end
    end
    return cost
end

function update_measurement_info!(measurement_info::MeasurementsInfo, h::T, hT::T, σ::T,
                                  res::T,
                                  i_measurement::Integer)::Nothing where {T <:
                                                                          AbstractFloat}
    ChainRulesCore.@ignore_derivatives begin
        measurement_info.simulated_values[i_measurement] = h
        mT = measurement_info.measurementT
        measurement_info.chi2_values[i_measurement] = (hT - mT[i_measurement])^2 / σ^2
        measurement_info.residuals[i_measurement] = res
    end
    return nothing
end
function update_measurement_info!(measurement_info::MeasurementsInfo, h, hT, σ, residual,
                                  i_measurement)::Nothing
    return nothing
end
