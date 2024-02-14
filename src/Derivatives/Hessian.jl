#=
    The top-level functions for computing the hessian via i) exactly via autodiff, ii) block-approximation via
    auto-diff and iii) guass-newton approximation.
=#

function compute_hessian!(hessian::Matrix{Float64},
                          θ_est::Vector{Float64},
                          _eval_hessian::Function,
                          cfg::ForwardDiff.HessianConfig,
                          simulation_info::SimulationInfo,
                          θ_indices::ParameterIndices,
                          prior_info::PriorInfo)::Nothing

    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough.
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final
    # retcode we cannot catch these cases
    simulation_info.could_solve[1] = true
    if all([simulation_info.ode_sols[id].retcode == ReturnCode.Success ||
            simulation_info.ode_sols[id].retcode == ReturnCode.Terminated
            for id in simulation_info.experimental_condition_id])
        try
            ForwardDiff.hessian!(hessian, _eval_hessian, θ_est, cfg)
            @views hessian .= Symmetric(hessian)
        catch
            hessian .= 0.0
        end
    else
        hessian .= 0.0
    end

    # Check if we could solve the ODE (first), and if Inf was returned (second)
    if simulation_info.could_solve[1] != true
        hessian .= 0.0
        return nothing
    end

    if prior_info.has_priors == true
        compute_hessian_prior!(hessian, θ_est, θ_indices, prior_info)
    end
    return nothing
end

# Compute the hessian via forward mode automatic differentitation where the final hessian is computed via
# n ForwardDiff-calls accross all experimental condtions. The most efficient approach for models with many
# parameters which are unique to each experimental condition.
function compute_hessian_split!(hessian::Matrix{Float64},
                                θ_est::Vector{Float64},
                                _eval_hessian::Function,
                                simulation_info::SimulationInfo,
                                θ_indices::ParameterIndices,
                                prior_info::PriorInfo,
                                exp_id_solve::Vector{Symbol} = [:all])::Nothing

    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough.
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final
    # retcode we cannot catch these cases
    simulation_info.could_solve[1] = true

    hessian .= 0.0
    for conditionId in simulation_info.experimental_condition_id
        map_condition_id = θ_indices.maps_conidition_id[conditionId]
        iθ_experimental_condition = unique(vcat(θ_indices.map_ode_problem.iθ_dynamic,
                                                map_condition_id.iθ_dynamic,
                                                θ_indices.iθ_not_ode))
        θ_input = θ_est[iθ_experimental_condition]
        h_tmp = zeros(length(θ_input), length(θ_input))
        eval_hessian = (θ_arg) -> begin
            _θ_est = convert.(eltype(θ_arg), θ_est)
            _θ_est[iθ_experimental_condition] .= θ_arg
            return _eval_hessian(_θ_est, [conditionId])
        end
        ForwardDiff.hessian!(h_tmp, eval_hessian, θ_input)
        try
            ForwardDiff.hessian!(h_tmp, eval_hessian, θ_input)
        catch
            hessian .= 0.0
            return nothing
        end
        @inbounds for i in eachindex(iθ_experimental_condition)
            @inbounds for j in eachindex(iθ_experimental_condition)
                hessian[iθ_experimental_condition[i], iθ_experimental_condition[j]] += h_tmp[i,
                                                                                             j]
            end
        end
    end

    # Check if we could solve the ODE (first), and if Inf was returned (second)
    if simulation_info.could_solve[1] != true
        hessian .= 0.0
        return nothing
    end

    if prior_info.has_priors == true
        compute_hessian_prior!(hessian, θ_est, θ_indices, prior_info)
    end
    return nothing
end

function compute_hessian_block!(hessian::Matrix{Float64},
                                θ_est::Vector{Float64},
                                compute_cost_θ_not_ODE::Function,
                                compute_cost_θ_dynamic::Function,
                                petab_ODE_cache::PEtabODEProblemCache,
                                cfg::ForwardDiff.HessianConfig,
                                simulation_info::SimulationInfo,
                                θ_indices::ParameterIndices,
                                prior_info::PriorInfo;
                                exp_id_solve::Vector{Symbol} = [:all])::Nothing

    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough.
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final
    # retcode we cannot catch these cases
    simulation_info.could_solve[1] = true

    # Avoid incorrect non-zero values
    hessian .= 0.0

    splitθ!(θ_est, θ_indices, petab_ODE_cache)
    θ_dynamic = petab_ODE_cache.θ_dynamic

    try
        if !isempty(θ_indices.iθ_dynamic)
            @views ForwardDiff.hessian!(hessian[θ_indices.iθ_dynamic, θ_indices.iθ_dynamic],
                                        compute_cost_θ_dynamic, θ_dynamic, cfg)
        else
            compute_cost_θ_dynamic(θ_dynamic)
        end
    catch
        hessian .= 0.0
        return nothing
    end

    # Check if we could solve the ODE (first), and if Inf was returned (second)
    if simulation_info.could_solve[1] != true
        hessian .= 0.0
        return nothing
    end

    iθ_not_ode = θ_indices.iθ_not_ode
    @views ForwardDiff.hessian!(hessian[iθ_not_ode, iθ_not_ode], compute_cost_θ_not_ODE,
                                θ_est[iθ_not_ode])

    # Even though this is a hessian approximation, due to ease of implementation and low run-time we compute the
    # full hessian for the priors
    if prior_info.has_priors == true
        compute_hessian_prior!(hessian, θ_est, θ_indices, prior_info)
    end
    return nothing
end

function compute_hessian_block_split!(hessian::Matrix{Float64},
                                      θ_est::Vector{Float64},
                                      compute_cost_θ_not_ODE::Function,
                                      _compute_cost_θ_dynamic::Function,
                                      petab_ODE_cache::PEtabODEProblemCache,
                                      simulation_info::SimulationInfo,
                                      θ_indices::ParameterIndices,
                                      prior_info::PriorInfo;
                                      exp_id_solve::Vector{Symbol} = [:all])::Nothing

    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough.
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final
    # retcode we cannot catch these cases
    simulation_info.could_solve[1] = true

    # Avoid incorrect non-zero values
    hessian .= 0.0

    splitθ!(θ_est, θ_indices, petab_ODE_cache)
    θ_dynamic = petab_ODE_cache.θ_dynamic

    for conditionId in simulation_info.experimental_condition_id
        map_condition_id = θ_indices.maps_conidition_id[conditionId]
        iθ_experimental_condition = unique(vcat(θ_indices.map_ode_problem.iθ_dynamic,
                                                map_condition_id.iθ_dynamic))
        θ_input = θ_dynamic[iθ_experimental_condition]
        h_tmp = zeros(length(θ_input), length(θ_input))
        compute_cost_θ_dynamic = (θ_arg) -> begin
            _θ_dynamic = convert.(eltype(θ_arg), θ_dynamic)
            @views _θ_dynamic[iθ_experimental_condition] .= θ_arg
            return _compute_cost_θ_dynamic(_θ_dynamic, [conditionId])
        end
        try
            ForwardDiff.hessian!(h_tmp, compute_cost_θ_dynamic, θ_input)
        catch
            hessian .= 0.0
            return nothing
        end
        @inbounds for i in eachindex(iθ_experimental_condition)
            @inbounds for j in eachindex(iθ_experimental_condition)
                hessian[iθ_experimental_condition[i], iθ_experimental_condition[j]] += h_tmp[i,
                                                                                             j]
            end
        end
    end

    # Check if we could solve the ODE (first), and if Inf was returned (second)
    if simulation_info.could_solve[1] != true
        hessian .= 0.0
        return nothing
    end

    iθ_not_ode = θ_indices.iθ_not_ode
    @views ForwardDiff.hessian!(hessian[iθ_not_ode, iθ_not_ode], compute_cost_θ_not_ODE,
                                θ_est[iθ_not_ode])

    # Even though this is a hessian approximation, due to ease of implementation and low run-time we compute the
    # full hessian for the priors
    if prior_info.has_priors == true
        compute_hessian_prior!(hessian, θ_est, θ_indices, prior_info)
    end
    return nothing
end

function compute_GaussNewton_hessian!(out::Matrix{Float64},
                                      θ_est::Vector{Float64},
                                      ode_problem::ODEProblem,
                                      compute_residuals_not_solve_ode!::Function,
                                      petab_model::PEtabModel,
                                      simulation_info::SimulationInfo,
                                      θ_indices::ParameterIndices,
                                      measurement_info::MeasurementsInfo,
                                      parameter_info::ParametersInfo,
                                      _solve_ode_all_conditions!::Function,
                                      prior_info::PriorInfo,
                                      cfg::ForwardDiff.JacobianConfig,
                                      cfg_not_solve_ode::ForwardDiff.JacobianConfig,
                                      petab_ODE_cache::PEtabODEProblemCache;
                                      reuse_sensitivities::Bool = false,
                                      split_over_conditions::Bool = false,
                                      return_jacobian::Bool = false,
                                      exp_id_solve::Vector{Symbol} = [:all],
                                      isremade::Bool = false)::Nothing

    # Avoid incorrect non-zero values
    fill!(out, 0.0)

    splitθ!(θ_est, θ_indices, petab_ODE_cache)
    @unpack θ_dynamic, θ_observable, θ_sd, θ_non_dynamic = petab_ODE_cache
    jacobian_gn = petab_ODE_cache.jacobian_gn
    fill!(jacobian_gn, 0.0)

    # Calculate gradient seperately for dynamic and non dynamic parameter.
    compute_jacobian_residuals_θ_dynamic!((@view jacobian_gn[θ_indices.iθ_dynamic, :]),
                                          θ_dynamic, θ_sd,
                                          θ_observable, θ_non_dynamic, petab_model,
                                          ode_problem,
                                          simulation_info, θ_indices, measurement_info,
                                          parameter_info,
                                          _solve_ode_all_conditions!, cfg, petab_ODE_cache;
                                          exp_id_solve = exp_id_solve,
                                          reuse_sensitivities = reuse_sensitivities,
                                          split_over_conditions = split_over_conditions,
                                          isremade = isremade)

    # Happens when at least one forward pass fails
    if !isempty(θ_dynamic) && all(jacobian_gn[θ_indices.iθ_dynamic, :] .== 1e8)
        out .= 0.0
        return nothing
    end
    @views ForwardDiff.jacobian!(jacobian_gn[θ_indices.iθ_not_ode, :]',
                                 compute_residuals_not_solve_ode!,
                                 petab_ODE_cache.residuals_gn, θ_est[θ_indices.iθ_not_ode],
                                 cfg_not_solve_ode)

    # In case of testing we might want to return the jacobian, else we are interested in the Guass-Newton approximaiton.
    if return_jacobian == false
        out .= jacobian_gn * transpose(jacobian_gn)
    else
        out .= jacobian_gn
        # Even though this is a hessian approximation, due to ease of implementation and low run-time we compute the
        # full hessian for the priors
        if prior_info.has_priors == true
            compute_hessian_prior!(out, θ_est, θ_indices, prior_info)
        end
    end
    return nothing
end

# Compute prior contribution to log-likelihood, note θ in on the parameter scale (e.g might be on log-scale)
function compute_hessian_prior!(hessian::Matrix{Float64},
                                θ::Vector{<:Real},
                                θ_indices::ParameterIndices,
                                prior_info::PriorInfo)::Nothing
    _evalPriors = (θ_est) -> begin
        θ_estT = transformθ(θ_est, θ_indices.θ_names, θ_indices)
        return -1.0 * compute_priors(θ_est, θ_estT, θ_indices.θ_names, prior_info) # We work with -loglik
    end
    hessian .+= ForwardDiff.hessian(_evalPriors, θ)
    return nothing
end
