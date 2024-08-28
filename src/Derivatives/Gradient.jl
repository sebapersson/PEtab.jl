#=
    The top-level functions for computing the gradient via i) exactly via forward-mode autodiff, ii) forward sensitivty
    eqations, iii) adjoint sensitivity analysis and iv) Zygote interface.

    Due to it slow speed Zygote does not have full support for all models, e.g, models with priors and pre-eq criteria.
=#

# Compute the gradient via forward mode automatic differentitation
function compute_gradient_autodiff!(gradient::Vector{Float64},
                                    θ_est::Vector{Float64},
                                    compute_cost_θ_not_ODE::Function,
                                    compute_cost_xdynamic::Function,
                                    cfg::ForwardDiff.GradientConfig,
                                    model_info::ModelInfo,
                                    probleminfo::PEtabODEProblemInfo;
                                    exp_id_solve::Vector{Symbol} = [:all],
                                    isremade::Bool = false)::Nothing
    @unpack simulation_info, θ_indices, prior_info = model_info
    @unpack cache = probleminfo
    fill!(gradient, 0.0)
    splitθ!(θ_est, θ_indices, cache)
    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough.
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final
    # retcode we cannot catch these cases
    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough.
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final
    # retcode we cannot catch these cases
    simulation_info.could_solve[1] = true

    # Case where based on the original PEtab file read into Julia we do not have any parameter vectors fixated.
    if isremade == false ||
       length(cache.xdynamic_grad) == cache.nxdynamic[1]
        tmp = cache.nxdynamic[1]
        cache.nxdynamic[1] = length(cache.xdynamic)
        try
            # In case of no dynamic parameters we still need to solve the ODE in order to obtain the gradient for
            # non-dynamic parameters
            if length(cache.xdynamic_grad) > 0
                ForwardDiff.gradient!(cache.xdynamic_grad,
                                      compute_cost_xdynamic, cache.xdynamic,
                                      cfg)
                @views gradient[θ_indices.xindices[:dynamic]] .= cache.xdynamic_grad
            else
                compute_cost_xdynamic(cache.xdynamic)
            end
        catch
            gradient .= 0.0
            return nothing
        end
        cache.nxdynamic[1] = tmp
    end

    # Case when we have dynamic parameters fixed. Here it is not always worth to move accross all chunks
    if !(isremade == false ||
         length(cache.xdynamic_grad) == cache.nxdynamic[1])
        try
            if cache.nxdynamic[1] != 0
                C = length(cfg.seeds)
                n_forward_passes = Int64(ceil(cache.nxdynamic[1] / C))
                __xdynamic = cache.xdynamic[cache.xdynamic_input_order]
                forwarddiff_gradient_chunks(compute_cost_xdynamic,
                                            cache.xdynamic_grad, __xdynamic,
                                            ForwardDiff.Chunk(C);
                                            n_forward_passes = n_forward_passes)
                @views gradient[θ_indices.xindices[:dynamic]] .= cache.xdynamic_grad[cache.xdynamic_output_order]
            else
                compute_cost_xdynamic(cache.xdynamic)
            end
        catch
            gradient .= 0.0
            return nothing
        end
    end

    # Check if we could solve the ODE (first), and if Inf was returned (second)
    if simulation_info.could_solve[1] != true
        gradient .= 0.0
        return nothing
    end

    θ_not_ode = @view θ_est[θ_indices.xindices[:not_system]]
    ForwardDiff.gradient!(cache.xnotode_grad, compute_cost_θ_not_ODE,
                          θ_not_ode)
    @views gradient[θ_indices.xindices[:not_system]] .= cache.xnotode_grad

    # If we have prior contribution its gradient is computed via autodiff for all parameters
    if prior_info.has_priors == true
        compute_gradient_prior!(gradient, θ_est, θ_indices, prior_info)
    end
    return nothing
end

# Compute the gradient via forward mode automatic differentitation where the final gradient is computed via
# n ForwardDiff-calls accross all experimental condtions. The most efficient approach for models with many
# parameters which are unique to each experimental condition.
function compute_gradient_autodiff_split!(gradient::Vector{Float64},
                                          θ_est::Vector{Float64},
                                          compute_cost_θ_not_ODE::Function,
                                          _compute_cost_xdynamic::Function,
                                          probleminfo::PEtabODEProblemInfo,
                                          model_info::ModelInfo;
                                          exp_id_solve = [:all])::Nothing
    @unpack cache = probleminfo
    @unpack simulation_info, θ_indices, prior_info = model_info
    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough.
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final
    # retcode we cannot catch these cases
    simulation_info.could_solve[1] = true

    splitθ!(θ_est, θ_indices, cache)
    xdynamic = cache.xdynamic
    fill!(cache.xdynamic_grad, 0.0)

    for conditionId in simulation_info.conditionids[:experiment]
        map_condition_id = θ_indices.maps_conidition_id[conditionId]
        iθ_experimental_condition = unique(vcat(θ_indices.map_ode_problem.sys_to_dynamic,
                                                map_condition_id.ix_dynamic))
        θ_input = xdynamic[iθ_experimental_condition]
        compute_cost_xdynamic = (θ_arg) -> begin
            _xdynamic = convert.(eltype(θ_arg), xdynamic)
            @views _xdynamic[iθ_experimental_condition] .= θ_arg
            return _compute_cost_xdynamic(_xdynamic, [conditionId])
        end
        try
            if length(θ_input) ≥ 1
                @views cache.xdynamic_grad[iθ_experimental_condition] .+= ForwardDiff.gradient(compute_cost_xdynamic,
                                                                                                              θ_input)::Vector{Float64}
            else
                compute_cost_xdynamic(θ_input)
            end
        catch
            gradient .= 1e8
            return nothing
        end
    end

    # Check if we could solve the ODE (first), and if Inf was returned (second)
    if simulation_info.could_solve[1] != true
        gradient .= 0.0
        return nothing
    end
    @views gradient[θ_indices.xindices[:dynamic]] .= cache.xdynamic_grad

    θ_not_ode = @view θ_est[θ_indices.xindices[:not_system]]
    ForwardDiff.gradient!(cache.xnotode_grad, compute_cost_θ_not_ODE,
                          θ_not_ode)
    @views gradient[θ_indices.xindices[:not_system]] .= cache.xnotode_grad

    # If we have prior contribution its gradient is computed via autodiff for all parameters
    if prior_info.has_priors == true
        compute_gradient_prior!(gradient, θ_est, θ_indices, prior_info)
    end
    return nothing
end

# Compute the gradient via forward sensitivity equations
function compute_gradient_forward_equations!(gradient::Vector{Float64},
                                             θ_est::Vector{Float64},
                                             compute_cost_θ_not_ODE::Function,
                                             _solve_ode_all_conditions!::Function,
                                             probleminfo::PEtabODEProblemInfo,
                                             model_info::ModelInfo,
                                             cfg::Union{ForwardDiff.JacobianConfig,
                                                        Nothing};
                                             exp_id_solve::Vector{Symbol} = [:all],
                                             isremade::Bool = false)::Nothing
    @unpack sensealg, cache, split_over_conditions = probleminfo
    @unpack simulation_info, petab_model, simulation_info, θ_indices = model_info
    @unpack parameter_info, prior_info, measurement_info = model_info
    ode_problem = probleminfo.odeproblem_gradient
    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough.
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final
    # retcode we cannot catch these cases
    simulation_info.could_solve[1] = true

    splitθ!(θ_est, θ_indices, cache)
    @unpack xdynamic, xobservable, xnoise, xnondynamic = cache

    # Calculate gradient seperately for dynamic and non dynamic parameter.
    compute_gradient_forward_equations!(cache.xdynamic_grad, xdynamic, xnoise,
                                        xobservable, xnondynamic, petab_model,
                                        sensealg, ode_problem, simulation_info, θ_indices,
                                        measurement_info, parameter_info,
                                        _solve_ode_all_conditions!, cfg, cache,
                                        exp_id_solve = exp_id_solve,
                                        split_over_conditions = split_over_conditions,
                                        isremade = isremade)
    @views gradient[θ_indices.xindices[:dynamic]] .= cache.xdynamic_grad

    # Happens when at least one forward pass fails and I set the gradient to 1e8
    if !isempty(cache.xdynamic_grad) &&
       all(cache.xdynamic_grad .== 0.0)
        gradient .= 0.0
        return nothing
    end

    θ_not_ode = @view θ_est[θ_indices.xindices[:not_system]]
    ReverseDiff.gradient!(cache.xnotode_grad, compute_cost_θ_not_ODE,
                          θ_not_ode)
    @views gradient[θ_indices.xindices[:not_system]] .= cache.xnotode_grad

    if prior_info.has_priors == true
        compute_gradient_prior!(gradient, θ_est, θ_indices, prior_info)
    end
    return nothing
end

# Compute prior contribution to log-likelihood
function compute_gradient_prior!(gradient::Vector{Float64},
                                 θ::Vector{<:Real},
                                 θ_indices::ParameterIndices,
                                 prior_info::PriorInfo)::Nothing
    _eval_priors = (θ_est) -> begin
        θ_estT = transform_x(θ_est, θ_indices.xids[:estimate], θ_indices)
        return -1.0 * compute_priors(θ_est, θ_estT, θ_indices.xids[:estimate], prior_info) # We work with -loglik
    end
    gradient .+= ForwardDiff.gradient(_eval_priors, θ)
    return nothing
end
