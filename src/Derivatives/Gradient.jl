#=
    The top-level functions for computing the gradient via i) exactly via forward-mode autodiff, ii) forward sensitivty
    eqations, iii) adjoint sensitivity analysis and iv) Zygote interface.

    Due to it slow speed Zygote does not have full support for all models, e.g, models with priors and pre-eq criteria.
=#

# Compute the gradient via forward mode automatic differentitation
function compute_gradient_autodiff!(gradient::Vector{Float64},
                                    θ_est::Vector{Float64},
                                    compute_cost_θ_not_ODE::Function,
                                    compute_cost_θ_dynamic::Function,
                                    petab_ODE_cache::PEtabODEProblemCache,
                                    cfg::ForwardDiff.GradientConfig,
                                    simulation_info::SimulationInfo,
                                    θ_indices::ParameterIndices,
                                    prior_info::PriorInfo,
                                    exp_id_solve::Vector{Symbol} = [:all];
                                    isremade::Bool = false)::Nothing
    fill!(gradient, 0.0)
    splitθ!(θ_est, θ_indices, petab_ODE_cache)
    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough.
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final
    # retcode we cannot catch these cases
    simulation_info.could_solve[1] = true

    # Case where based on the original PEtab file read into Julia we do not have any parameter vectors fixated.
    if isremade == false ||
       length(petab_ODE_cache.gradient_θ_dyanmic) == petab_ODE_cache.nθ_dynamic[1]
        tmp = petab_ODE_cache.nθ_dynamic[1]
        petab_ODE_cache.nθ_dynamic[1] = length(petab_ODE_cache.θ_dynamic)
        try
            # In case of no dynamic parameters we still need to solve the ODE in order to obtain the gradient for
            # non-dynamic parameters
            if length(petab_ODE_cache.gradient_θ_dyanmic) > 0
                ForwardDiff.gradient!(petab_ODE_cache.gradient_θ_dyanmic,
                                      compute_cost_θ_dynamic, petab_ODE_cache.θ_dynamic,
                                      cfg)
                @views gradient[θ_indices.iθ_dynamic] .= petab_ODE_cache.gradient_θ_dyanmic
            else
                compute_cost_θ_dynamic(petab_ODE_cache.θ_dynamic)
            end
        catch
            gradient .= 0.0
            return nothing
        end
        petab_ODE_cache.nθ_dynamic[1] = tmp
    end

    # Case when we have dynamic parameters fixed. Here it is not always worth to move accross all chunks
    if !(isremade == false ||
         length(petab_ODE_cache.gradient_θ_dyanmic) == petab_ODE_cache.nθ_dynamic[1])
        try
            if petab_ODE_cache.nθ_dynamic[1] != 0
                C = length(cfg.seeds)
                n_forward_passes = Int64(ceil(petab_ODE_cache.nθ_dynamic[1] / C))
                __θ_dynamic = petab_ODE_cache.θ_dynamic[petab_ODE_cache.θ_dynamic_input_order]
                forwarddiff_gradient_chunks(compute_cost_θ_dynamic,
                                            petab_ODE_cache.gradient_θ_dyanmic, __θ_dynamic,
                                            ForwardDiff.Chunk(C);
                                            n_forward_passes = n_forward_passes)
                @views gradient[θ_indices.iθ_dynamic] .= petab_ODE_cache.gradient_θ_dyanmic[petab_ODE_cache.θ_dynamic_output_order]
            else
                compute_cost_θ_dynamic(petab_ODE_cache.θ_dynamic)
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

    θ_not_ode = @view θ_est[θ_indices.iθ_not_ode]
    ForwardDiff.gradient!(petab_ODE_cache.gradient_θ_not_ode, compute_cost_θ_not_ODE,
                          θ_not_ode)
    @views gradient[θ_indices.iθ_not_ode] .= petab_ODE_cache.gradient_θ_not_ode

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
                                          _compute_cost_θ_dynamic::Function,
                                          petab_ODE_cache::PEtabODEProblemCache,
                                          simulation_info::SimulationInfo,
                                          θ_indices::ParameterIndices,
                                          prior_info::PriorInfo,
                                          exp_id_solve = [:all])::Nothing

    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough.
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final
    # retcode we cannot catch these cases
    simulation_info.could_solve[1] = true

    splitθ!(θ_est, θ_indices, petab_ODE_cache)
    θ_dynamic = petab_ODE_cache.θ_dynamic
    fill!(petab_ODE_cache.gradient_θ_dyanmic, 0.0)

    for conditionId in simulation_info.experimental_condition_id
        map_condition_id = θ_indices.maps_conidition_id[conditionId]
        iθ_experimental_condition = unique(vcat(θ_indices.map_ode_problem.iθ_dynamic,
                                                map_condition_id.iθ_dynamic))
        θ_input = θ_dynamic[iθ_experimental_condition]
        compute_cost_θ_dynamic = (θ_arg) -> begin
            _θ_dynamic = convert.(eltype(θ_arg), θ_dynamic)
            @views _θ_dynamic[iθ_experimental_condition] .= θ_arg
            return _compute_cost_θ_dynamic(_θ_dynamic, [conditionId])
        end
        try
            if length(θ_input) ≥ 1
                @views petab_ODE_cache.gradient_θ_dyanmic[iθ_experimental_condition] .+= ForwardDiff.gradient(compute_cost_θ_dynamic,
                                                                                                              θ_input)::Vector{Float64}
            else
                compute_cost_θ_dynamic(θ_input)
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
    @views gradient[θ_indices.iθ_dynamic] .= petab_ODE_cache.gradient_θ_dyanmic

    θ_not_ode = @view θ_est[θ_indices.iθ_not_ode]
    ForwardDiff.gradient!(petab_ODE_cache.gradient_θ_not_ode, compute_cost_θ_not_ODE,
                          θ_not_ode)
    @views gradient[θ_indices.iθ_not_ode] .= petab_ODE_cache.gradient_θ_not_ode

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
                                             petab_model::PEtabModel,
                                             ode_problem::ODEProblem,
                                             sensealg,
                                             simulation_info::SimulationInfo,
                                             θ_indices::ParameterIndices,
                                             measurement_info::MeasurementsInfo,
                                             parameter_info::ParametersInfo,
                                             _solve_ode_all_conditions!::Function,
                                             prior_info::PriorInfo,
                                             cfg::Union{ForwardDiff.JacobianConfig,
                                                        Nothing},
                                             petab_ODE_cache::PEtabODEProblemCache;
                                             split_over_conditions::Bool = false,
                                             exp_id_solve::Vector{Symbol} = [:all],
                                             isremade::Bool = false)::Nothing

    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough.
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final
    # retcode we cannot catch these cases
    simulation_info.could_solve[1] = true

    splitθ!(θ_est, θ_indices, petab_ODE_cache)
    @unpack θ_dynamic, θ_observable, θ_sd, θ_non_dynamic = petab_ODE_cache

    # Calculate gradient seperately for dynamic and non dynamic parameter.
    compute_gradient_forward_equations!(petab_ODE_cache.gradient_θ_dyanmic, θ_dynamic, θ_sd,
                                        θ_observable, θ_non_dynamic, petab_model,
                                        sensealg, ode_problem, simulation_info, θ_indices,
                                        measurement_info, parameter_info,
                                        _solve_ode_all_conditions!, cfg, petab_ODE_cache,
                                        exp_id_solve = exp_id_solve,
                                        split_over_conditions = split_over_conditions,
                                        isremade = isremade)
    @views gradient[θ_indices.iθ_dynamic] .= petab_ODE_cache.gradient_θ_dyanmic

    # Happens when at least one forward pass fails and I set the gradient to 1e8
    if !isempty(petab_ODE_cache.gradient_θ_dyanmic) &&
       all(petab_ODE_cache.gradient_θ_dyanmic .== 0.0)
        gradient .= 0.0
        return nothing
    end

    θ_not_ode = @view θ_est[θ_indices.iθ_not_ode]
    ReverseDiff.gradient!(petab_ODE_cache.gradient_θ_not_ode, compute_cost_θ_not_ODE,
                          θ_not_ode)
    @views gradient[θ_indices.iθ_not_ode] .= petab_ODE_cache.gradient_θ_not_ode

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
        θ_estT = transformθ(θ_est, θ_indices.θ_names, θ_indices)
        return -1.0 * compute_priors(θ_est, θ_estT, θ_indices.θ_names, prior_info) # We work with -loglik
    end
    gradient .+= ForwardDiff.gradient(_eval_priors, θ)
    return nothing
end
