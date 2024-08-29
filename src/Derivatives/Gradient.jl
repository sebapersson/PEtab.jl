#=
    The top-level functions for computing the gradient via i) exactly via forward-mode autodiff, ii) forward sensitivty
    eqations, iii) adjoint sensitivity analysis and iv) Zygote interface.

    Due to it slow speed Zygote does not have full support for all models, e.g, models with priors and pre-eq criteria.
=#

function grad_forward_AD!(grad::Vector{T}, x::Vector{T}, _nllh_not_solveode::Function,
                          _nllh_solveode::Function, cfg::ForwardDiff.GradientConfig,
                          probleminfo::PEtabODEProblemInfo, model_info::ModelInfo;
                          cids::Vector{Symbol} = [:all], isremade::Bool = false)::Nothing where T <: AbstractFloat
    cache = probleminfo.cache
    @unpack simulation_info, θ_indices, prior_info = model_info
    @unpack xdynamic_grad, xnotode_grad, xdynamic, nxdynamic = cache


    # As a subset of ForwardDiff chunks might fail, return code status is checked via
    # simulation info
    simulation_info.could_solve[1] = true

    # When remaking the problem order of parameters a subset of xdynamic is fixed which
    # must be accounted for in gradient computations
    fill!(grad, 0.0)
    split_x!(x, θ_indices, cache)
    if isremade == false || length(xdynamic) == nxdynamic[1]
        tmp = nxdynamic[1]
        nxdynamic[1] = length(xdynamic)
        try
            # In case of no length(xdynamic) = 0 the ODE must still be solved to get
            # the gradient of nondynamic parameters
            if length(xdynamic_grad) != 0
                ForwardDiff.gradient!(xdynamic_grad, _nllh_solveode, xdynamic, cfg)
                @views grad[θ_indices.xindices[:dynamic]] .= xdynamic_grad
            else
                _ = _nllh_solveode(xdynamic)
            end
        catch
            fill!(grad, 0.0)
            return nothing
        end
        nxdynamic[1] = tmp
    else
        try
            if nxdynamic[1] != 0
                C = length(cfg.seeds)
                chunk = ForwardDiff.Chunk(C)
                nforward_passes = ceil(nxdynamic[1] / C) |> Int64
                _xdynamic = xdynamic[cache.xdynamic_input_order]
                forwarddiff_gradient_chunks(_nllh_solveode, xdynamic_grad, _xdynamic,
                                            chunk; n_forward_passes = nforward_passes)
                @views grad[θ_indices.xindices[:dynamic]] .= xdynamic_grad[cache.xdynamic_output_order]
            else
                _ = _nllh_solveode(xdynamic)
            end
        catch
            fill!(grad, 0.0)
            return nothing
        end
    end

    # In case ODE could not be solved return zero gradient
    if simulation_info.could_solve[1] != true
        fill!(grad, 0.0)
        return nothing
    end

    # Not dynamic parameter not part of ODE (only need an ODE solution for gradient)
    x_notode = @view x[θ_indices.xindices[:not_system]]
    ForwardDiff.gradient!(xnotode_grad, _nllh_not_solveode, x_notode)
    @views grad[θ_indices.xindices[:not_system]] .= xnotode_grad

    # TODO: Refactor prior later
    if prior_info.has_priors == true
        compute_gradient_prior!(grad, x, θ_indices, prior_info)
    end
    return nothing
end

# Compute the gradient via forward mode automatic differentitation where the final gradient is computed via
# n ForwardDiff-calls accross all experimental condtions. The most efficient approach for models with many
# parameters which are unique to each experimental condition.
function compute_gradient_autodiff_split!(gradient::Vector{Float64},
                                          x::Vector{Float64},
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

    split_x!(x, θ_indices, cache)
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
        fill!(gradient, 0.0)
        return nothing
    end
    @views gradient[θ_indices.xindices[:dynamic]] .= cache.xdynamic_grad

    θ_not_ode = @view x[θ_indices.xindices[:not_system]]
    ForwardDiff.gradient!(cache.xnotode_grad, compute_cost_θ_not_ODE,
                          θ_not_ode)
    @views gradient[θ_indices.xindices[:not_system]] .= cache.xnotode_grad

    # If we have prior contribution its gradient is computed via autodiff for all parameters
    if prior_info.has_priors == true
        compute_gradient_prior!(gradient, x, θ_indices, prior_info)
    end
    return nothing
end

# Compute the gradient via forward sensitivity equations
function compute_gradient_forward_equations!(gradient::Vector{Float64},
                                             x::Vector{Float64},
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

    split_x!(x, θ_indices, cache)
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
        fill!(gradient, 0.0)
        return nothing
    end

    θ_not_ode = @view x[θ_indices.xindices[:not_system]]
    ReverseDiff.gradient!(cache.xnotode_grad, compute_cost_θ_not_ODE,
                          θ_not_ode)
    @views gradient[θ_indices.xindices[:not_system]] .= cache.xnotode_grad

    if prior_info.has_priors == true
        compute_gradient_prior!(gradient, x, θ_indices, prior_info)
    end
    return nothing
end

# Compute prior contribution to log-likelihood
function compute_gradient_prior!(gradient::Vector{Float64},
                                 θ::Vector{<:Real},
                                 θ_indices::ParameterIndices,
                                 prior_info::PriorInfo)::Nothing
    _eval_priors = (x) -> begin
        xT = transform_x(x, θ_indices.xids[:estimate], θ_indices)
        return -1.0 * compute_priors(x, xT, θ_indices.xids[:estimate], prior_info) # We work with -loglik
    end
    gradient .+= ForwardDiff.gradient(_eval_priors, θ)
    return nothing
end
