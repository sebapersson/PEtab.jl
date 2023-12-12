#=
    Functions specific to gradient compuations via forward sensitivity equations. Notice that we can solve the
    forward system either via i) solving the expanded ODE-system or ii) by using AutoDiff to obtain the sensitivites,
    which efficiently are the jacobian of the ODESolution.

    There are two cases. When we compute the Jacobian of the ODE-solution via autodiff (sensealg=:AutoDiff) we compute
    a big Jacobian matrix (sensitivity matrix) across all experimental condition, while using one of the Julia forward
    algorithms we compute a "small" Jacobian for each experimental condition.
=#


function compute_gradient_forward_equations!(gradient::Vector{Float64},
                                             θ_dynamic::Vector{Float64},
                                             θ_sd::Vector{Float64},
                                             θ_observable::Vector{Float64},
                                             θ_non_dynamic::Vector{Float64},
                                             petab_model::PEtabModel,
                                             sensealg,
                                             ode_problem::ODEProblem,
                                             simulation_info::SimulationInfo,
                                             θ_indices::ParameterIndices,
                                             measurement_info ::MeasurementsInfo,
                                             parameter_info::ParametersInfo,
                                             _solve_ode_all_conditions!::Function,
                                             cfg::Union{ForwardDiff.JacobianConfig, Nothing},
                                             petab_ODE_cache::PEtabODEProblemCache;
                                             exp_id_solve::Vector{Symbol} = [:all],
                                             split_over_conditions::Bool=false,
                                             isremade::Bool=false)::Nothing

    θ_dynamicT = transformθ(θ_dynamic, θ_indices.θ_dynamic_names, θ_indices, :θ_dynamic, petab_ODE_cache)
    θ_sdT = transformθ(θ_sd, θ_indices.θ_sd_names, θ_indices, :θ_sd, petab_ODE_cache)
    θ_observableT = transformθ(θ_observable, θ_indices.θ_observable_names, θ_indices, :θ_observable, petab_ODE_cache)
    θ_non_dynamicT = transformθ(θ_non_dynamic, θ_indices.θ_non_dynamic_names, θ_indices, :θ_non_dynamic, petab_ODE_cache)

    # Solve the expanded ODE system for the sensitivites
    success = solve_sensitivites(ode_problem, simulation_info, θ_indices, petab_model, sensealg, θ_dynamicT,
                                 _solve_ode_all_conditions!, cfg, petab_ODE_cache, exp_id_solve, split_over_conditions,
                                 isremade)
    if success != true
        @warn "Failed to solve sensitivity equations"
        gradient .= 1e8
        return nothing
    end
    if isempty(θ_dynamic)
        return nothing
    end

    gradient .= 0.0
    for i in eachindex(simulation_info.experimental_condition_id)
        experimental_condition_id = simulation_info.experimental_condition_id[i]
        simulation_condition_id = simulation_info.simulation_condition_id[i]

        if exp_id_solve[1] != :all && experimental_condition_id ∉ exp_id_solve
            continue
        end

        sol = simulation_info.ode_sols_derivatives[experimental_condition_id]

        # If we have a callback it needs to be properly handled
        compute_gradient_forward_equations_condition!(gradient, sol, petab_ODE_cache, sensealg, θ_dynamicT, θ_sdT,
                                                      θ_observableT,  θ_non_dynamicT, experimental_condition_id, 
                                                      simulation_condition_id, simulation_info, petab_model, θ_indices, 
                                                      measurement_info, parameter_info)
    end
    return nothing
end


function solve_sensitivites(ode_problem::ODEProblem,
                            simulation_info::SimulationInfo,
                            θ_indices::ParameterIndices,
                            petab_model::PEtabModel,
                            sensealg::Symbol,
                            θ_dynamic::AbstractVector,
                            _solve_ode_all_conditions!::Function,
                            cfg::ForwardDiff.JacobianConfig,
                            petab_ODE_cache::PEtabODEProblemCache,
                            exp_id_solve::Vector{Symbol},
                            split_over_conditions::Bool,
                            isremade::Bool=false)::Bool

    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough.
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final
    # retcode we cannot catch these cases
    simulation_info.could_solve[1] = true
    petab_ODE_cache.S .= 0.0
    if split_over_conditions == false

        # Case where based on the original PEtab file read into Julia we do not have any parameter vectors fixated.
        if isremade == false || length(petab_ODE_cache.gradient_θ_dyanmic) == petab_ODE_cache.nθ_dynamic[1]

            # Allow correct mapping to sensitivity matrix
            tmp = petab_ODE_cache.nθ_dynamic[1]
            petab_ODE_cache.nθ_dynamic[1] = length(θ_dynamic)

            if isempty(θ_dynamic)
                _solve_ode_all_conditions!(petab_ODE_cache.sol_values, θ_dynamic)
                petab_ODE_cache.S .= 0.0
            end

            if !isempty(θ_dynamic)
                ForwardDiff.jacobian!(petab_ODE_cache.S, _solve_ode_all_conditions!, petab_ODE_cache.sol_values, θ_dynamic, cfg)
            end

            petab_ODE_cache.nθ_dynamic[1] = tmp
        end

        # Case when we have dynamic parameters fixed. Here it is not always worth to move accross all chunks
        if !(isremade == false || length(petab_ODE_cache.gradient_θ_dyanmic) == petab_ODE_cache.nθ_dynamic[1])
            if petab_ODE_cache.nθ_dynamic[1] != 0
                C = length(cfg.seeds)
                n_forward_passes = Int64(ceil(petab_ODE_cache.nθ_dynamic[1] / C))
                __θ_dynamic = θ_dynamic[petab_ODE_cache.θ_dynamic_input_order]
                forwarddiff_jacobian_chunks(_solve_ode_all_conditions!, petab_ODE_cache.sol_values, petab_ODE_cache.S, __θ_dynamic, ForwardDiff.Chunk(C); n_forward_passes=n_forward_passes)
                @views petab_ODE_cache.S .= petab_ODE_cache.S[:, petab_ODE_cache.θ_dynamic_output_order]
            end

            if petab_ODE_cache.nθ_dynamic[1] == 0
                _solve_ode_all_conditions!(petab_ODE_cache.sol_values, θ_dynamic)
                petab_ODE_cache.S .= 0.0
            end
        end
    end

    # Slower option, but more efficient if there are several parameters unique to an experimental condition
    if split_over_conditions == true

        petab_ODE_cache.S .= 0.0
        S_tmp = similar(petab_ODE_cache.S)
        for condition_id in simulation_info.experimental_condition_id
            map_condition_id = θ_indices.maps_conidition_id[condition_id]
            iθ_experimental_condition = unique(vcat(θ_indices.map_ode_problem.iθ_dynamic, map_condition_id.iθ_dynamic))
            θ_input = θ_dynamic[iθ_experimental_condition]
            compute_sensitivities_condition! = (sol_values, θ_arg) ->    begin
                                                                                    _θ_dynamic = convert.(eltype(θ_arg), θ_dynamic)
                                                                                    _θ_dynamic[iθ_experimental_condition] .= θ_arg
                                                                                    _solve_ode_all_conditions!(sol_values, _θ_dynamic, [condition_id])
                                                                                end
            @views ForwardDiff.jacobian!(S_tmp[:, iθ_experimental_condition], compute_sensitivities_condition!, petab_ODE_cache.sol_values, θ_input)
            @views petab_ODE_cache.S[:, iθ_experimental_condition] .+= S_tmp[:, iθ_experimental_condition]
        end
    end

    return simulation_info.could_solve[1]
end


function compute_gradient_forward_equations_condition!(gradient::Vector{Float64},
                                                       sol::ODESolution,
                                                       petab_ODE_cache::PEtabODEProblemCache,
                                                       sensealg::Symbol,
                                                       θ_dynamic::Vector{Float64},
                                                       θ_sd::Vector{Float64},
                                                       θ_observable::Vector{Float64},
                                                       θ_non_dynamic::Vector{Float64},
                                                       experimental_condition_id::Symbol,
                                                       simulation_condition_id::Symbol,
                                                       simulation_info::SimulationInfo,
                                                       petab_model::PEtabModel,
                                                       θ_indices::ParameterIndices,
                                                       measurement_info::MeasurementsInfo,
                                                       parameter_info::ParametersInfo)::Nothing

    i_per_time_point = simulation_info.i_per_time_point[experimental_condition_id]
    time_observed = simulation_info.time_observed[experimental_condition_id]
    time_position_ode_sol = simulation_info.time_position_ode_sol[experimental_condition_id]

    # To compute
    compute∂G∂u! = (out, u, p, t, i) -> begin compute∂G∂_(out, u, p, t, i, i_per_time_point,
                                                         measurement_info, parameter_info,
                                                         θ_indices, petab_model,
                                                         θ_sd, θ_observable,  θ_non_dynamic,
                                                         petab_ODE_cache.∂h∂u, petab_ODE_cache.∂σ∂u, compute∂G∂U=true)
                                            end
    compute∂G∂p! = (out, u, p, t, i) -> begin compute∂G∂_(out, u, p, t, i, i_per_time_point,
                                                         measurement_info, parameter_info,
                                                         θ_indices, petab_model,
                                                         θ_sd, θ_observable,  θ_non_dynamic,
                                                         petab_ODE_cache.∂h∂p, petab_ODE_cache.∂σ∂p, compute∂G∂U=false)
                                        end

    # Extract which parameters we compute gradient for in this specific experimental condition
    map_condition_id = θ_indices.maps_conidition_id[simulation_condition_id]
    # Unique is needed to account for condition specific parameters which maps to potentially several
    # parameters in ODEProblem.p
    iθ_experimental_condition = unique(vcat(θ_indices.map_ode_problem.iθ_dynamic, map_condition_id.iθ_dynamic))

    # Loop through solution and extract sensitivites
    n_model_states = length(petab_model.state_names)
    petab_ODE_cache.p .= dual_to_float.(sol.prob.p)
    p = petab_ODE_cache.p
    u = petab_ODE_cache.u
    ∂G∂p, ∂G∂p_, ∂G∂u = petab_ODE_cache.∂G∂p, petab_ODE_cache.∂G∂p_, petab_ODE_cache.∂G∂u
    _gradient = petab_ODE_cache._gradient
    fill!(_gradient, 0.0)
    fill!(∂G∂p, 0.0)
    for i in eachindex(time_observed)
        u .= dual_to_float.((@view sol[:, i]))
        compute∂G∂u!(∂G∂u, u, p, time_observed[i], i)
        compute∂G∂p!(∂G∂p_, u, p, time_observed[i], i)
        # We need to extract the correct indices from the big sensitivity matrix (row is observation at specific time
        # point). Overall, positions are precomputed in time_position_ode_sol
        i_start, i_end = (time_position_ode_sol[i]-1)*n_model_states+1, (time_position_ode_sol[i]-1)*n_model_states + n_model_states
        _S = @view petab_ODE_cache.S[i_start:i_end, iθ_experimental_condition]
        @views _gradient[iθ_experimental_condition] .+= transpose(_S)*∂G∂u
        ∂G∂p .+= ∂G∂p_
    end

    # Thus far have have computed dY/dθ, but for parameters on the log-scale we want dY/dθ_log. We can adjust via;
    # dY/dθ_log = log(10) * θ * dY/dθ
    adjust_gradient_θ_transformed!(gradient, _gradient, ∂G∂p, θ_dynamic, θ_indices,
                                         simulation_condition_id, autodiff_sensitivites=true)

    return nothing
end
