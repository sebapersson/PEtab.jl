#=
    Functions specific to gradient compuations via forward sensitivity equations. Notice that we can solve the
    forward system either via i) solving the expanded ODE-system or ii) by using AutoDiff to obtain the sensitivites,
    which efficiently are the jacobian of the ODESolution.

    There are two cases. When we compute the Jacobian of the ODE-solution via autodiff (sensealg=:AutoDiff) we compute
    a big Jacobian matrix (sensitivity matrix) across all experimental condition, while using one of the Julia forward
    algorithms we compute a "small" Jacobian for each experimental condition.
=#

function compute_gradient_forward_equations!(gradient::Vector{Float64},
                                             xdynamic::Vector{Float64},
                                             xnoise::Vector{Float64},
                                             xobservable::Vector{Float64},
                                             xnondynamic::Vector{Float64},
                                             petab_model::PEtabModel,
                                             sensealg,
                                             ode_problem::ODEProblem,
                                             simulation_info::SimulationInfo,
                                             θ_indices::ParameterIndices,
                                             measurement_info::MeasurementsInfo,
                                             parameter_info::ParametersInfo,
                                             _solve_ode_all_conditions!::Function,
                                             cfg::Union{ForwardDiff.JacobianConfig,
                                                        Nothing},
                                             cache::PEtabODEProblemCache;
                                             exp_id_solve::Vector{Symbol} = [:all],
                                             split_over_conditions::Bool = false,
                                             isremade::Bool = false)::Nothing
    xnoise_ps = PEtab.transform_x(xnoise, θ_indices, :xnoise, cache)
    xobservable_ps = PEtab.transform_x(xobservable, θ_indices, :xobservable, cache)
    xnondynamic_ps = PEtab.transform_x(xnondynamic, θ_indices, :xnondynamic, cache)
    xdynamic_ps = PEtab.transform_x(xdynamic, θ_indices, :xdynamic, cache)

    # Solve the expanded ODE system for the sensitivites
    success = solve_sensitivites(ode_problem, simulation_info, θ_indices, petab_model,
                                 sensealg, xdynamic_ps,
                                 _solve_ode_all_conditions!, cfg, cache,
                                 exp_id_solve, split_over_conditions,
                                 isremade)
    if success != true
        @warn "Failed to solve sensitivity equations"
        gradient .= 1e8
        return nothing
    end
    if isempty(xdynamic)
        return nothing
    end

    gradient .= 0.0
    for i in eachindex(simulation_info.conditionids[:experiment])
        experimental_condition_id = simulation_info.conditionids[:experiment][i]
        simulation_condition_id = simulation_info.conditionids[:simulation][i]

        if exp_id_solve[1] != :all && experimental_condition_id ∉ exp_id_solve
            continue
        end

        sol = simulation_info.odesols_derivatives[experimental_condition_id]

        # If we have a callback it needs to be properly handled
        compute_gradient_forward_equations_condition!(gradient, sol, cache,
                                                      sensealg, xdynamic_ps, xnoise_ps,
                                                      xobservable_ps, xnondynamic_ps,
                                                      experimental_condition_id,
                                                      simulation_condition_id,
                                                      simulation_info, petab_model,
                                                      θ_indices,
                                                      measurement_info, parameter_info)
    end
    return nothing
end

function solve_sensitivites(ode_problem::ODEProblem,
                            simulation_info::SimulationInfo,
                            θ_indices::ParameterIndices,
                            petab_model::PEtabModel,
                            sensealg::Symbol,
                            xdynamic::AbstractVector,
                            _solve_ode_all_conditions!::Function,
                            cfg::ForwardDiff.JacobianConfig,
                            cache::PEtabODEProblemCache,
                            exp_id_solve::Vector{Symbol},
                            split_over_conditions::Bool,
                            isremade::Bool = false)::Bool

    # We need to track a variable if ODE system could be solve as checking retcode on solution array it not enough.
    # This is because for ForwardDiff some chunks can solve the ODE, but other fail, and thus if we check the final
    # retcode we cannot catch these cases
    simulation_info.could_solve[1] = true
    cache.S .= 0.0
    if split_over_conditions == false

        # Case where based on the original PEtab file read into Julia we do not have any parameter vectors fixated.
        if isremade == false ||
           length(cache.xdynamic_grad) == cache.nxdynamic[1]

            # Allow correct mapping to sensitivity matrix
            tmp = cache.nxdynamic[1]
            cache.nxdynamic[1] = length(xdynamic)

            if isempty(xdynamic)
                _solve_ode_all_conditions!(cache.odesols, xdynamic)
                cache.S .= 0.0
            end

            if !isempty(xdynamic)
                ForwardDiff.jacobian!(cache.S, _solve_ode_all_conditions!,
                                      cache.odesols, xdynamic, cfg)
            end

            cache.nxdynamic[1] = tmp
        end

        # Case when we have dynamic parameters fixed. Here it is not always worth to move accross all chunks
        if !(isremade == false ||
             length(cache.xdynamic_grad) == cache.nxdynamic[1])
            if cache.nxdynamic[1] != 0
                C = length(cfg.seeds)
                n_forward_passes = Int64(ceil(cache.nxdynamic[1] / C))
                __xdynamic = xdynamic[cache.xdynamic_input_order]
                forwarddiff_jacobian_chunks(_solve_ode_all_conditions!,
                                            cache.odesols, cache.S,
                                            __xdynamic, ForwardDiff.Chunk(C);
                                            n_forward_passes = n_forward_passes)
                @views cache.S .= cache.S[:,
                                                              cache.xdynamic_output_order]
            end

            if cache.nxdynamic[1] == 0
                _solve_ode_all_conditions!(cache.odesols, xdynamic)
                cache.S .= 0.0
            end
        end
    end

    # Slower option, but more efficient if there are several parameters unique to an experimental condition
    if split_over_conditions == true
        cache.S .= 0.0
        S_tmp = similar(cache.S)
        for condition_id in simulation_info.conditionids[:experiment]
            map_condition_id = θ_indices.maps_conidition_id[condition_id]
            iθ_experimental_condition = unique(vcat(θ_indices.map_ode_problem.sys_to_dynamic,
                                                    map_condition_id.ix_dynamic))
            θ_input = xdynamic[iθ_experimental_condition]
            compute_sensitivities_condition! = (odesols, θ_arg) -> begin
                _xdynamic = convert.(eltype(θ_arg), xdynamic)
                _xdynamic[iθ_experimental_condition] .= θ_arg
                _solve_ode_all_conditions!(odesols, _xdynamic, [condition_id])
            end
            @views ForwardDiff.jacobian!(S_tmp[:, iθ_experimental_condition],
                                         compute_sensitivities_condition!,
                                         cache.odesols, θ_input)
            @views cache.S[:, iθ_experimental_condition] .+= S_tmp[:,
                                                                             iθ_experimental_condition]
        end
    end

    return simulation_info.could_solve[1]
end

function compute_gradient_forward_equations_condition!(gradient::Vector{Float64},
                                                       sol::ODESolution,
                                                       cache::PEtabODEProblemCache,
                                                       sensealg::Symbol,
                                                       xdynamic::Vector{Float64},
                                                       xnoise::Vector{Float64},
                                                       xobservable::Vector{Float64},
                                                       xnondynamic::Vector{Float64},
                                                       experimental_condition_id::Symbol,
                                                       simulation_condition_id::Symbol,
                                                       simulation_info::SimulationInfo,
                                                       petab_model::PEtabModel,
                                                       θ_indices::ParameterIndices,
                                                       measurement_info::MeasurementsInfo,
                                                       parameter_info::ParametersInfo)::Nothing
    imeasurements_t = simulation_info.imeasurements_t[experimental_condition_id]
    time_observed = simulation_info.tsaves[experimental_condition_id]
    time_position_ode_sol = simulation_info.smatrixindices[experimental_condition_id]

    # To compute
    compute∂G∂u! = (out, u, p, t, i) -> begin
        compute∂G∂_(out, u, p, t, i, imeasurements_t,
                    measurement_info, parameter_info,
                    θ_indices, petab_model,
                    xnoise, xobservable, xnondynamic,
                    cache.∂h∂u, cache.∂σ∂u, compute∂G∂U = true)
    end
    compute∂G∂p! = (out, u, p, t, i) -> begin
        compute∂G∂_(out, u, p, t, i, imeasurements_t,
                    measurement_info, parameter_info,
                    θ_indices, petab_model,
                    xnoise, xobservable, xnondynamic,
                    cache.∂h∂p, cache.∂σ∂p, compute∂G∂U = false)
    end

    # Extract which parameters we compute gradient for in this specific experimental condition
    map_condition_id = θ_indices.maps_conidition_id[simulation_condition_id]
    # Unique is needed to account for condition specific parameters which maps to potentially several
    # parameters in ODEProblem.p
    iθ_experimental_condition = unique(vcat(θ_indices.map_ode_problem.sys_to_dynamic,
                                            map_condition_id.ix_dynamic))

    # Loop through solution and extract sensitivites
    n_model_states = length(states(petab_model.sys_mutated))
    cache.p .= dual_to_float.(sol.prob.p)
    p = cache.p
    u = cache.u
    ∂G∂p, ∂G∂p_, ∂G∂u = cache.∂G∂p, cache.∂G∂p_, cache.∂G∂u
    _gradient = cache.forward_eqs_grad
    fill!(_gradient, 0.0)
    fill!(∂G∂p, 0.0)
    for i in eachindex(time_observed)
        u .= dual_to_float.((@view sol[:, i]))
        compute∂G∂u!(∂G∂u, u, p, time_observed[i], i)
        compute∂G∂p!(∂G∂p_, u, p, time_observed[i], i)
        # We need to extract the correct indices from the big sensitivity matrix (row is observation at specific time
        # point). Overall, positions are precomputed in time_position_ode_sol
        i_start, i_end = (time_position_ode_sol[i] - 1) * n_model_states + 1,
                         (time_position_ode_sol[i] - 1) * n_model_states + n_model_states
        _S = @view cache.S[i_start:i_end, iθ_experimental_condition]
        @views _gradient[iθ_experimental_condition] .+= transpose(_S) * ∂G∂u
        ∂G∂p .+= ∂G∂p_
    end

    # Thus far have have computed dY/dθ, but for parameters on the log-scale we want dY/dθ_log. We can adjust via;
    # dY/dθ_log = log(10) * θ * dY/dθ
    adjust_gradient_θ_transformed!(gradient, _gradient, ∂G∂p, xdynamic, θ_indices,
                                   simulation_condition_id, autodiff_sensitivites = true)

    return nothing
end
