function compute_jacobian_residuals_xdynamic!(jacobian::Union{Matrix{Float64}, SubArray},
                                               xdynamic::Vector{Float64},
                                               xnoise::Vector{Float64},
                                               xobservable::Vector{Float64},
                                               xnondynamic::Vector{Float64},
                                               petab_model::PEtabModel,
                                               ode_problem::ODEProblem,
                                               simulation_info::SimulationInfo,
                                               θ_indices::ParameterIndices,
                                               measurement_info::MeasurementsInfo,
                                               parameter_info::ParametersInfo,
                                               _solve_ode_all_conditions!::Function,
                                               cfg::ForwardDiff.JacobianConfig,
                                               cache::PEtabODEProblemCache;
                                               exp_id_solve::Vector{Symbol} = [:all],
                                               reuse_sensitivities::Bool = false,
                                               split_over_conditions::Bool = false,
                                               isremade::Bool = false)::Nothing
    xnoise_ps = PEtab.transform_x(xnoise, θ_indices, :xnoise, cache)
    xobservable_ps = PEtab.transform_x(xobservable, θ_indices, :xobservable, cache)
    xnondynamic_ps = PEtab.transform_x(xnondynamic, θ_indices, :xnondynamic, cache)
    xdynamic_ps = PEtab.transform_x(xdynamic, θ_indices, :xdynamic, cache)

    if reuse_sensitivities == false
        # Solve the expanded ODE system for the sensitivites
        success = solve_sensitivites(ode_problem, simulation_info, θ_indices, petab_model,
                                     :ForwardDiff, xdynamic_ps,
                                     _solve_ode_all_conditions!, cfg, cache,
                                     exp_id_solve, split_over_conditions,
                                     isremade)

        if success != true
            @warn "Failed to solve sensitivity equations"
            jacobian .= 1e8
            return nothing
        end
    end
    if isempty(xdynamic)
        jacobian .= 0.0
        return nothing
    end

    # Compute the gradient by looping through all experimental conditions.
    for i in eachindex(simulation_info.conditionids[:experiment])
        experimental_condition_id = simulation_info.conditionids[:experiment][i]
        simulation_condition_id = simulation_info.conditionids[:simulation][i]

        if exp_id_solve[1] != :all && experimental_condition_id ∉ exp_id_solve
            continue
        end

        sol = simulation_info.odesols_derivatives[experimental_condition_id]

        # If we have a callback it needs to be properly handled
        compute_jacobian_residuals_condition!(jacobian, sol, cache, xdynamic_ps,
                                              xnoise_ps, xobservable_ps,
                                              xnondynamic_ps, experimental_condition_id,
                                              simulation_condition_id, simulation_info,
                                              petab_model, θ_indices,
                                              measurement_info, parameter_info)
    end
    return nothing
end

function compute_jacobian_residuals_condition!(jacobian::M,
                                               sol::ODESolution,
                                               cache::PEtabODEProblemCache,
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
                                               parameter_info::ParametersInfo)::Nothing where {
                                                                                               M <:
                                                                                               AbstractMatrix
                                                                                               }
    imeasurements_t = simulation_info.imeasurements_t[experimental_condition_id]
    time_observed = simulation_info.tsaves[experimental_condition_id]
    time_position_ode_sol = simulation_info.smatrixindices[experimental_condition_id]

    # To compute
    compute∂G∂u = (out, u, p, t, i, it) -> begin
        compute∂G∂_(out, u, p, t, i, it,
                    measurement_info, parameter_info,
                    θ_indices, petab_model,
                    xnoise, xobservable, xnondynamic,
                    cache.∂h∂u, cache.∂σ∂u, compute∂G∂U = true,
                    compute_residuals = true)
    end
    compute∂G∂p = (out, u, p, t, i, it) -> begin
        compute∂G∂_(out, u, p, t, i, it,
                    measurement_info, parameter_info,
                    θ_indices, petab_model,
                    xnoise, xobservable, xnondynamic,
                    cache.∂h∂p, cache.∂σ∂p, compute∂G∂U = false,
                    compute_residuals = true)
    end

    # Extract relevant parameters for the experimental conditions
    map_condition_id = θ_indices.maps_conidition_id[simulation_condition_id]
    iθ_experimental_condition = vcat(θ_indices.map_ode_problem.sys_to_dynamic,
                                     map_condition_id.ix_dynamic)

    # Loop through solution and extract sensitivites
    n_model_states = length(states(petab_model.sys_mutated))
    cache.p .= dual_to_float.(sol.prob.p)
    p = cache.p
    u = cache.u
    ∂G∂p = cache.∂G∂p
    ∂G∂u = cache.∂G∂u
    _gradient = cache.forward_eqs_grad
    for i in eachindex(time_observed)
        u .= dual_to_float.((@view sol[:, i]))
        t = time_observed[i]
        i_start, i_end = (time_position_ode_sol[i] - 1) * n_model_states + 1,
                         (time_position_ode_sol[i] - 1) * n_model_states + n_model_states
        _S = @view cache.S[i_start:i_end, iθ_experimental_condition]
        for i_measurement in imeasurements_t[i]
            compute∂G∂u(∂G∂u, u, p, t, 1, [[i_measurement]])
            compute∂G∂p(∂G∂p, u, p, t, 1, [[i_measurement]])
            @views _gradient[iθ_experimental_condition] .= transpose(_S) * ∂G∂u

            # Thus far have have computed dY/dθ for the residuals, but for parameters on the log-scale we want dY/dθ_log.
            # We can adjust via; dY/dθ_log = log(10) * θ * dY/dθ
            adjust_gradient_θ_transformed!((@view jacobian[:, i_measurement]), _gradient,
                                           ∂G∂p, xdynamic, θ_indices,
                                           simulation_condition_id,
                                           autodiff_sensitivites = true)
        end
    end
    return nothing
end

# To compute the gradient for non-dynamic parameters
function compute_residuals_not_solve_ode!(residuals::T1,
                                          xnoise::T2,
                                          xobservable::T2,
                                          xnondynamic::T2,
                                          probleminfo::PEtabODEProblemInfo,
                                          model_info::ModelInfo;
                                          exp_id_solve::Vector{Symbol} = [:all])::T1 where {
                                                                                            T1 <:
                                                                                            AbstractVector,
                                                                                            T2 <:
                                                                                            AbstractVector
                                                                                            }
    @unpack petab_model, simulation_info, θ_indices, measurement_info, parameter_info = model_info
    @unpack cache = probleminfo
    # To be able to use ReverseDiff sdParamEstUse and obsParamEstUse cannot be overwritten.
    # Hence new vectors have to be created.
    xnoise_ps = PEtab.transform_x(xnoise, θ_indices, :xnoise, cache)
    xobservable_ps = PEtab.transform_x(xobservable, θ_indices, :xobservable, cache)
    xnondynamic_ps = PEtab.transform_x(xnondynamic, θ_indices, :xnondynamic, cache)

    # Compute residuals per experimental conditions
    for experimental_condition_id in simulation_info.conditionids[:experiment]
        if exp_id_solve[1] != :all && experimental_condition_id ∉ exp_id_solve
            continue
        end

        ode_sol = simulation_info.odesols_derivatives[experimental_condition_id]
        sucess = compute_residuals_condition!(residuals, ode_sol, xnoise_ps, xobservable_ps,
                                              xnondynamic_ps,
                                              petab_model, experimental_condition_id,
                                              simulation_info, θ_indices,
                                              measurement_info, parameter_info,
                                              cache)
        if sucess == false
            residuals .= Inf
            break
        end
    end

    return residuals
end

# For an experimental condition compute residuals
function compute_residuals_condition!(residuals::T1,
                                      ode_sol::ODESolution,
                                      xnoise::T2,
                                      xobservable::T2,
                                      xnondynamic::T2,
                                      petab_model::PEtabModel,
                                      experimental_condition_id::Symbol,
                                      simulation_info::SimulationInfo,
                                      θ_indices::ParameterIndices,
                                      measurement_info::MeasurementsInfo,
                                      parameter_info::ParametersInfo,
                                      cache::PEtabODEProblemCache)::Bool where {
                                                                                          T1 <:
                                                                                          AbstractVector,
                                                                                          T2 <:
                                                                                          AbstractVector
                                                                                          }
    if !(ode_sol.retcode == ReturnCode.Success || ode_sol.retcode == ReturnCode.Terminated)
        return false
    end

    @unpack u, p = cache
    # Compute y_model and sd for all observations having id conditionID
    for i_measurement in simulation_info.imeasurements[experimental_condition_id]
        t = measurement_info.time[i_measurement]
        u .= dual_to_float.((@view ode_sol[:,
                                           simulation_info.imeasurements_t_sol[i_measurement]]))
        p .= dual_to_float.(ode_sol.prob.p)
        hT = computehT(u, t, p, xobservable, xnondynamic, petab_model, i_measurement,
                       measurement_info, θ_indices, parameter_info)
        σ = computeσ(u, t, p, xnoise, xnondynamic, petab_model, i_measurement,
                     measurement_info, θ_indices, parameter_info)

        # By default a positive ODE solution is not enforced (even though the user can provide it as option).
        # In case with transformations on the data the code can crash, hence Inf is returned in case the
        # model data transformation can not be perfomred.
        if isinf(hT)
            return false
        end

        if measurement_info.measurement_transforms[i_measurement] == :lin
            residuals[i_measurement] = (hT - measurement_info.measurement[i_measurement]) /
                                       σ
        elseif measurement_info.measurement_transforms[i_measurement] == :log10
            residuals[i_measurement] = (hT - measurement_info.measurementT[i_measurement]) /
                                       σ
        elseif measurement_info.measurement_transforms[i_measurement] == :log
            residuals[i_measurement] = (hT - measurement_info.measurementT[i_measurement]) /
                                       σ
        else
            @error "Transformation ",
                   measurement_info.measurement_transforms[i_measurement],
                   " not yet supported."
        end
    end
    return true
end
