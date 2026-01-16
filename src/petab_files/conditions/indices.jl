function _get_indices_est(
        xids::Dict{Symbol, Vector{Symbol}}, ml_models::MLModels
    )::Dict{Symbol, Vector{Int32}}
    # Mechanistic parameters
    xi_dynamic_mech = _get_indices(xids[:est_to_dynamic_mech], xids[:estimate])
    xi_noise = _get_indices(xids[:noise], xids[:estimate])
    xi_observable = _get_indices(xids[:observable], xids[:estimate])
    xi_nondynamic_mech = _get_indices(xids[:nondynamic_mech], xids[:estimate])
    xi_not_system_mech = _get_indices(xids[:not_system_mech], xids[:estimate])
    xi_est = Dict(
        :est_to_dynamic_mech => xi_dynamic_mech, :est_to_noise => xi_noise,
        :est_to_observable => xi_observable, :est_to_nondynamic_mech => xi_nondynamic_mech,
        :est_to_not_system_mech => xi_not_system_mech
    )

    # ML-models. Each model gets its own key
    i_start = length(xids[:estimate]) - length(xids[:ml_est])
    for ml_id in xids[:ml_est]
        xi_est[ml_id] = _get_indices_ml_model(i_start, ml_models[ml_id])
        i_start = xi_est[ml_id][end]
    end

    # All dynamic (mechanistic + ML)
    xi_est_to_dynamic = _get_indices(xids[:est_to_dynamic_mech], xids[:estimate])
    for ml_id in xids[:ml_est]
        !in(ml_id, xids[:ml_est]) && continue
        xi_est_to_dynamic = vcat(xi_est_to_dynamic, xi_est[ml_id])
    end
    xi_est[:est_to_dynamic] = xi_est_to_dynamic

    # All (mechanistic + ML) not part of ODE-system
    xi_not_system = deepcopy(xi_not_system_mech)
    for ml_id in xids[:ml_nondynamic]
        !in(ml_id, xids[:ml_est]) && continue
        xi_not_system = vcat(xi_not_system, xi_est[ml_id])
    end
    xi_est[:est_to_not_system] = xi_not_system
    return xi_est
end

function _get_indices_dynamic(
        xids::Dict{Symbol, Vector{Symbol}}, ml_models::MLModels
    )::Dict{Symbol, Vector{Int32}}
    indices_dynamic = Dict{Symbol, Vector{Int32}}()

    # Mechanistic parameters
    xi_dynamic_to_mech = Int32.(1:length(xids[:est_to_dynamic_mech]))

    # ML models pre- and in ML-model
    xi_ml_ode = Int32[]
    xi_ml_pre_simulate = Int32[]
    i_start = length(xids[:estimate]) - length(xids[:ml_est])
    for ml_id in xids[:ml_est]
        xi = _get_indices_ml_model(i_start, ml_models[ml_id])
        i_start = xi[end]
        ml_id in xids[:ml_nondynamic] && continue

        if ml_id in xids[:ml_pre_simulate]
            xi_ml_pre_simulate = vcat(xi_ml_pre_simulate, xi)
        elseif ml_id in xids[:ml_in_ode]
            xi_ml_ode = vcat(xi_ml_ode, xi)
        end
    end

    indices_dynamic = Dict{Symbol, Vector{Int32}}(
        :dynamic_to_mech => xi_dynamic_to_mech, :dynamic_to_ml_sys => xi_ml_ode,
        :dynamic_to_ml_pre_simulate => xi_ml_pre_simulate
    )

    # Individual indices for each ML-model
    ml_ids = Iterators.flatten((xids[:ml_in_ode], xids[:ml_pre_simulate]))
    i_start = length(indices_dynamic[:dynamic_to_mech])
    for ml_id in xids[:ml_est]
        !in(ml_id, ml_ids) && continue
        indices_dynamic[ml_id] = _get_indices_ml_model(i_start, ml_models[ml_id])
        i_start = indices_dynamic[ml_id][end]
    end

    # Indices for parameters which values are set by a ML-model
    ix_sys_ml_pre_simulate_outputs = Int32[]
    for output_id in xids[:sys_ml_pre_simulate_outputs]
        isys = 1
        for id_sys in xids[:sys]
            if id_sys in xids[:ml_est]
                isys += _get_n_ml_parameters(ml_models[id_sys])
                continue
            end
            if id_sys == output_id
                push!(ix_sys_ml_pre_simulate_outputs, isys)
                break
            end
            isys += 1
        end
    end
    indices_dynamic[:sys_ml_pre_simulate_outputs] = ix_sys_ml_pre_simulate_outputs

    # For mapping ODEProblem ML parameters to xdynamic
    isys = 0
    xi_sys_to_dynamic_ml = Int32[]
    for id_sys in xids[:sys]
        if id_sys in xids[:ml_est]
            xi_sys_to_dynamic_ml = vcat(
                xi_sys_to_dynamic_ml, _get_indices_ml_model(isys, ml_models[id_sys])
            )
            isys = xi_sys_to_dynamic_ml[end]
            continue
        end
        isys += 1
    end
    indices_dynamic[:sys_to_dynamic_ml] = xi_sys_to_dynamic_ml
    return indices_dynamic
end

function _get_indices_not_system(
        xids::Dict{Symbol, Vector{Symbol}}, ml_models::MLModels
    )::Dict{Symbol, Vector{Int32}}
    # Mechanistic parameters
    ix_noise = _get_indices(xids[:noise], xids[:not_system_mech])
    ix_observable = _get_indices(xids[:observable], xids[:not_system_mech])
    ix_nondynamic_mech = _get_indices(xids[:nondynamic_mech], xids[:not_system_mech])
    indices_not_system = Dict(
        :not_system_to_noise => ix_noise, :not_system_to_observable => ix_observable,
        :not_system_to_nondynamic_mech => ix_nondynamic_mech
    )

    # ML parameters (belong to the class of non-dynamic)
    i_start = length(xids[:not_system_mech])
    for ml_id in xids[:ml_nondynamic]
        indices_not_system[ml_id] = _get_indices_ml_model(i_start, ml_models[ml_id])
        i_start = indices_not_system[ml_id][end]
    end
    return indices_not_system
end
