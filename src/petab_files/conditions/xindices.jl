function _get_xindices_xest(xids::Dict{Symbol, Vector{Symbol}}, ml_models::MLModels)::Dict{Symbol, Vector{Int32}}
    # For mapping mechanistic parameters from xest to respective subsets
    xi_dynamic_mech = _get_xindices(xids[:dynamic_mech], xids[:estimate])
    xi_noise = _get_xindices(xids[:noise], xids[:estimate])
    xi_observable = _get_xindices(xids[:observable], xids[:estimate])
    xi_nondynamic = _get_xindices(xids[:nondynamic_mech], xids[:estimate])
    xi_not_system_mech = _get_xindices(xids[:not_system_mech], xids[:estimate])
    xindices_est = Dict(:dynamic_mech => xi_dynamic_mech, :noise => xi_noise, :observable => xi_observable, :nondynamic_mech => xi_nondynamic, :not_system_mech => xi_not_system_mech)

    # Indices for each neural network. Each neural-net gets its own key in xindices_est
    istart = length(xids[:estimate]) - length(xids[:ml_est])
    for pid in xids[:ml_est]
        xindices_est[pid] = _get_xindices_net(pid, istart, ml_models)
        istart = xindices_est[pid][end]
    end
    # As above, also track not part of ODESystem
    xi_not_system_tot = deepcopy(xi_not_system_mech)
    for ml_id in xids[:ml_nondynamic]
        !(ml_id in xids[:ml_est]) && continue
        xi_not_system_tot = vcat(xi_not_system_tot, xindices_est[ml_id])
    end
    xindices_est[:not_system_tot] = xi_not_system_tot
    return xindices_est
end

function _get_xindices_dynamic(xids::Dict{Symbol, Vector{Symbol}}, ml_models::MLModels)::Dict{Symbol, Vector{Int32}}
    xindices = Dict{Symbol, Vector{Int32}}()
    # Mechanistic parameters
    xindices[:xdynamic_to_mech] = Int32.(1:length(xids[:dynamic_mech]))

    # Mechanistic + neural net parameters, for each neural category
    xi_xest_to_xdynamic = _get_xindices(xids[:dynamic_mech], xids[:estimate])
    xi_nn_in_ode, xi_nn_preode = Int32[], Int32[]
    istart = length(xids[:estimate]) - length(xids[:ml_est])
    # Need to loop of keys(nn) to get consistent mapping when accessing for example xest
    # and xdynamic
    for pid in xids[:ml_est]
        xn = _get_xindices_net(pid, istart, ml_models)
        istart = xn[end]
        pid in xids[:ml_nondynamic] && continue
        if pid in xids[:ml_preode]
            xi_nn_preode = vcat(xi_nn_preode, xn)
        elseif pid in xids[:ml_in_ode]
            xi_nn_in_ode = vcat(xi_nn_in_ode, xn)
        end
        xi_xest_to_xdynamic = vcat(xi_xest_to_xdynamic, xn)
    end
    xindices[:nn_in_ode] = xi_nn_in_ode
    xindices[:nn_preode] = xi_nn_preode
    xindices[:xest_to_xdynamic] = xi_xest_to_xdynamic

    # Get indices in xdynamic for each neural net in xdynamic
    istart = length(xindices[:xdynamic_to_mech])
    for pid in xids[:ml_est]
        !(pid in Iterators.flatten((xids[:ml_in_ode], xids[:ml_preode]))) && continue
        xindices[pid] = _get_xindices_net(pid, istart, ml_models)
        istart = xindices[pid][end]
    end

    # Indices for extracting which parameters are given by a ML model
    ix_sys_nn_preode_output = Int32[]
    for output_id in xids[:ml_preode_outputs]
        isys = 1
        for id_sys in xids[:sys]
            if id_sys in xids[:ml_est]
                isys += _get_n_ml_model_parameters(ml_models, [id_sys])
                continue
            end
            if id_sys == output_id
                push!(ix_sys_nn_preode_output, isys)
                break
            end
            isys += 1
        end
    end
    xindices[:ml_preode_outputs] = ix_sys_nn_preode_output

    # Indices for mapping ODEProblem.p ML parameters to xdynamic
    isys = 0
    xi_sys_to_dynamic_nn = Int32[]
    for id_sys in xids[:sys]
        if id_sys in xids[:ml_est]
            xi_sys_to_dynamic_nn = vcat(xi_sys_to_dynamic_nn, _get_xindices_net(id_sys, isys, ml_models))
            isys = xi_sys_to_dynamic_nn[end]
            continue
        end
        isys += 1
    end
    xindices[:sys_to_dynamic_nn] = xi_sys_to_dynamic_nn

    # nn_preode outputs are for effiency added to xdynamic, and these parameters are
    # sent to the end of the vector
    np = length(xindices[:xest_to_xdynamic])
    xindices[:xdynamic_to_nnout] = (np+1):(np + length(xids[:ml_preode_outputs]))
    return xindices
end

function _get_xindices_notsys(xids::Dict{Symbol, Vector{Symbol}}, ml_models::Union{Nothing, MLModels})::Dict{Symbol, Vector{Int32}}
    # Mechanistic parameters
    ixnoise = _get_xindices(xids[:noise], xids[:not_system_mech])
    ixobservable = _get_xindices(xids[:observable], xids[:not_system_mech])
    ixnondynamic_mech = _get_xindices(xids[:nondynamic_mech], xids[:not_system_mech])
    xindices_notsys = Dict(:noise => ixnoise, :observable => ixobservable, :nondynamic_mech => ixnondynamic_mech)
    # Neural net parameters
    istart = length(xids[:not_system_mech])
    for pid in xids[:ml_nondynamic]
        xindices_notsys[pid] = _get_xindices_net(pid, istart, ml_models)
        istart = xindices_notsys[pid][end]
    end
    return xindices_notsys
end
