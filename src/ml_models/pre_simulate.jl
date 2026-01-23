# TODO: Should be doable to do a fast function here is input is known constant
function _net!(out, x, x_ml_cache::DiffCache, map_ml_model::MLModelPreSimulateMap, ml_model::MLModel)::Nothing
    x_ml = get_tmp(x_ml_cache, x)
    x_ml .= x[(length(map_ml_model.ix_dynamic_mech) + 1):end]
    inputs = map_ml_model.get_input(x, map_ml_model)
    ml_out, st = ml_model.lux_model(inputs, x_ml, ml_model.st)
    ml_model.st = st
    out .= ml_out
    return nothing
end
function _net!(out, x, x_ml::ComponentArray, map_ml_model::MLModelPreSimulateMap, ml_model::MLModel)::Nothing
    inputs = map_ml_model.get_input(x, map_ml_model)
    ml_out, st = ml_model.lux_model(inputs, x_ml, ml_model.st)
    ml_model.st = st
    out .= ml_out
    return nothing
end

# For ReverseDiff the tape forward-pass fails with get_tmp. get_tmp is only needed for
# ForwardDiff, and when the input depends on a parameter to estimate. If none of the
# inputs are estimated, that is, are known at compile-time it is possible to compile the
# function and enjoy good performance on the CPU. If one of the inputs depend on parameters
# to estimate, ForwardDiff can be used instead (and hopefully in the future Enzyme can
# make all this code obsolete)
function _net_reversediff!(out, x_ml, inputs, ml_model::MLModel)::Nothing
    _out, st = ml_model.lux_model(inputs, x_ml, ml_model.st)
    ml_model.st = st
    out .= _out
    return nothing
end

function _jac_ml_model_pre_simulate!(probinfo::PEtabODEProblemInfo, model_info::ModelInfo)::Nothing
    @unpack cache = probinfo
    for (cid, ml_models_pre_ode) in probinfo.ml_models_pre_ode
        for (ml_id, ml_model_pre_ode) in ml_models_pre_ode
            # Only relevant if neural-network parameters are estimated, otherwise a normal
            # reverse pass of the code is fast
            !haskey(cache.x_ml_models, ml_id) && continue

            @unpack tape, jac_ml_model, outputs, computed, forward! = ml_model_pre_ode
            # Parameter mapping. If one of the inputs is a parameter to estimate, the
            # Jacobian is also computed of the input parameter.
            map_ml_model = model_info.xindices.maps_ml_pre_simulate[cid][ml_id]
            x_ml = get_tmp(cache.x_ml_models_cache[ml_id], 1.0)

            _outputs = get_tmp(outputs, x_ml)
            if !isempty(map_ml_model.ix_dynamic_mech)
                xdynamic_mech = get_tmp(cache.xdynamic_mech, 1.0)
                xdynamic_mech_ps = transform_x(xdynamic_mech, model_info.xindices, :xdynamic_mech, cache)
                x = _get_ml_model_pre_ode_x(ml_model_pre_ode, xdynamic_mech_ps, x_ml, map_ml_model)
                ForwardDiff.jacobian!(jac_ml_model, forward!, _outputs, x)
            else
                ReverseDiff.jacobian!(jac_ml_model, tape, x_ml)
                _outputs .= ReverseDiff.value(tape.output)
            end
            computed[1] = true
        end
    end
    return nothing
end

function _set_grax_x_ml_pre_simulate!(xdynamic_grad::AbstractVector, simid::Symbol, probinfo::PEtabODEProblemInfo, model_info::ModelInfo)::Nothing
    isempty(probinfo.ml_models_pre_ode) && return nothing

    @unpack indices_dynamic, maps_ml_pre_simulate = model_info.xindices
    for (ml_id, ml_model_pre_ode) in probinfo.ml_models_pre_ode[simid]
        map_ml_model = maps_ml_pre_simulate[simid][ml_id]
        grad_output = probinfo.cache.grad_ml_pre_simulate_outputs[map_ml_model.ix_ml_outputs]

        # Needed to account for neural-net parameter potentially not being estimated
        ix = map_ml_model.ix_dynamic_mech
        if haskey(indices_dynamic, ml_id)
            ix = vcat(ix, indices_dynamic[ml_id])
        end
        xdynamic_grad[collect(ix)] .+= vec(grad_output' * ml_model_pre_ode.jac_ml_model)
    end
    return nothing
end


function reset_ml_pre_simulate!(probinfo::PEtabODEProblemInfo)::Nothing
    for ml_models_pre_ode in values(probinfo.ml_models_pre_ode)
        for ml_model_pre_ode in values(ml_models_pre_ode)
            ml_model_pre_ode.computed[1] = false
        end
    end
    return nothing
end
