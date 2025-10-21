function _net!(out, x, x_ml_model::DiffCache, inputs::Vector{<:DiffCache}, map_ml_model::MLModelPreODEMap, ml_model::MLModel)::Nothing
    _x_ml_model = get_tmp(x_ml_model, x)
    _x_ml_model .= x[(sum(map_ml_model.nxdynamic_inputs) + 1):end]
    _inputs = (_get_input(inputs[i], x, map_ml_model, i) for i in eachindex(inputs))
    _inputs = length(inputs) == 1 ? first(_inputs) : Tuple(_inputs)
    _out, st = ml_model.model(_inputs, _x_ml_model, ml_model.st)
    ml_model.st = st
    out .= _out
    return nothing
end
function _net!(out, x, x_ml_model::ComponentArray, inputs::Vector{<:DiffCache}, map_ml_model::MLModelPreODEMap, ml_model::MLModel)::Nothing
    _inputs = (_get_input(inputs[i], x, map_ml_model, i) for i in eachindex(inputs))
    _inputs = length(inputs) == 1 ? first(_inputs) : Tuple(_inputs)
    _out, st = ml_model.model(_inputs, x_ml_model, ml_model.st)
    ml_model.st = st
    out .= _out
    return nothing
end

# For ReverseDiff the tape forward-pass fails with get_tmp. get_tmp is only needed for
# ForwardDiff, and when the input depends on a parameter to estimate. If none of the
# inputs are estimated, that is, are known at compile-time it is possible to compile the
# function and enjoy good performance on the CPU. If one of the inputs depend on parameters
# to estimate, ForwardDiff can be used instead (and hopefully in the future Enzyme can
# make all this code obselete)
function _net_reversediff!(out, x_ml_model, inputs, ml_model::MLModel)::Nothing
    _out, st = ml_model.model(inputs, x_ml_model, ml_model.st)
    ml_model.st = st
    out .= _out
    return nothing
end

function _get_input(map_ml_model::MLModelPreODEMap, iarg::Int64)
    if map_ml_model.file_input[iarg] == false
        input = map_ml_model.constant_inputs[iarg][map_ml_model.iconstant_inputs[iarg]]
    else
        input = map_ml_model.constant_inputs[iarg]
    end
    return input
end
function _get_input(inputs::DiffCache, x, map_ml_model::MLModelPreODEMap, iarg::Int64)
    if map_ml_model.file_input[iarg] == false
        _inputs = get_tmp(inputs, x)
        _inputs[map_ml_model.iconstant_inputs[iarg]] .= map_ml_model.constant_inputs[iarg]
        _inputs[map_ml_model.ixdynamic_inputs[iarg]] .= x[1:map_ml_model.nxdynamic_inputs[iarg]]
    else
        _inputs = convert.(eltype(x), map_ml_model.constant_inputs[iarg])
    end
    return _inputs
end

function _jac_ml_model_preode!(probinfo::PEtabODEProblemInfo, model_info::ModelInfo)::Nothing
    @unpack cache = probinfo
    for (cid, ml_models_pre_ode) in probinfo.ml_models_pre_ode
        for (ml_model_id, ml_model_pre_ode) in ml_models_pre_ode
            # Only relevant if neural-network parameters are estimated, otherwise a normal
            # reverse pass of the code is fast
            !haskey(cache.xnn, ml_model_id) && continue

            @unpack tape, jac_ml_model, outputs, computed, forward! = ml_model_pre_ode
            # Parameter mapping. If one of the inputs is a parameter to estimate, the
            # Jacobian is also computed of the input parameter.
            map_ml_model = model_info.xindices.maps_nn_preode[cid][ml_model_id]
            x_ml_model = get_tmp(cache.xnn[ml_model_id], 1.0)

            _outputs = get_tmp(outputs, x_ml_model)
            if sum(map_ml_model.nxdynamic_inputs) > 0
                xdynamic_mech = get_tmp(cache.xdynamic_mech, 1.0)
                xdynamic_mech_ps = transform_x(xdynamic_mech, model_info.xindices, :xdynamic_mech, cache)
                _x = _get_ml_model_pre_ode_x(ml_model_pre_ode, xdynamic_mech_ps, x_ml_model, map_ml_model)
                ForwardDiff.jacobian!(jac_ml_model, forward!, _outputs, _x)
            else
                ReverseDiff.jacobian!(jac_ml_model, tape, x_ml_model)
                _outputs .= ReverseDiff.value(tape.output)
            end
            computed[1] = true
        end
    end
    return nothing
end

function _set_grad_x_nn_preode!(xdynamic_grad::AbstractVector, simid::Symbol, probinfo::PEtabODEProblemInfo, model_info::ModelInfo)::Nothing
    isempty(probinfo.ml_models_pre_ode) && return nothing
    @unpack xindices_dynamic, maps_nn_preode = model_info.xindices
    for (ml_model_id, ml_model_pre_ode) in probinfo.ml_models_pre_ode[simid]
        map_ml_model = maps_nn_preode[simid][ml_model_id]
        grad_nn_output = probinfo.cache.grad_nn_preode[map_ml_model.ix_nn_outputs]
        # Needed to account for neural-net parameter potentially not being estimated
        ix = reduce(vcat, map_ml_model.ixdynamic_mech_inputs)
        if haskey(xindices_dynamic, ml_model_id)
            ix = vcat(ix, xindices_dynamic[ml_model_id])
        end
        xdynamic_grad[collect(ix)] .+= vec(grad_nn_output' * ml_model_pre_ode.jac_ml_model)
    end
    return nothing
end

function _reset_nn_preode!(probinfo::PEtabODEProblemInfo)::Nothing
    for ml_models_pre_ode in values(probinfo.ml_models_pre_ode)
        for ml_model_pre_ode in values(ml_models_pre_ode)
            ml_model_pre_ode.computed[1] = false
        end
    end
    return nothing
end
