"""Prepare neural network parameter vector."""
function _prepare_net_params(param_cache::DiffCache, x, map_nn::NNPreODEMap)
    tmp_params = get_tmp(param_cache, x)
    tmp_params .= x[(map_nn.nxdynamic_inputs + 1):end]
    return tmp_params
end

_prepare_net_params(params::ComponentArray, x, map_nn::NNPreODEMap) = params

"""Assemble neural network input vector."""
function _prepare_net_inputs(map_nn::NNPreODEMap, x, input_cache::DiffCache)
    if map_nn.file_input == false
        tmp_inputs = get_tmp(input_cache, x)
        tmp_inputs[map_nn.iconstant_inputs] .= map_nn.constant_inputs
        tmp_inputs[map_nn.ixdynamic_inputs] .= x[1:map_nn.nxdynamic_inputs]
    else
        tmp_inputs = convert.(eltype(x), map_nn.constant_inputs)
    end
    return tmp_inputs
end

function _net!(output, x_vec, net_params, input_cache::DiffCache,
               map_nn::NNPreODEMap, nnmodel::NNModel)::Nothing
    params = _prepare_net_params(net_params, x_vec, map_nn)
    inputs = _prepare_net_inputs(map_nn, x_vec, input_cache)

    net_out, new_state = nnmodel.nn(inputs, params, nnmodel.st)
    nnmodel.st = new_state
    output .= net_out
    return nothing
end

# For ReverseDiff the tape forward-pass fails with get_tmp. get_tmp is only needed for
# ForwardDiff, and when the input depends on a parameter to estimate. If none of the
# inputs are estimated, that is, are known at compile-time it is possible to compile the
# function and enjoy good performance on the CPU. If one of the inputs depend on parameters
# to estimate, ForwardDiff can be used instead (and hopefully in the future Enzyme can
# make all this code obselete)
function _net_reversediff!(out, pnn, inputs::Array{<:AbstractFloat}, nnmodel::NNModel)::Nothing
    _out, st = nnmodel.nn(inputs, pnn, nnmodel.st)
    nnmodel.st = st
    out .= _out
    return nothing
end

function _jac_nn_preode!(probinfo::PEtabODEProblemInfo, model_info::ModelInfo)::Nothing
    @unpack cache = probinfo
    for (cid, net_functions) in probinfo.f_nns_preode
        for (netid, net_function) in net_functions
            # Only relevant if neural-network parameters are estimated, otherwise a normal
            # reverse pass of the code is fast
            !haskey(cache.xnn, netid) && continue

            @unpack tape, jac_nn, outputs, computed, nn! = net_function
            # Parameter mapping. If one of the inputs is a parameter to estimate, the
            # Jacobian is also computed of the input parameter.
            map_nn = model_info.xindices.maps_nn_preode[cid][netid]
            pnn = get_tmp(cache.xnn[netid], 1.0)

            _outputs = get_tmp(outputs, pnn)
            if map_nn.nxdynamic_inputs > 0
                xdynamic_mech = get_tmp(cache.xdynamic_mech, 1.0)
                xdynamic_mech_ps = transform_x(xdynamic_mech, model_info.xindices, :xdynamic_mech, cache)
                _x = _get_f_nn_preode_x(net_function, xdynamic_mech_ps, pnn, map_nn)
                ForwardDiff.jacobian!(jac_nn, nn!, _outputs, _x)
            else
                ReverseDiff.jacobian!(jac_nn, tape, pnn)
                _outputs .= ReverseDiff.value(tape.output)
            end
            computed[1] = true
        end
    end
    return nothing
end

function _set_grad_x_nn_preode!(xdynamic_grad::AbstractVector, simid::Symbol,
                                probinfo::PEtabODEProblemInfo,
                                model_info::ModelInfo)::Nothing
    isempty(probinfo.f_nns_preode) && return nothing
    @unpack xindices_dynamic, maps_nn_preode = model_info.xindices
    for (netid, net_function) in probinfo.f_nns_preode[simid]
        nn_map = maps_nn_preode[simid][netid]
        output_grad = probinfo.cache.grad_nn_preode[nn_map.ix_nn_outputs]
        # Needed to account for neural-net parameter potentially not being estimated
        if haskey(xindices_dynamic, netid)
            ix = Iterators.flatten((nn_map.ixdynamic_mech_inputs, xindices_dynamic[netid])) |
                collect
        else
            ix = nn_map.ixdynamic_mech_inputs
        end
        xdynamic_grad[ix] .+= vec(output_grad' * net_function.jac_nn)
    end
    return nothing
end

function _reset_nn_preode!(probinfo::PEtabODEProblemInfo)::Nothing
    for net_functions in values(probinfo.f_nns_preode)
        for net_function in values(net_functions)
            net_function.computed[1] = false
        end
    end
    return nothing
end
