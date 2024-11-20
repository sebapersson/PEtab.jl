function _net!(out, x, pnn::DiffCache, inputs::DiffCache, map_nn::NNPreODEMap, nn)::Nothing
    _inputs = get_tmp(inputs, x)
    _pnn = get_tmp(pnn, x)
    _inputs[map_nn.iconstant_inputs] .= map_nn.constant_inputs
    @views _inputs[map_nn.ixdynamic_inputs] .= x[1:map_nn.nxdynamic_inputs]
    @views _pnn .= x[(map_nn.nxdynamic_inputs + 1):end]
    st, net = nn
    _out, st = net(_inputs, _pnn, st)
    out .= _out
    nn[1] = st
    return nothing
end

function _jac_nn_preode!(probinfo::PEtabODEProblemInfo, model_info::ModelInfo)::Nothing
    @unpack cache = probinfo
    for (cid, f_nns_preode) in probinfo.f_nns_preode
        for (netid, f_nn_preode) in f_nns_preode
            # Parameter mapping. If one of the inputs is a parameter to estimate, the
            # Jacobian is also computed of the input parameter.
            map_nn = model_info.xindices.maps_nn_preode[cid][netid]
            pnn = get_tmp(cache.xnn[netid], 1.0)
            xdynamic_mech = get_tmp(cache.xdynamic_mech, 1.0)
            xdynamic_mech_ps = transform_x(xdynamic_mech, model_info.xindices, :xdynamic, cache)
            _x = _get_f_nn_preode_x(f_nn_preode, xdynamic_mech_ps, pnn, map_nn)

            @unpack tape, jac_nn, outputs, inputs, computed = f_nn_preode
            ReverseDiff.jacobian!(jac_nn, tape, _x)
            outputs_tape = ReverseDiff.value(tape.output)
            outputs_cache = get_tmp(outputs, outputs_tape)
            outputs_cache .= outputs_tape
            computed[1] = true
        end
    end
    return nothing
end

function _set_grad_x_nn_preode!(xdynamic_grad::AbstractVector, simid::Symbol, probinfo::PEtabODEProblemInfo, model_info::ModelInfo)::Nothing
    isempty(probinfo.f_nns_preode) && return nothing
    @unpack xindices_dynamic, maps_nn_preode = model_info.xindices
    for (netid, f_nn_preode) in probinfo.f_nns_preode[simid]
        map_nn = maps_nn_preode[simid][netid]
        grad_nn_output = probinfo.cache.grad_nn_preode[map_nn.ix_nn_outputs]
        ix = Iterators.flatten((map_nn.ixdynamic_mech_inputs, xindices_dynamic[netid])) |>
            collect
        xdynamic_grad[ix] .+= vec(grad_nn_output' * f_nn_preode.jac_nn)
    end
    return nothing
end

function _reset_nn_preode!(probinfo::PEtabODEProblemInfo)::Nothing
    for f_nns_preode in values(probinfo.f_nns_preode)
        for f_nn_preode in values(f_nns_preode)
            f_nn_preode.computed[1] = false
        end
    end
    return nothing
end
