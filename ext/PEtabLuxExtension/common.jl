function _reshape_array(x, mapping)
    dims_out = size(x)[last.(mapping)]
    xout = reshape(deepcopy(x), dims_out)
    for i in eachindex(Base.CartesianIndices(x))
        inew = zeros(Int64, length(i.I))
        for j in eachindex(i.I)
            inew[j] = i.I[mapping[j].second]
        end
        xout[inew...] = x[i]
    end
    return xout
end

function PEtab._setup_nnmodels(nnmodels::Dict)::Dict
    rng = Random.default_rng()
    nn = Dict()
    for (netid, nnmodel) in nnmodels
        _, _st = Lux.setup(rng, nnmodel[:net])
        nn[netid] = [_st, nnmodel[:net]]
    end
    return nn
end

function PEtab._get_pnns(nnmodels_in_ode::Dict)::Dict{Symbol, NamedTuple}
    rng = Random.default_rng()
    pnns = Dict()
    for (netid, nnmodel) in nnmodels_in_ode
        _pnn, _ = Lux.setup(rng, nnmodel[:net])
        pnns[Symbol("p_$netid")] = _pnn
    end
    return pnns
end

function PEtab._get_n_net_parameters(net)
    return Lux.LuxCore.parameterlength(net)
end

function PEtab._get_nn_initialparameters(net)::ComponentArray
    rng = Random.default_rng(1)
    return Lux.initialparameters(rng, net) |> ComponentArray .|> Float64
end
