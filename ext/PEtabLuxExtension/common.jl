function PEtab._reshape_array(x, mapping)
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

function PEtab._setup_nnmodels(nnmodels::Dict{Symbol, <:NNModel})::Dict
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
        _pnn, _ = Lux.setup(rng, nnmodel.nn)
        pnns[netid] = _pnn
    end
    return pnns
end

function PEtab._get_n_net_parameters(nnmodel::PEtab.NNModel)
    return Lux.LuxCore.parameterlength(nnmodel.nn)
end

function PEtab._get_nn_initialparameters(nnmodel::PEtab.NNModel)::ComponentArray
    rng = Random.default_rng(1)
    return Lux.initialparameters(rng, nnmodel.nn) |> ComponentArray .|> Float64
end

function PEtab._reshape_io_data(x::Array{T})::Array{T} where T <: AbstractFloat
    order_py = 1:length(size(x))
    order_jl = reverse(order_py)
    imap = zeros(Int64, length(order_jl))
    for i in eachindex(order_jl)
        imap[i] = findfirst(x -> x == order_jl[i], order_py)
    end
    map = collect(1:length(order_py)) .=> imap
    return PEtab._reshape_array(x, map)
end
