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

function PEtab._setup_ml_models(ml_models::PEtab.MLModels)::Dict
    rng = Random.default_rng()
    nn = Dict()
    for (ml_id, ml_model) in ml_models
        _, _st = Lux.setup(rng, ml_model[:net])
        nn[ml_id] = [_st, ml_model[:net]]
    end
    return nn
end

function PEtab._get_ml_model_ps(ml_models_in_ode::Dict)::Dict{Symbol, NamedTuple}
    rng = Random.default_rng()
    pnns = Dict()
    for (ml_id, ml_model) in ml_models_in_ode
        _pnn, _ = Lux.setup(rng, ml_model.model)
        pnns[ml_id] = _pnn
    end
    return pnns
end

function PEtab._get_n_ml_model_parameters(ml_model::PEtab.MLModel)
    return Lux.LuxCore.parameterlength(ml_model.model)
end

function PEtab._get_ml_model_initialparameters(ml_model::PEtab.MLModel)::ComponentArray
    rng = Random.default_rng(1)
    return Lux.initialparameters(rng, ml_model.model) |> ComponentArray .|> Float64
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
