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

function PEtab._get_lux_ps(ml_model::PEtab.MLModel)
    return PEtab._get_lux_ps(Random.default_rng(), ml_model)
end
function PEtab._get_lux_ps(rng::Random.AbstractRNG, ml_model::PEtab.MLModel)
    ps, _ = Lux.setup(rng, ml_model.lux_model)
    return ps
end

function PEtab._get_n_ml_parameters(ml_model::PEtab.MLModel)
    return Lux.LuxCore.parameterlength(ml_model.lux_model)
end

function PEtab._get_ml_model_initialparameters(ml_model::PEtab.MLModel)::ComponentArray
    rng = Random.default_rng(1)
    return Lux.initialparameters(rng, ml_model.lux_model) |> ComponentArray .|> Float64
end

function PEtab._reshape_io_data(x::Array{T})::Array{T} where {T <: AbstractFloat}
    order_py = 1:length(size(x))
    order_jl = reverse(order_py)
    imap = zeros(Int64, length(order_jl))
    for i in eachindex(order_jl)
        imap[i] = findfirst(x -> x == order_jl[i], order_py)
    end
    map = collect(1:length(order_py)) .=> imap
    return PEtab._reshape_array(x, map)
end
