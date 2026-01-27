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
function PEtab._get_lux_ps(::Type{ComponentArray}, ml_model::PEtab.MLModel)::ComponentArray
    return PEtab._get_lux_ps(Random.default_rng(), ComponentArray, ml_model)
end
function PEtab._get_lux_ps(rng::Random.AbstractRNG, ml_model::PEtab.MLModel)
    return Lux.initialparameters(rng, ml_model.lux_model)
end
function PEtab._get_lux_ps(
        rng::Random.AbstractRNG, ::Type{ComponentArray}, ml_model::PEtab.MLModel
    )::ComponentArray
    return PEtab._get_lux_ps(rng, ml_model) |>
        ComponentArray |>
        f64
end

function PEtab._get_n_ml_parameters(ml_model::PEtab.MLModel)::Integer
    return Lux.LuxCore.parameterlength(ml_model.lux_model)
end

function PEtab._reshape_io_data(x::Array{T})::Array{T} where {T <: AbstractFloat}
    order_py = 1:length(size(x))
    order_jl = reverse(order_py)
    idx_map = zeros(Int64, length(order_jl))
    for i in eachindex(order_jl)
        idx_map[i] = findfirst(x -> x == order_jl[i], order_py)
    end
    map = collect(1:length(order_py)) .=> idx_map
    return PEtab._reshape_array(x, map)
end
