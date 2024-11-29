function PEtab.set_ps_net!(ps::ComponentArray, df_ps::DataFrame, netname::Symbol, nn)::Nothing
    df_net = df_ps[startswith.( df_ps[!, :parameterId], "$(netname)_"), :]
    for (id, layer) in pairs(nn.layers)
        df_layer = df_net[startswith.(df_net[!, :parameterId], "$(netname)_$(id)_"), :]
        ps_layer = ps[id]
        isempty(ps_layer) && continue
        _set_ps_layer!(ps_layer, layer, df_layer)
        ps[id] = ps_layer
    end
    return nothing
end

function _set_ps_layer!(ps::ComponentArray, layer::Lux.Dense, df_ps::DataFrame)::Nothing
    @unpack in_dims, out_dims, use_bias = layer
    @assert size(ps.weight) == (out_dims, in_dims) "Error in dimension of weights for Dense layer"
    ps_weight = _get_ps_layer(df_ps, 2, :weight)
    @views ps.weight .= ps_weight

    use_bias == false && return nothing
    @assert size(ps.bias) == (out_dims, ) "Error in dimension of bias for Dense layer"
    ps_bias = _get_ps_layer(df_ps, 1, :bias)
    @views ps.bias .= ps_bias
    return nothing
end
function _set_ps_layer!(ps::ComponentArray, layer::Lux.Bilinear, df_ps::DataFrame)::Nothing
    @unpack in1_dims, in2_dims, out_dims, use_bias = layer
    @assert size(ps.weight) == (out_dims, in1_dims, in2_dims) "Error in dimension of weights for Bilinear layer"
    ps_weight = _get_ps_layer(df_ps, 3, :weight)
    @views ps.weight .= ps_weight

    use_bias == false && return nothing
    @assert size(ps.bias) == (out_dims, ) "Error in dimension of bias for Dense layer"
    ps_bias = _get_ps_layer(df_ps, 1, :bias)
    @views ps.bias .= ps_bias
    return nothing
end
function _set_ps_layer!(ps::ComponentArray, layer::Lux.Conv, df_ps::DataFrame)::Nothing
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    @assert size(ps.weight) == (kernel_size..., in_chs, out_chs) "Error in dimension of weights for Conv layer"
    _ps_weight = _get_ps_layer(df_ps, length((kernel_size..., in_chs, out_chs)), :weight)
    if length(kernel_size) == 1
        ps_weight = PEtab._reshape_array(_ps_weight, CONV1D_MAP)
    elseif length(kernel_size) == 2
        ps_weight = PEtab._reshape_array(_ps_weight, CONV2D_MAP)
    elseif length(kernel_size) == 3
        ps_weight = PEtab._reshape_array(_ps_weight, CONV3D_MAP)
    end
    @views ps.weight .= ps_weight

    use_bias == false && return nothing
    @assert size(ps.bias) == (out_chs, ) "Error in dimension of bias for Conv layer"
    ps_bias = _get_ps_layer(df_ps, 1, :bias)
    @views ps.bias .= ps_bias
    return nothing
end
function _set_ps_layer!(ps::ComponentArray, layer::Lux.ConvTranspose, df_ps::DataFrame)::Nothing
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    @assert size(ps.weight) == (kernel_size..., out_chs, in_chs) "Error in dimension of weights for ConvTranspose layer"
    _ps_weight = _get_ps_layer(df_ps, length((kernel_size..., out_chs, in_chs)), :weight)
    if length(kernel_size) == 1
        ps_weight = PEtab._reshape_array(_ps_weight, CONV1D_MAP)
    elseif length(kernel_size) == 2
        ps_weight = PEtab._reshape_array(_ps_weight, CONV2D_MAP)
    elseif length(kernel_size) == 3
        ps_weight = PEtab._reshape_array(_ps_weight, CONV3D_MAP)
    end
    @views ps.weight .= ps_weight

    use_bias == false && return nothing
    @assert size(ps.bias) == (out_chs, ) "Error in dimension of bias for ConvTranspose layer"
    ps_bias = _get_ps_layer(df_ps, 1, :bias)
    @views ps.bias .= ps_bias
    return nothing
end
function _set_ps_layer!(ps::ComponentArray, layer::Union{Lux.BatchNorm, Lux.InstanceNorm}, df_ps::DataFrame)::Nothing
    @unpack affine, chs = layer
    @assert size(ps.scale) == (chs,) "Error in dimension of scale for $(typeof(layer)) layer"
    @assert size(ps.bias) == (chs,) "Error in dimension of scale for $(typeof(layer)) layer"
    # In Lux.jl the weight is named scale
    ps_scale = _get_ps_layer(df_ps, 1, :weight)
    ps_bias = _get_ps_layer(df_ps, 1, :bias)
    @views ps.scale .= ps_scale
    @views ps.bias .= ps_bias
    return nothing
end
function _set_ps_layer!(ps::ComponentArray, layer::Lux.LayerNorm, df_ps::DataFrame)::Nothing
    @unpack shape, affine = layer
    affine == false && return nothing
    @assert length(shape) ≤ 4 "To many input dimensions for LayerNorm"
    # Somehow the dimension in Lux.jl is (shape, 1), while in PyTorch it is shape (but
    # permuted)
    @assert size(ps.scale) == (shape..., 1) "Error in dimension of scale for LayerNorm layer"
    @assert size(ps.bias) == (shape..., 1) "Error in dimension of bias for LayerNorm layer"
    # In Lux.jl the weight is named scale
    _ps_scale = _get_ps_layer(df_ps, length(shape), :weight)
    _ps_bias = _get_ps_layer(df_ps, length(shape), :bias)
    if length(shape) == 4
        ps_scale = PEtab._reshape_array(_ps_scale, LAYERNORM4_MAP)
        ps_bias = PEtab._reshape_array(_ps_bias, LAYERNORM4_MAP)
    elseif length(shape) == 3
        ps_scale = PEtab._reshape_array(_ps_scale, LAYERNORM3_MAP)
        ps_bias = PEtab._reshape_array(_ps_bias, LAYERNORM3_MAP)
    elseif length(shape) == 2
        ps_scale = PEtab._reshape_array(_ps_scale, LAYERNORM2_MAP)
        ps_bias = PEtab._reshape_array(_ps_bias, LAYERNORM2_MAP)
    else
        ps_scale = _ps_scale
        ps_bias = _ps_bias
    end
    @views ps.scale .= ps_scale
    @views ps.bias .= ps_bias
    return nothing
end
function _set_ps_layer!(::ComponentArray, ::PS_FREE_LAYERS, ::DataFrame)::Nothing
    return nothing
end

function _get_ps_layer(df_layer::DataFrame, lendim::Int64, which::Symbol)
    if which == :weight
        df = df_layer[occursin.("weight_", df_layer[!, :parameterId]), :]
    else
        df = df_layer[occursin.("bias_", df_layer[!, :parameterId]), :]
    end
    ix = Any[]
    for pid in df[!, :parameterId]
        _ix = parse.(Int64, collect(m.match for m in eachmatch(r"\d+", pid))[((end-lendim+1):end)])
        # Python -> Julia indexing
        _ix .+= 1
        push!(ix, Tuple(_ix))
    end
    out = zeros(maximum(ix))
    for i in eachindex(ix)
        out[ix[i]...] = df[i, :value]
    end
    return out
end
