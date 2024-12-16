function PEtab.set_ps_net!(ps::ComponentArray, path_h5::String, nn)::Nothing
    file = h5open(path_h5, "r")
    for (layerid, layer) in pairs(nn.layers)
        ps_layer = ps[layerid]
        isempty(ps_layer) && continue
        _set_ps_layer!(ps_layer, layer, file[string(layerid)])
        ps[layerid] = ps_layer
    end
    close(file)
    return nothing
end

function _set_ps_layer!(ps::ComponentArray, layer::Lux.Dense, file_group)::Nothing
    @unpack in_dims, out_dims, use_bias = layer
    @assert size(ps.weight) == (out_dims, in_dims) "Error in dimension of weights for Dense layer"
    ps_weight = _get_ps_layer(file_group, :weight)
    @views ps.weight .= ps_weight

    use_bias == false && return nothing
    @assert size(ps.bias) == (out_dims, ) "Error in dimension of bias for Dense layer"
    ps_bias = _get_ps_layer(file_group, :bias)
    @views ps.bias .= ps_bias
    return nothing
end
function _set_ps_layer!(ps::ComponentArray, layer::Lux.Bilinear, file_group)::Nothing
    @unpack in1_dims, in2_dims, out_dims, use_bias = layer
    @assert size(ps.weight) == (out_dims, in1_dims, in2_dims) "Error in dimension of weights for Bilinear layer"
    ps_weight = _get_ps_layer(file_group, :weight)
    @views ps.weight .= ps_weight

    use_bias == false && return nothing
    @assert size(ps.bias) == (out_dims, ) "Error in dimension of bias for Dense layer"
    ps_bias = _get_ps_layer(file_group, :bias)
    @views ps.bias .= ps_bias
    return nothing
end
function _set_ps_layer!(ps::ComponentArray, layer::Lux.Conv, file_group)::Nothing
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    @assert size(ps.weight) == (kernel_size..., in_chs, out_chs) "Error in dimension of weights for Conv layer"
    _ps_weight = _get_ps_layer(file_group, :weight)
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
    ps_bias = _get_ps_layer(file_group, :bias)
    @views ps.bias .= ps_bias
    return nothing
end
function _set_ps_layer!(ps::ComponentArray, layer::Lux.ConvTranspose, file_group)::Nothing
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    @assert size(ps.weight) == (kernel_size..., out_chs, in_chs) "Error in dimension of weights for ConvTranspose layer"
    _ps_weight = _get_ps_layer(file_group, :weight)
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
    ps_bias = _get_ps_layer(file_group, :bias)
    @views ps.bias .= ps_bias
    return nothing
end
function _set_ps_layer!(ps::ComponentArray, layer::Union{Lux.BatchNorm, Lux.InstanceNorm}, file_group)::Nothing
    @unpack affine, chs = layer
    @assert size(ps.scale) == (chs,) "Error in dimension of scale for $(typeof(layer)) layer"
    @assert size(ps.bias) == (chs,) "Error in dimension of scale for $(typeof(layer)) layer"
    # In Lux.jl the weight is named scale
    ps_scale = _get_ps_layer(file_group, :weight)
    ps_bias = _get_ps_layer(file_group, :bias)
    @views ps.scale .= ps_scale
    @views ps.bias .= ps_bias
    return nothing
end
function _set_ps_layer!(ps::ComponentArray, layer::Lux.LayerNorm, file_group)::Nothing
    @unpack shape, affine = layer
    affine == false && return nothing
    @assert length(shape) ≤ 4 "To many input dimensions for LayerNorm"
    # Somehow the dimension in Lux.jl is (shape, 1), while in PyTorch it is shape (but
    # permuted)
    @assert size(ps.scale) == (shape..., 1) "Error in dimension of scale for LayerNorm layer"
    @assert size(ps.bias) == (shape..., 1) "Error in dimension of bias for LayerNorm layer"
    # In Lux.jl the weight is named scale
    _ps_scale = _get_ps_layer(file_group, :weight)
    _ps_bias = _get_ps_layer(file_group, :bias)
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

function _get_ps_layer(file_group, which::Symbol)
    if which == :weight
        ps = HDF5.read_dataset(file_group, "weight")
    else
        ps = HDF5.read_dataset(file_group, "bias")
    end
    # Julia is column-major, while the standard format is row-major
    return permutedims(ps, reverse(1:ndims(ps)))
end
