function PEtab._set_ml_model_ps!(
        ps::ComponentArray, path_h5::String, lux_model, ml_id::Symbol
    )::Nothing
    file = h5open(path_h5, "r")

    net_parameters = file["parameters"]["$(ml_id)"]
    st = Lux.initialstates(Random.default_rng(1), lux_model)
    for (layerid, layer) in pairs(lux_model.layers)
        ps_layer = ps[layerid]
        isempty(ps_layer) && continue
        _set_ps_layer!(ps_layer, layer, st[layerid], net_parameters[string(layerid)])
        ps[layerid] = ps_layer
    end
    close(file)
    return nothing
end

function _set_ps_layer!(
        ps::ComponentArray, layer::Lux.Experimental.FrozenLayer, st_layer::NamedTuple,
        file_group
    )::Nothing
    _set_ps_layer!(ps, layer.layer, st_layer, file_group)
    return nothing
end
function _set_ps_layer!(
        ps::ComponentArray, layer::Lux.Dense, st_layer::NamedTuple, file_group
    )::Nothing
    @unpack in_dims, out_dims, use_bias = layer
    if _is_frozen(st_layer, :weight) == false
        @assert size(ps.weight) == (out_dims, in_dims) "Error in dimension of weights for Dense layer"
        ps_weight = _get_ps_layer(file_group, :weight)
        _set_ps_array!(ps, :weight, ps_weight)
    end
    if _is_frozen(st_layer, :bias) == false
        use_bias == false && return nothing
        @assert size(ps.bias) == (out_dims,) "Error in dimension of bias for Dense layer"
        ps_bias = _get_ps_layer(file_group, :bias)
        _set_ps_array!(ps, :bias, ps_bias)
    end
    return nothing
end
function _set_ps_layer!(
        ps::ComponentArray, layer::Lux.Bilinear, st_layer::NamedTuple, file_group
    )::Nothing
    @unpack in1_dims, in2_dims, out_dims, use_bias = layer
    if _is_frozen(st_layer, :weight) == false
        @assert size(ps.weight) == (out_dims, in1_dims, in2_dims) "Error in dimension of weights for Bilinear layer"
        ps_weight = _get_ps_layer(file_group, :weight)
        _set_ps_array!(ps, :weight, ps_weight)
    end
    if _is_frozen(st_layer, :bias) == false
        use_bias == false && return nothing
        @assert size(ps.bias) == (out_dims,) "Error in dimension of bias for Dense layer"
        ps_bias = _get_ps_layer(file_group, :bias)
        _set_ps_array!(ps, :bias, ps_bias)
    end
    return nothing
end
function _set_ps_layer!(
        ps::ComponentArray, layer::Lux.Conv, st_layer::NamedTuple, file_group
    )::Nothing
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    if _is_frozen(st_layer, :weight) == false
        @assert size(ps.weight) == (kernel_size..., in_chs, out_chs) "Error in dimension \
            of weights for Conv layer"
        _ps_weight = _get_ps_layer(file_group, :weight)
        if length(kernel_size) == 1
            ps_weight = PEtab._reshape_array(_ps_weight, CONV1D_MAP)
        elseif length(kernel_size) == 2
            ps_weight = PEtab._reshape_array(_ps_weight, CONV2D_MAP)
        elseif length(kernel_size) == 3
            ps_weight = PEtab._reshape_array(_ps_weight, CONV3D_MAP)
        end
        _set_ps_array!(ps, :weight, ps_weight)
    end
    if _is_frozen(st_layer, :bias) == false
        use_bias == false && return nothing
        @assert size(ps.bias) == (out_chs,) "Error in dimension of bias for Conv layer"
        ps_bias = _get_ps_layer(file_group, :bias)
        _set_ps_array!(ps, :bias, ps_bias)
    end
    return nothing
end
function _set_ps_layer!(
        ps::ComponentArray, layer::Lux.ConvTranspose, st_layer::NamedTuple, file_group
    )::Nothing
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    if _is_frozen(st_layer, :weight) == false
        @assert size(ps.weight) == (kernel_size..., out_chs, in_chs) "Error in dimension \
            of weights for ConvTranspose layer"
        _ps_weight = _get_ps_layer(file_group, :weight)
        if length(kernel_size) == 1
            ps_weight = PEtab._reshape_array(_ps_weight, CONV1D_MAP)
        elseif length(kernel_size) == 2
            ps_weight = PEtab._reshape_array(_ps_weight, CONV2D_MAP)
        elseif length(kernel_size) == 3
            ps_weight = PEtab._reshape_array(_ps_weight, CONV3D_MAP)
        end
        _set_ps_array!(ps, :weight, ps_weight)
    end
    if _is_frozen(st_layer, :bias) == false
        use_bias == false && return nothing
        @assert size(ps.bias) == (out_chs,) "Error in dimension of bias for \
            ConvTranspose layer"
        ps_bias = _get_ps_layer(file_group, :bias)
        _set_ps_array!(ps, :bias, ps_bias)
    end
    return nothing
end
function _set_ps_layer!(
        ps::ComponentArray, layer::Union{Lux.BatchNorm, Lux.InstanceNorm},
        st_layer::NamedTuple, file_group
    )::Nothing
    @unpack affine, chs = layer
    affine == false && return nothing
    # In Lux.jl the weight is named scale
    if _is_frozen(st_layer, :scale) == false
        @assert size(ps.scale) == (chs,) "Error in dimension of scale for $(typeof(layer)) \
            layer"
        ps_scale = _get_ps_layer(file_group, :weight)
        _set_ps_array!(ps, :scale, ps_scale)
    end
    if _is_frozen(st_layer, :bias) == false
        @assert size(ps.bias) == (chs,) "Error in dimension of scale for $(typeof(layer)) \
            layer"
        ps_bias = _get_ps_layer(file_group, :bias)
        _set_ps_array!(ps, :bias, ps_bias)
    end
    return nothing
end
function _set_ps_layer!(
        ps::ComponentArray, layer::Lux.LayerNorm, st_layer::NamedTuple, file_group
    )::Nothing
    @unpack shape, affine = layer
    affine == false && return nothing
    @assert length(shape) ≤ 4 "To many input dimensions for LayerNorm"
    if _is_frozen(st_layer, :scale) == false
        # In Lux.jl the weight is named scale
        # Somehow the dimension in Lux.jl is (shape, 1), while in PyTorch it is shape (but
        # permuted)
        @assert size(ps.scale) == (shape..., 1) "Error in dimension of scale for LayerNorm layer"
        _ps_scale = _get_ps_layer(file_group, :weight)
        if length(shape) == 4
            ps_scale = PEtab._reshape_array(_ps_scale, LAYERNORM4_MAP)
        elseif length(shape) == 3
            ps_scale = PEtab._reshape_array(_ps_scale, LAYERNORM3_MAP)
        elseif length(shape) == 2
            ps_scale = PEtab._reshape_array(_ps_scale, LAYERNORM2_MAP)
        else
            ps_scale = _ps_scale
        end
        _set_ps_array!(ps, :scale, ps_scale)
    end
    if _is_frozen(st_layer, :bias) == false
        @assert size(ps.bias) == (shape..., 1) "Error in dimension of bias for \
            LayerNorm layer"
        _ps_bias = _get_ps_layer(file_group, :bias)
        if length(shape) == 4
            ps_bias = PEtab._reshape_array(_ps_bias, LAYERNORM4_MAP)
        elseif length(shape) == 3
            ps_bias = PEtab._reshape_array(_ps_bias, LAYERNORM3_MAP)
        elseif length(shape) == 2
            ps_bias = PEtab._reshape_array(_ps_bias, LAYERNORM2_MAP)
        else
            ps_bias = _ps_bias
        end
        _set_ps_array!(ps, :bias, ps_bias)
    end
    return nothing
end
function _set_ps_layer!(
        ::ComponentArray, ::PS_FREE_LAYERS, ::NamedTuple, ::DataFrame
    )::Nothing
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

function _set_ps_array!(
        ps::ComponentArray, id::Symbol, value::Array{<:AbstractFloat}
    )::Nothing
    # Happens when a layer if frozen
    isempty(ps[id]) && return nothing
    @views ps[id] .= value
    return nothing
end

function _is_frozen(st_layer::NamedTuple, arrayid::Symbol)::Bool
    !haskey(st_layer, :frozen_params) && return false
    return haskey(st_layer[:frozen_params], arrayid)
end
