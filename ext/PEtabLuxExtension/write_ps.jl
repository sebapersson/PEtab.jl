function PEtab.nn_ps_to_h5!(nn, ps::Union{ComponentArray, NamedTuple}, path::String)::Nothing
    if isfile(path)
        rm(path)
    end
    file = h5open(path, "w")
    for (layername, layer) in pairs(nn.layers)
        _ps_to_h5!(file, layer, ps[layername], layername)
    end
    close(file)
    return nothing
end

"""
    _ps_to_h5!(layer::Lux.Dense, ps, netname, layername)::Nothing

Transforms parameters (`ps`) for a Lux layer to a h5 file with the data stored in the
order expected by PyTorich.

For `Dense` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(out_features, in_features)`
- `bias` of dimension `(out_features)`
"""
function _ps_to_h5!(file, layer::Lux.Dense, ps::Union{NamedTuple, ComponentArray}, layername::Symbol)::Nothing
    @unpack in_dims, out_dims, use_bias = layer
    g = create_group(file, string(layername))
    _ps_weight_to_h5!(g, ps)
    _ps_bias_to_h5!(g, ps, use_bias)
    return nothing
end
"""
    _ps_to_h5!(layer::Lux.ConvTranspose, ...)::Nothing

For `Conv` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(in_channels, out_channels, kernel_size)`
- `bias` of dimension `(out_channels)`

!!! note
    Note, in Lux.jl `weight` has `(kernel_size, in_channels, out_channels)`. This is fixed
    by the importer.
"""
function _ps_to_h5!(file, layer::Lux.Conv, ps::Union{NamedTuple, ComponentArray}, layername::Symbol)::Nothing
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    if length(kernel_size) == 1
        _psweigth = PEtab._reshape_array(ps.weight, CONV1D_MAP)
    elseif length(kernel_size) == 2
        _psweigth = PEtab._reshape_array(ps.weight, CONV2D_MAP)
    elseif length(kernel_size) == 3
        _psweigth = PEtab._reshape_array(ps.weight, CONV3D_MAP)
    end
    _ps = ComponentArray(weight = _psweigth)
    g = create_group(file, string(layername))
    _ps_weight_to_h5!(g, _ps)
    _ps_bias_to_h5!(g, ps, use_bias)
    return nothing
end
"""
    _ps_to_h5!(layer::Lux.ConvTranspose, ...)::Nothing

For `ConvTranspose` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(in_channels, out_channels, kernel_size)`
- `bias` of dimension `(out_channels)`

!!! note
    Note, in Lux.jl `weight` has `(kernel_size, out_channels, in_channels)`. This is fixed
    by the importer.
"""
function _ps_to_h5!(file, layer::Lux.ConvTranspose, ps::Union{NamedTuple, ComponentArray}, layername::Symbol)::Nothing
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    if length(kernel_size) == 1
        _psweigth = PEtab._reshape_array(ps.weight, CONV1D_MAP)
    elseif length(kernel_size) == 2
        _psweigth = PEtab._reshape_array(ps.weight, CONV2D_MAP)
    elseif length(kernel_size) == 3
        _psweigth = PEtab._reshape_array(ps.weight, CONV3D_MAP)
    end
    _ps = ComponentArray(weight = _psweigth)
    g = create_group(file, string(layername))
    _ps_weight_to_h5!(g, _ps)
    _ps_bias_to_h5!(g, ps, use_bias)
    return nothing
end
"""
    _ps_to_h5!(layer::Lux.Bilinear, ...)::Nothing

For `Bilinear` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(out_features, in_features1, in_features2)`
- `bias` of dimension `(out_features)`
"""
function _ps_to_h5!(file, layer::Lux.Bilinear, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::Nothing
    @unpack in1_dims, in2_dims, out_dims, use_bias = layer
    g = create_group(file, string(layername))
    _ps_weight_to_h5!(g, ps)
    _ps_bias_to_h5!(g, ps, use_bias)
    return nothing
end
"""
    layer_ps_to_h5!(layer::Union{Lux.BatchNorm, Lux.InstanceNorm}, ...)::Nothing

For `BatchNorm` and `InstanceNorm` layer possible parameters that are saved to a DataFrame
are:
- `scale/weight` of dimension `(num_features)`
- `bias` of dimension `(num_features)`
!!! note
    in Lux.jl the dimension argument `num_features` is chs (number of input channels)
"""
function layer_ps_to_h5!(file, layer::Union{Lux.BatchNorm, Lux.InstanceNorm}, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::Nothing
    @unpack affine, chs = layer
    affine == false && return DataFrame()
    g = create_group(file, string(layername))
    _ps_weight_to_h5!(g, ps; scale = true)
    _ps_bias_to_h5!(g, ps, true)
    return nothing
end
"""
    layer_ps_to_h5!(layer::Lux.LayerNorm, ...)::Nothing

For `LayerNorm` layer possible parameters that are saved to a DataFrame are:
- `scale/weight` of `size(input)` dimension
- `bias` of `size(input)` dimension

!!! note
    Input order differs between Lux.jl and PyTorch. Order `["C", "D", "H", "W"]` in
    PyTorch corresponds to `["W", "H", "D", "C"]` in Lux.jl. Basically, regardless of input
    dimension the Lux.jl dimension is the PyTorch dimension reversed.
"""
function layer_ps_to_h5!(file, layer::LayerNorm, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::Nothing
    @unpack shape, affine = layer
    affine == false && return DataFrame()
    # Note, in Lux.jl the input dimension is `size(input, 1)`.
    if length(shape) == 4
        _psweigth = PEtab._reshape_array(ps.scale[:, :, :, :, 1], LAYERNORM4_MAP)
        _psbias = PEtab._reshape_array(ps.bias[:, :, :, :, 1], LAYERNORM4_MAP)
    elseif length(shape) == 3
        _psweigth = PEtab._reshape_array(ps.scale[:, :, :, 1], LAYERNORM3_MAP)
        _psbias = PEtab._reshape_array(ps.bias[:, :, :, 1], LAYERNORM3_MAP)
    elseif length(shape) == 2
        _psweigth = PEtab._reshape_array(ps.scale[:, :, 1], LAYERNORM2_MAP)
        _psbias = PEtab._reshape_array(ps.bias[:, :, 1], LAYERNORM2_MAP)
    elseif length(shape) == 1
        _psweigth = ps.scale[:, 1]
        _psbias = ps.bias[:, 1]
    end
    _ps = ComponentArray(weight = _psweigth, bias = _psbias)
    g = create_group(file, string(layername))
    _ps_weight_to_h5!(g, _ps)
    _ps_bias_to_h5!(g, _ps, true)
    return nothing
end
"""
    _ps_to_h5!(layer::PS_FREE_LAYERS, ...)::Nothing

Layers without parameters to estimate.
"""
function _ps_to_h5!(layer::PS_FREE_LAYERS, ::Union{NamedTuple, ComponentArray}, ::Symbol)::Nothing
    return nothing
end

function _ps_weight_to_h5!(g, ps; scale::Bool = false)::Nothing
    # For Batchnorm in Lux.jl the weight layer is referred to as scale.
    ps_weight = scale == false ? ps.weight : ps.scale
    # To account for Python (for which the standard is defined) is row-major
    ps_weight = permutedims(ps_weight, reverse(1:ndims(ps_weight)))
    g["weight"] = ps_weight
    return nothing
end

function _ps_bias_to_h5!(g, ps, use_bias)::Nothing
    use_bias == false && return nothing
    if length(size(ps.bias)) > 1
        ps_bias = permutedims(ps.bias, reverse(1:ndims(ps.bias)))
    else
        ps_bias = vec(ps.bias)
    end
    g["bias"] = ps_bias
    return nothing
end
