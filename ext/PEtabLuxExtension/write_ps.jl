function nn_ps_to_tidy(nn, ps::Union{ComponentArray, NamedTuple}, netname::Symbol)::DataFrame
    df_ps = DataFrame()
    for (layername, layer) in pairs(nn.layers)
        ps_layer = ps[layername]
        df_layer = _ps_to_tidy(layer, ps_layer, netname, layername)
        df_ps = vcat(df_ps, df_layer)
    end
    return df_ps
end

"""
    _ps_to_tidy(layer::Lux.Dense, ps, netname, layername)::DataFrame

Transforms parameters (`ps`) for a Lux layer to a tidy DataFrame `df` with columns `value`
and `parameterId`.

A `Lux.Dense` layer has two sets of parameters, `weight` and optionally `bias`. For
`Dense` and all other form of layers `weight` and `bias` are stored as:

- `netname_layername_weight_ix`: weight for output `i` and input `j`
- `netname_layername_bias_ix`: bias for output `i`

Where `ix` depends on the `Tensor` the parameters are stored in. For example, if
`size(weight) = (5, 2)` `ix` is on the form `ix = i_j`. Here, it is important to note that
the PEtab standard uses Julia tensor. For example, `x = ones(5, 3, 2)` can be thought of
as a Tensor with height 5, width 3 and depth 2. In PyTorch `x` would correspond to
`x = torch.ones(2, 5, 3)`.

For `Dense` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(out_features, in_features)`
- `bias` of dimension `(out_features)`
"""
function _ps_to_tidy(layer::Lux.Dense, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::DataFrame
    @unpack in_dims, out_dims, use_bias = layer
    df_weight = _ps_weight_to_tidy(ps, netname, layername)
    df_bias = _ps_bias_to_tidy(ps, (out_dims, ), netname, layername, use_bias)
    return vcat(df_weight, df_bias)
end
"""
    _ps_to_tidy(layer::Lux.ConvTranspose, ...)::DataFrame

For `Conv` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(in_channels, out_channels, kernel_size)`
- `bias` of dimension `(out_channels)`

!!! note
    Note, in Lux.jl `weight` has `(kernel_size, in_channels, out_channels)`. This is fixed
    by the importer.
"""
function _ps_to_tidy(layer::Lux.Conv, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::DataFrame
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    if length(kernel_size) == 1
        _psweigth = PEtab._reshape_array(ps.weight, CONV1D_MAP)
    elseif length(kernel_size) == 2
        _psweigth = PEtab._reshape_array(ps.weight, CONV2D_MAP)
    elseif length(kernel_size) == 3
        _psweigth = PEtab._reshape_array(ps.weight, CONV3D_MAP)
    end
    _ps = ComponentArray(weight = _psweigth)
    df_weight = _ps_weight_to_tidy(_ps, netname, layername)
    df_bias = _ps_bias_to_tidy(ps, (out_chs, ), netname, layername, use_bias)
    return vcat(df_weight, df_bias)
end
"""
    _ps_to_tidy(layer::Lux.ConvTranspose, ...)::DataFrame

For `ConvTranspose` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(in_channels, out_channels, kernel_size)`
- `bias` of dimension `(out_channels)`

!!! note
    Note, in Lux.jl `weight` has `(kernel_size, out_channels, in_channels)`. This is fixed
    by the importer.
"""
function _ps_to_tidy(layer::Lux.ConvTranspose, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::DataFrame
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    if length(kernel_size) == 1
        _psweigth = PEtab._reshape_array(ps.weight, CONV1D_MAP)
    elseif length(kernel_size) == 2
        # For the mapping, see comment above on image format in Lux.Conv
        _psweigth = PEtab._reshape_array(ps.weight, CONV2D_MAP)
    elseif length(kernel_size) == 3
        # See comment on Lux.Conv
        _psweigth = PEtab._reshape_array(ps.weight, CONV3D_MAP)
    end
    _ps = ComponentArray(weight = _psweigth)
    df_weight = _ps_weight_to_tidy(_ps, netname, layername)
    df_bias = _ps_bias_to_tidy(ps, (out_chs, ), netname, layername, use_bias)
    return vcat(df_weight, df_bias)
end
"""
    _ps_to_tidy(layer::Lux.Bilinear, ...)::DataFrame

For `Bilinear` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(out_features, in_features1, in_features2)`
- `bias` of dimension `(out_features)`
"""
function _ps_to_tidy(layer::Lux.Bilinear, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::DataFrame
    @unpack in1_dims, in2_dims, out_dims, use_bias = layer
    df_weight = _ps_weight_to_tidy(ps, netname, layername)
    df_bias = _ps_bias_to_tidy(ps, (out_dims, ), netname, layername, use_bias)
    return vcat(df_weight, df_bias)
end
"""
    layer_ps_to_tidy(layer::Union{Lux.BatchNorm, Lux.InstanceNorm}, ...)::DataFrame

For `BatchNorm` and `InstanceNorm` layer possible parameters that are saved to a DataFrame
are:
- `scale/weight` of dimension `(num_features)`
- `bias` of dimension `(num_features)`
!!! note
    in Lux.jl the dimension argument `num_features` is chs (number of input channels)
"""
function layer_ps_to_tidy(layer::Union{Lux.BatchNorm, Lux.InstanceNorm}, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::DataFrame
    @unpack affine, chs = layer
    affine == false && return DataFrame()
    df_weight = _ps_weight_to_tidy(ps, netname, layername; scale = true)
    df_bias = _ps_bias_to_tidy(ps, (chs, ), netname, layername, true)
    return vcat(df_weight, df_bias)
end
"""
    layer_ps_to_tidy(layer::Lux.LayerNorm, ...)::DataFrame

For `LayerNorm` layer possible parameters that are saved to a DataFrame are:
- `scale/weight` of `size(input)` dimension
- `bias` of `size(input)` dimension

!!! note
    Input order differs between Lux.jl and PyTorch. Order `["C", "D", "H", "W"]` in
    PyTorch corresponds to `["W", "H", "D", "C"]` in Lux.jl. Basically, regardless of input
    dimension the Lux.jl dimension is the PyTorch dimension reversed.
"""
function layer_ps_to_tidy(layer::LayerNorm, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::DataFrame
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
    df_weight = _ps_weight_to_tidy(_ps, netname, layername)
    df_bias = _ps_bias_to_tidy(_ps, size(_ps.bias), netname, layername, true)
    return vcat(df_weight, df_bias)
end
"""
    _ps_to_tidy(layer::PS_FREE_LAYERS, ...)::DataFrame

Layers without parameters to estimate.
"""
function _ps_to_tidy(layer::PS_FREE_LAYERS, ::Union{NamedTuple, ComponentArray}, ::Symbol, ::Symbol)::DataFrame
    return DataFrame()
end

function _ps_weight_to_tidy(ps, netname::Symbol, layername::Symbol; scale::Bool = false)::DataFrame
    # For Batchnorm in Lux.jl the weight layer is refered to as scale.
    if scale == false
        ps_weight = ps.weight
    else
        ps_weight = ps.scale
    end
    if length(size(ps_weight)) > 1
        iweight = getfield.(findall(x -> true, ones(size(ps_weight))), :I)
    else
        iweight = 1:length(ps_weight)
    end
    iweight = [iw .- 1 for iw in iweight]
    weight_names =  ["weight" * prod("_" .* string.(ix)) for ix in iweight]
    df_weight = DataFrame(parameterId = "$(netname)_$(layername)_" .* weight_names,
                          value = vec(ps_weight))
    return df_weight
end

function _ps_bias_to_tidy(ps, bias_dims, netname::Symbol, layername::Symbol, use_bias)::DataFrame
    use_bias == false && return DataFrame()
    if length(bias_dims) > 1
        ibias = getfield.(findall(x -> true, ones(bias_dims)), :I)
    else
        ibias = (0:bias_dims[1]-1)
    end
    bias_names =  ["bias" * prod("_" .* string.(ix)) for ix in ibias]
    df_bias = DataFrame(parameterId = "$(netname)_$(layername)_" .* bias_names,
                        value = vec(ps.bias))
    return df_bias
end
