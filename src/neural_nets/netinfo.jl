const PS_FREE_LAYERS = Union{Lux.MaxPool, Lux.MeanPool, Lux.LPPool, Lux.AdaptiveMaxPool, Lux.AdaptiveMeanPool, Lux.FlattenLayer, Lux.Dropout, Lux.AlphaDropout}

const CONV_ARGS = ["kernel_size" => 1, "in_channels => out_channels" => 2]
const CONV_KWARGS = ["stride" => "stride", "padding" => "pad", "dilation" => "dilation",
                     "bias" => "use_bias"]
const CONVT_KWARGS = vcat(CONV_KWARGS, ["output_padding" => "outpad"])
const MAXPOOL_KWARGS = ["stride" => "stride", "padding" => "pad", "dilation" => "dilation"]
const AVGPOOL_KWARGS = ["stride" => "stride", "padding" => "pad"]
const LPPOOL_KWARGS = ["stride" => "stride", "norm_type" => "p"]
const BATCHNORM_KWARGS = ["eps" => "epsilon", "momentum" => "momentum",
                          "track_running_stats" => "track_stats", "affine" => "affine"]
const INSTANCENORM_KWARGS = BATCHNORM_KWARGS
const LAYERNORM_KWARGS = ["eps" => "epsilon", "elementwise_affine" => "affine"]

const LAYERS = Dict(
    "Linear" => (lux_layer = Lux.Dense,
                 args = ["in_features" => 1, "out_features" => 2],
                 kwargs = ["bias" => "use_bias"]),
    "Bilinear" => (lux_layer = Lux.Bilinear,
                   args = ["(in1_features, in2_features) => out_features" => 1],
                   kwargs = ["bias" => "use_bias"]),
    "Conv1d" => (lux_layer = Lux.Conv,
                 args = CONV_ARGS,
                 kwargs = CONV_KWARGS,
                 kwargs_julia = (; cross_correlation = true)),
    "Conv2d" => (lux_layer = Lux.Conv,
                 args = CONV_ARGS,
                 kwargs = CONV_KWARGS,
                 kwargs_julia = (; cross_correlation = true)),
    "Conv3d" => (lux_layer = Lux.Conv,
                 args = CONV_ARGS,
                 kwargs = CONV_KWARGS,
                 kwargs_julia = (; cross_correlation = true)),
    "ConvTranspose1d" => (lux_layer = Lux.ConvTranspose,
                          args = CONV_ARGS,
                          kwargs = CONVT_KWARGS,
                          kwargs_julia = (; cross_correlation = true)),
    "ConvTranspose2d" => (lux_layer = Lux.ConvTranspose,
                          args = CONV_ARGS,
                          kwargs = CONVT_KWARGS,
                          kwargs_julia = (; cross_correlation = true)),
    "ConvTranspose3d" => (lux_layer = Lux.ConvTranspose,
                          args = CONV_ARGS,
                          kwargs = CONVT_KWARGS,
                          kwargs_julia = (; cross_correlation = true)),
    "Flatten" => (lux_layer = Lux.FlattenLayer,
                  args = String[],
                  kwargs = ["start_dim", "end_dim"]),
    "MaxPool1d" => (lux_layer = Lux.MaxPool,
                    args = ["kernel_size" => 1],
                    kwargs = MAXPOOL_KWARGS,
                    tuple_args = ["kernel_size"]),
    "MaxPool2d" => (lux_layer = Lux.MaxPool,
                    args = ["kernel_size" => 1],
                    kwargs = MAXPOOL_KWARGS),
    "MaxPool3d" => (lux_layer = Lux.MaxPool,
                    args = ["kernel_size" => 1],
                    kwargs = MAXPOOL_KWARGS),
    "AvgPool1d" => (lux_layer = Lux.MeanPool,
                    args = ["kernel_size" => 1],
                    kwargs = AVGPOOL_KWARGS,
                    tuple_args = ["kernel_size"]),
    "AvgPool2d" => (lux_layer = Lux.MeanPool,
                    args = ["kernel_size" => 1],
                    kwargs = AVGPOOL_KWARGS),
    "AvgPool3d" => (lux_layer = Lux.MeanPool,
                    args = ["kernel_size" => 1],
                    kwargs = AVGPOOL_KWARGS),
    "LPPool1d" => (lux_layer = Lux.LPPool,
                   args = ["kernel_size" => 1],
                   kwargs = LPPOOL_KWARGS,
                   tuple_args = ["kernel_size"]),
    "LPPool2d" => (lux_layer = Lux.LPPool,
                   args = ["kernel_size" => 1],
                   kwargs = LPPOOL_KWARGS),
    "LPPool3d" => (lux_layer = Lux.LPPool,
                   args = ["kernel_size" => 1],
                   kwargs = LPPOOL_KWARGS),
    "AdaptiveAvgPool1d" => (lux_layer = Lux.AdaptiveMeanPool,
                            args = ["output_size" => 1],
                            kwargs = String[],
                            tuple_args = ["output_size"]),
    "AdaptiveAvgPool2d" => (lux_layer = Lux.AdaptiveMeanPool,
                            args = ["output_size" => 1],
                            kwargs = String[],),
    "AdaptiveAvgPool3d" => (lux_layer = Lux.AdaptiveMeanPool,
                            args = ["output_size" => 1],
                            kwargs = String[],),
    "AdaptiveMaxPool1d" => (lux_layer = Lux.AdaptiveMaxPool,
                            args = ["output_size" => 1],
                            kwargs = String[],
                            tuple_args = ["output_size"]),
    "AdaptiveMaxPool2d" => (lux_layer = Lux.AdaptiveMaxPool,
                            args = ["output_size" => 1],
                            kwargs = String[],),
    "AdaptiveMaxPool3d" => (lux_layer = Lux.AdaptiveMaxPool,
                            args = ["output_size" => 1],
                            kwargs = String[],),
    "Dropout" => (lux_layer = Lux.Dropout,
                  args = ["p" => 1],
                  kwargs = String[],),
    "AlphaDropout" => (lux_layer = Lux.AlphaDropout,
                       args = ["p" => 1],
                       kwargs = String[],),
    "Dropout1d" => (lux_layer = Lux.Dropout,
                    args = ["p" => 1],
                    kwargs = String[],
                    kwargs_julia = (; :dims => (2))),
    "Dropout2d" => (lux_layer = Lux.Dropout,
                    args = ["p" => 1],
                    kwargs = String[],
                    kwargs_julia = (; :dims => (3))),
    "Dropout3d" => (lux_layer = Lux.Dropout,
                    args = ["p" => 1],
                    kwargs = String[],
                    kwargs_julia = (; :dims => (4))),
    "BatchNorm1d" => (lux_layer = Lux.BatchNorm,
                      args = ["num_features" => 1],
                      kwargs = BATCHNORM_KWARGS),
    "BatchNorm2d" => (lux_layer = Lux.BatchNorm,
                      args = ["num_features" => 1],
                      kwargs = BATCHNORM_KWARGS),
    "BatchNorm3d" => (lux_layer = Lux.BatchNorm,
                      args = ["num_features" => 1],
                      kwargs = BATCHNORM_KWARGS),
    "InstanceNorm1d" => (lux_layer = Lux.InstanceNorm,
                         args = ["num_features" => 1],
                         kwargs = BATCHNORM_KWARGS),
    "InstanceNorm2d" => (lux_layer = Lux.InstanceNorm,
                         args = ["num_features" => 1],
                         kwargs = BATCHNORM_KWARGS),
    "InstanceNorm3d" => (lux_layer = Lux.InstanceNorm,
                         args = ["num_features" => 1],
                         kwargs = BATCHNORM_KWARGS),
    "LayerNorm" => (lux_layer = Lux.LayerNorm,
                    args = ["normalized_shape" => 1],
                    kwargs = LAYERNORM_KWARGS,
                    tuple_args = ["normalized_shape"]))
const ACTIVATION_FUNCTIONS = Dict(
    "relu" => (fn = "Lux.relu", nargs = 1),
    "relu6" => (fn = "Lux.relu6", nargs = 1),
    "hardtanh" => (fn = "Lux.hardtanh", nargs = 1),
    "hardswish" => (fn = "Lux.hardswish", nargs = 1),
    "selu" => (fn = "Lux.selu", nargs = 1),
    "leaky_relu" => (fn = "Lux.leakyrelu", nargs = 1),
    "gelu" => (fn = "Lux.gelu", nargs = 1),
    "log_sigmoid" => (fn = "Lux.logsigmoid", nargs = 1),
    "tanhshrink" => (fn = "Lux.tanhshrink", nargs = 1),
    "softsign" => (fn = "Lux.softsign", nargs = 1),
    "softplus" => (fn = "Lux.softplus", nargs = 1),
    "sigmoid" => (fn = "Lux.sigmoid", nargs = 1),
    "tanh" => (fn = "Lux.tanh", nargs = 1),
    "hardsigmoid" => (fn = "Lux.hardsigmoid", nargs = 1),
    "mish" => (fn = "Lux.mish", nargs = 1),
    "elu" => (fn = Lux.elu, nargs = 2, args = ["alpha" => 2]),
    "celu" => (fn = Lux.celu, nargs = 2, args = ["alpha" => 2]),
    "leaky_relu" => (fn = Lux.leakyrelu, nargs = 2, args = ["negative_slope" => 2]),
    "rrelu" => (fn = Lux.rrelu, nargs = 3, args = ["lower" => 2, "upper" => 3]),
    "softshrink" => (fn = Lux.softshrink, nargs = 2, args = ["lambd" => 2]),
    "softmax" => (fn = Lux.softmax, nargs = 1, kwargs = ["dim" => "dims"]),
    "log_softmax" => (fn = Lux.logsoftmax, nargs = 1, kwargs = ["dim" => "dims"]))

# Maps below describe how to map weights between Lux and PyTorch (maps are bijective).
# As example [1 => 5] means that index position 1 in PyTorch is index poistion 5 in Lux.jl
const CONV1D_MAP = [1 => 3, 2 => 2, 3 => 1]
#=
    Julia (Lux.jl) and PyTorch encode images differently, and thus the W-matrix:
    In PyTorch: (in_chs, out_chs, H, W)
    In Julia  : (W, H, in_chs, out_chs)
=#
const CONV2D_MAP = [1 => 4, 2 => 3, 3 => 2, 4 => 1]
#=
    Julia (Lux.jl) and PyTorch encode 3d-images differently, and thus the W-matrix:
    In PyTorch: (in_chs, out_chs, D, H, W)
    In Julia  : (W, H, D, in_chs, out_chs)
=#
const CONV3D_MAP = [1 => 5, 2 => 4, 3 => 3, 4 => 2, 5 => 1]
#=
    Julia (Lux.jl) and PyTorch encode 3d-images differently, and thus
    In PyTorch: (C, D, H, W)
    In Julia  : (W, H, D, C)
=#
const LAYERNORM4_MAP = [1 => 4, 2 => 3, 3 => 2, 4 => 1]
#=
    Julia (Lux.jl) and PyTorch encode 2d-images differently, and thus
    In PyTorch: (C, H, W)
    In Julia  : (W, H, C)
=#
const LAYERNORM3_MAP = [1 => 3, 2 => 2, 3 => 1]
#=
    Julia (Lux.jl) and PyTorch encode 1d-images differently, and thus
    In PyTorch: (C, W)
    In Julia  : (W, C)
=#
const LAYERNORM2_MAP = [1 => 2, 2 => 1]
