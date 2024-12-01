function PEtab.load_nnmodels(path_yaml::String)::Dict{Symbol, <:PEtab.NNModel}
    yaml_file = YAML.load_file(path_yaml)
    sciml_info = yaml_file["extensions"]["petab_sciml"]
    nnmodels = Dict{Symbol, PEtab.NNModel}()
    for _netfile in sciml_info["net_files"]
        netfile = joinpath(dirname(path_yaml), _netfile)
        net, netid = parse_to_lux(netfile)
        input_info, output_info = String[], String[]
        for (id, nninfo) in sciml_info["hybridization"][netid]
            if id == "input"
                input_info = nninfo isa Vector ? nninfo : [nninfo]
            elseif id == "output"
                output_info = nninfo isa Vector ? nninfo : [nninfo]
            end
        end
        nnmodels[Symbol(netid)] = PEtab.NNModel(net; input_info = input_info, output_info = output_info)
    end
    return nnmodels
end

function PEtab.NNModel(net::Union{Lux.Chain, Lux.CompactLuxLayer}; dirdata = nothing, inputs::Vector{T} = Symbol[], outputs::Vector{T} = Symbol[], input_info::Vector{String} = String[], output_info = String[])::NNModel where T <: Union{String, Symbol}
    rng = Random.default_rng()
    st = Lux.initialstates(rng, net)
    if isnothing(dirdata)
        dirdata = ""
    elseif !isdir(dirdata)
        throw(PEtab.PEtabInputError("For a NNmodel dirdata keyword argument must be a \
            valid directory. This does not hold for $dirdata"))
    end
    if (!isempty(inputs) && isempty(outputs)) || (isempty(inputs) && !isempty(outputs))
        throw(PEtab.PEtabInputError("If either input or output is provided to a NNModel \
            then both input and output must be provided"))
    end
    _inputs = Symbol.(inputs)
    _outputs = Symbol.(outputs)
    return NNModel(net, st, dirdata, _inputs, _outputs, input_info, output_info)
end

function PEtab.parse_to_lux(path_yaml::String)
    network_yaml = YAML.load_file(path_yaml)["models"][1]
    layers = Dict([_parse_layer(l) for l in network_yaml["layers"]])
    input, output, forward_steps = _parse_forward_pass(network_yaml["forward"], layers)
    model_str = _template_nn_model(layers, input, output, forward_steps)
    return eval(Meta.parse(model_str)), network_yaml["mlmodel_id"]
end

function _parse_forward_pass(forward_trace::Vector{<:Dict}, layers::Dict)::Tuple{String, String, Vector{String}}
    # To avoid naming conflicts, each input and output is given a __x__
    input_arg = "__" * forward_trace[1]["name"] * "__"
    output_arg = "__" * forward_trace[end]["args"][1] * "__"

    forward_steps = fill("", length(forward_trace[2:end-1]))
    for (i, stepinfo) in pairs(forward_trace[2:end-1])
        step_input = prod("__" .* stepinfo["args"] .* "__, ")
        # Ensure tuple input if ninputs > 1
        if length(stepinfo["args"]) > 1
            step_input = "($step_input)"
        end
        step_output = "__" * stepinfo["name"] * "__"

        if haskey(layers, stepinfo["target"])
            _f = stepinfo["target"]
            forward_steps[i] = "$(step_output) = $(_f)($(step_input))"
        elseif haskey(ACTIVATION_FUNCTIONS, stepinfo["target"])
            _f = ACTIVATION_FUNCTIONS[stepinfo["target"]]
            forward_steps[i] = _parse_activation_function(step_output, step_input, stepinfo, _f)
        else
            @error "Problem parsing forward step"
        end
    end
    return input_arg, output_arg, forward_steps
end

function _parse_layer(layer_parse::Dict)
    layer_info = LAYERS[layer_parse["layer_type"]]

    # Most layers can be parsed the same way, but flatten differes quite substantially
    # between PyTorch and Lux.
    if layer_parse["layer_type"] == "Flatten"
        return _parse_flatten_layer(layer_parse)
    end

    _args = Vector{Any}(undef, length(layer_info.args))
    for arg in layer_info.args
        _parse_layer_arg!(_args, arg, layer_parse, layer_info)
    end

    _kwargs = _parse_layer_kwargs(layer_parse, layer_info)

    return layer_parse["layer_id"] => layer_info.lux_layer(_args...; _kwargs...)
end

function _parse_flatten_layer(layer_parse)
    end_dim, start_dim = layer_parse["args"]["end_dim"], layer_parse["args"]["start_dim"]
    # Default most common case, only reduces dimension by 1 in the input tensor
    if start_dim == 1 && end_dim == -1
        return layer_parse["layer_id"] => "Lux.FlattenLayer()"
    elseif start_dim == 0 && end_dim == -1
        return layer_parse["layer_id"] => "vec"
    end
    @error "Could not parse Flatten layer dimensions. This is best fixed providing \
            start_dim and end_dim. If start_dim > 1 Julia cannot unfortunately flatten"
end

function _parse_layer_arg!(args_parsed, arg, layer_parse, layer_info)::Nothing
    argname, iarg = arg
    # Most args in Julia are on a simple number/tuple, while some (after if-statement) are
    # pair: a => b, which must be considered during parsing.
    if !occursin("=>", argname)
        val = layer_parse["args"][argname]
        if val isa Vector
            val = Tuple(val)
        # A subset of args must be tuple (while they can be a single int in PyTorch)
        elseif haskey(layer_info, :tuple_args) && argname in layer_info[:tuple_args]
            val = Tuple(val)
        end
        # Lux.jl prefers images in the foramt [W, H] and PyTorch [H, W] to store the images
        # in memory order. Therefore, for operations with kernels (e.g. Conv, Pool...), the
        # kernel order is reversed.
        if argname in ["kernel_size", "output_size", "normalized_shape"]
            args_parsed[iarg] = reverse(val)
        else
            args_parsed[iarg] = val
        end
        return nothing
    end
    # Parsing arguments that must be on the form a => b
    argname1, argname2 = replace.(split(argname, "=>"), " " => "")
    if !occursin(",", argname1)
        args_parsed[iarg] = layer_parse["args"][argname1] => layer_parse["args"][argname2]
        return nothing
    end
    # For Bilinear we have (a, b) => c
    argname1 = replace(argname1, r"\(|\)" => "")
    p1, p2 = split(argname1, ',')
    args_parsed[iarg] = (layer_parse["args"][p1], layer_parse["args"][p2])  => layer_parse["args"][argname2]
    return nothing
end

function _parse_layer_kwargs(layer_parse, layer_info)::NamedTuple
    # kwargs/args common between Lux and PyTorch
    kwarg_vals = Any[]
    kwarg_ids = Symbol[]
    for (kwarg, kwarg_julia) in layer_info.kwargs
        !haskey(layer_parse["args"], kwarg) && continue
        val = layer_parse["args"][kwarg]
        # if nothing, use the fact that Lux and PyTorch has the same defaults
        if isnothing(val)
            continue
        elseif val isa Vector
            val = Tuple(val)
        elseif haskey(layer_info, :tuple_args) && kwarg in layer_info[:tuple_args]
            val = Tuple(val)
        end
        # Account for images being stored in different order between Lux.jl and PyTorch
        if kwarg_julia in ["pad", "stride", "dilation"] && val isa Tuple
            val = reverse(val)
        end
        push!(kwarg_vals, val)
        push!(kwarg_ids, Symbol(kwarg_julia))
    end
    kwargs = NamedTuple{Tuple(kwarg_ids)}(kwarg_vals)
    # kwargs only in Julia. These arise from the fact that PyTorch has functions like
    # Dropout1d, Dropout2d, while Julia has only 1 Dropout function, and the PyTorch
    # functionality is obtained via keyword arguments
    if haskey(layer_info, :kwargs_julia)
        kwargs = merge(kwargs, layer_info[:kwargs_julia])
    end
    return kwargs
end

function _parse_activation_function(step_output::String, step_input::String, stepinfo::Dict, actinfo::NamedTuple)::String
    @assert length(stepinfo["args"]) == 1 "To many inputs to activation function $(actinfo.fn)"
    # For activation functions with only 1 arg (e.g. tanh), via Lux.fast_activation Lux.jl
    # tries to find the fastest implementation
    if actinfo[:nargs] == 1 && !haskey(actinfo, :kwargs)
        return "$(step_output) = Lux.fast_activation($(actinfo.fn), $(step_input))"
    end
    # Remove last , in step input for parsing to work downstream
    step_input = step_input[1:end-2]

    # Multiple input activation functions (e.g. elu). Note, parsed into Julia syntax
    args = fill("", actinfo.nargs)
    args[1] = step_input
    if actinfo.nargs > 1
        for (argname, argpos) in actinfo.args
            args[argpos] = string(stepinfo["kwargs"][argname])
        end
    end
    args = prod(args .* ", ")[1:end-2]

    # Dim is a special keyword which must be adjusted between Lux and PyTorch as they are
    # 1 and 0 indexed based respectively
    if haskey(actinfo, :kwargs)
        kwargs = String[]
        for (argname, argname_julia) in actinfo.kwargs
            argval = stepinfo["kwargs"][argname]
            argval = argname == "dim" ? argval + 1 : argval
            push!(kwargs, "$(argname_julia) = $(argval), ")
        end
        kwargs = prod(kwargs)[1:end-2]
        args = args * "; " * kwargs
    end
    return "$(step_output) = $(actinfo.fn)($args)"
end