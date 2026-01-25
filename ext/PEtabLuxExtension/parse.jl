function PEtab.parse_to_lux(path_yaml::String; freeze_info::Union{Nothing, Dict} = nothing)
    network_yaml = YAML.load_file(path_yaml)
    layers = Dict([_parse_layer(l) for l in network_yaml["layers"]])
    inputs, outputs, forward_steps = _parse_forward_pass(network_yaml, layers)
    model_str = _template_nn_model(layers, inputs, outputs, forward_steps, freeze_info)
    nn = eval(Meta.parse(model_str)) |> f64
    return nn, network_yaml["nn_model_id"]
end

function _parse_forward_pass(
        network_yaml::Dict, layers::Dict
    )::Tuple{String, String, Vector{String}}
    forward_trace = network_yaml["forward"]
    input_arguments = network_yaml["inputs"]
    # To avoid naming conflicts, each input and output is given a __x__
    n_input_args = length(input_arguments)
    inputs = ""
    for i in 1:n_input_args
        inputs *= "__" * forward_trace[i]["name"] * "__"
        i == n_input_args && continue
        inputs *= ", "
    end
    outputs = "__" * forward_trace[end]["args"][1] * "__"

    forward_steps = fill("", length(forward_trace[(n_input_args + 1):(end - 1)]))
    for (i, step_info) in pairs(forward_trace[(n_input_args + 1):(end - 1)])
        # torch.cat arguments are provided as a Vector{Vector} due to being provided
        # in tuple inside torch
        if step_info["target"] == "cat"
            step_input = prod("__" .* step_info["args"][1] .* "__, ")
        else
            step_input = prod("__" .* step_info["args"] .* "__, ")
        end

        # Ensure tuple input if ninputs > 1
        if length(step_info["args"]) > 1
            step_input = "($step_input)"
        end
        step_output = "__" * step_info["name"] * "__"

        if haskey(layers, step_info["target"])
            _f = step_info["target"]
            forward_steps[i] = "$(step_output) = $(_f)($(step_input))"
        elseif haskey(ACTIVATION_FUNCTIONS, step_info["target"])
            _f = ACTIVATION_FUNCTIONS[step_info["target"]]
            forward_steps[i] = _parse_activation_function(
                step_output, step_input, step_info, _f
            )
        elseif step_info["target"] == "cat"
            forward_steps[i] = _parse_cat(step_output, step_input, step_info)
        else
            @error "Problem parsing forward step"
        end
    end
    return inputs, outputs, forward_steps
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
    return @error "Could not parse Flatten layer dimensions. This is best fixed providing \
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
    args_parsed[iarg] = (layer_parse["args"][p1], layer_parse["args"][p2]) => layer_parse["args"][argname2]
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

function _parse_activation_function(
        step_output::String, step_input::String, step_info::Dict, actinfo::NamedTuple
    )::String
    @assert length(step_info["args"]) == 1 "To many inputs to activation function \
        $(actinfo.fn)"
    # For activation functions with only 1 arg (e.g. tanh), via Lux.fast_activation Lux.jl
    # tries to find the fastest implementation
    if actinfo[:nargs] == 1 && !haskey(actinfo, :kwargs)
        return "$(step_output) = Lux.fast_activation($(actinfo.fn), $(step_input))"
    end
    # Remove last , in step input for parsing to work downstream
    step_input = step_input[1:(end - 2)]

    # Multiple input activation functions (e.g. elu). Note, parsed into Julia syntax
    args = fill("", actinfo.nargs)
    args[1] = step_input
    if actinfo.nargs > 1
        for (argname, argpos) in actinfo.args
            args[argpos] = string(step_info["kwargs"][argname])
        end
    end
    args = prod(args .* ", ")[1:(end - 2)]

    # Dim is a special keyword which must be adjusted between Lux and PyTorch as they are
    # 1 and 0 indexed based respectively
    if haskey(actinfo, :kwargs)
        kwargs = String[]
        for (argname, argname_julia) in actinfo.kwargs
            argval = step_info["kwargs"][argname]
            argval = argname == "dim" ? argval + 1 : argval
            push!(kwargs, "$(argname_julia) = $(argval), ")
        end
        kwargs = prod(kwargs)[1:(end - 2)]
        args = args * "; " * kwargs
    end
    return "$(step_output) = $(actinfo.fn)($args)"
end

function _parse_cat(step_output::String, step_input::String, step_info::Dict)::String
    args = step_input[1:(end - 2)]
    dim = step_info["kwargs"]["dim"] + 1
    return "$(step_output) = cat($(args); dims = $(dim))"
end

function _parse_freeze(ml_model::PEtab.MLModel, path_yaml::String)::Dict
    paths = PEtab._get_petab_paths(path_yaml)
    petab_tables = PEtab.read_tables_v2(path_yaml)
    petab_ml_parameters = PEtab.PEtabMLParameters(petab_tables, PEtab.MLModels(ml_model))
    ml_id = ml_model.ml_id

    i_ml = findall(x -> x == ml_id, petab_ml_parameters.ml_id)
    all(petab_ml_parameters.estimate[i_ml] .== false) && return Dict()
    all(petab_ml_parameters.estimate[i_ml] .== true) && return Dict()

    rng = Random.default_rng()
    ps, _ = Lux.setup(rng, ml_model.lux_model)
    ps = ComponentArray(ps) |> f64
    PEtab._set_ml_model_ps!(ps, ml_id, PEtab.MLModels(ml_model), paths, petab_tables)

    ml_model_indices = PEtab._get_ml_model_indices(ml_id, petab_ml_parameters.mapping_table_id)
    freeze_info = Dict{Symbol, Dict}()
    for ml_model_index in ml_model_indices[2:end]
        estimate = petab_ml_parameters.estimate[ml_model_index] == true
        mapping_table_id = string(petab_ml_parameters.mapping_table_id[ml_model_index])
        @assert count(".", mapping_table_id) ≤ 2 "Only two . are allowed when specifying \
            network layer"

        layer_id = PEtab._get_layer_id(mapping_table_id)
        array_id = PEtab._get_array_id(mapping_table_id)

        if !isempty(layer_id) && isempty(array_id)
            estimate == true && continue
            layer_id = Symbol(layer_id)
            array_ids = keys(ps[Symbol(layer_id)])
            freeze_info[layer_id] = Dict()
            for array_id in array_ids
                freeze_info[layer_id][array_id] = ps[layer_id][array_id]
            end
            continue
        end

        layer_id, array_id = Symbol.((layer_id, array_id))
        if !haskey(freeze_info, layer_id) && estimate == false
            freeze_info[layer_id] = Dict(array_id => ps[layer_id][array_id])

        elseif (
                haskey(freeze_info, layer_id) && haskey(freeze_info[layer_id], array_id) &&
                    estimate == true
            )
            delete!(freeze_info[layer_id], array_id)

        elseif haskey(freeze_info, layer_id) && estimate == false
            haskey(freeze_info[layer_id], array_id) && continue
            freeze_info[layer_id][array_id] = ps[layer_id][array_id]
        end
    end
    return freeze_info
end

function _parse_input!(array_inputs::Dict{Symbol, Array{<:Real}}, input, iarg)
    if input isa Array{<:Real}
        array_inputs[Symbol("__arg$(iarg)")] = input
        return :_ARRAY_INPUT
    else
        return Symbol.(input)
    end
end
