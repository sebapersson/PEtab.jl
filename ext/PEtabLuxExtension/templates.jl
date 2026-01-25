function _template_nn_model(
        layers, inputs, output, forward_steps, freeze_info::Union{Nothing, Dict}
    )::String
    model_str = "@compact(\n"
    for (id, layer) in layers
        if isnothing(freeze_info) || !haskey(freeze_info, Symbol(id))
            model_str *= "\t$(id) = " * string(layer) * ",\n"
            continue
        end
        which_params = keys(freeze_info[Symbol(id)]) |> collect .|> string
        which_params = '(' * prod(":" .* which_params .* ", ") * ')'
        model_str *= "\t$(id) = Lux.Experimental.freeze(" * string(layer) *
            ", $(which_params)),\n"
    end
    model_str *= ") do ($(inputs))\n"
    for forward_step in forward_steps
        model_str *= "\t$(forward_step)\n"
    end
    model_str *= "\t@return $(output)\nend"
    return model_str
end
