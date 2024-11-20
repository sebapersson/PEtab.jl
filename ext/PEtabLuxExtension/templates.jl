function _template_nn_model(layers, input, output, forward_steps)::String
    model_str = "@compact(\n"
    for (id, layer) in layers
        model_str *= "\t$(id) = " * string(layer) * ",\n"
    end
    model_str *= ") do $(input)\n"
    for forward_step in forward_steps
        model_str *= "\t$(forward_step)\n"
    end
    model_str *= "\t@return $(output)\nend"
    return model_str
end
