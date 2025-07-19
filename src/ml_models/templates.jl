function _template_odeproblem(model_SBML_prob, model_SBML, ml_models_in_ode::Dict, petab_tables::PEtabTables)::String
    @unpack umodel, ps, odes = model_SBML_prob
    fode = "function f_$(model_SBML.name)(du, u, p, t, ml_models)::Nothing\n"
    fode *= "\t" * prod(umodel .* ", ") * " = u\n"
    fode *= "\t@unpack " * prod(ps .* ", ") * " = p\n"
    for ml_model_id in keys(ml_models_in_ode)
        fode *= _template_nn_in_ode(ml_model_id, petab_tables)
    end
    for ode in odes
        fode *= "\t" * ode
    end
    fode *= "\treturn nothing\n"
    fode *= "end"
    return fode
end

function _template_nn_in_ode(ml_model_id::Symbol, petab_tables::PEtabTables)::String
    hybridization_df = petab_tables[:hybridization]
    mappings_df = petab_tables[:mapping]

    input_variables = get_ml_model_petab_variables(mappings_df, ml_model_id, :inputs)
    inputs_df = filter(r -> r.targetId in input_variables, hybridization_df)
    input_expressions = inputs_df.targetValue
    output_variables = get_ml_model_petab_variables(mappings_df, ml_model_id, :outputs)
    outputs_df = filter(row -> row.targetValue in output_variables, hybridization_df)
    output_targets = ""
    for (i, output_target) in pairs(outputs_df.targetId)
        output_targets *= "\t$output_target = " * outputs_df.targetValue[i] * "\n"
    end

    inputs = "[" * prod(input_expressions .* ",") * "]"
    outputs_p = prod(output_variables .* ", ")
    outputs_net = "out, st_$(ml_model_id)"
    formula = "\n\tml_model_$(ml_model_id) = ml_models[:$(ml_model_id)]\n"
    formula *= "\txnn_$(ml_model_id) = p[:$(ml_model_id)]\n"
    formula *= "\t$(outputs_net) = ml_model_$(ml_model_id).model($inputs, xnn_$(ml_model_id), ml_model_$(ml_model_id).st)\n"
    formula *= "\tml_model_$(ml_model_id).st = st_$(ml_model_id)\n"
    formula *= "\t$(outputs_p) = out\n"
    formula *= "$(output_targets)\n\n"
    return formula
end
