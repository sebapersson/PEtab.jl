function _template_odeproblem(model_SBML_prob, model_SBML, nnmodels_in_ode::Dict, petab_tables::PEtabTables)::String
    @unpack umodel, ps, odes = model_SBML_prob
    fode = "function f_$(model_SBML.name)(du, u, p, t, nnmodels)::Nothing\n"
    fode *= "\t" * prod(umodel .* ", ") * " = u\n"
    fode *= "\t@unpack " * prod(ps .* ", ") * " = p\n"
    for netid in keys(nnmodels_in_ode)
        fode *= _template_nn_in_ode(netid, petab_tables)
    end
    for ode in odes
        fode *= "\t" * ode
    end
    fode *= "\treturn nothing\n"
    fode *= "end"
    return fode
end

function _template_nn_in_ode(netid::Symbol, petab_tables::PEtabTables)::String
    hybridization_df = petab_tables[:hybridization]
    mappings_df = petab_tables[:mapping_table]

    input_variables = _get_net_petab_variables(mappings_df, netid, :inputs)
    inputs_df = filter(r -> r.targetId in input_variables, hybridization_df)
    input_expressions = inputs_df.targetValue
    output_variables = _get_net_petab_variables(mappings_df, netid, :outputs)
    outputs_df = filter(row -> row.targetValue in output_variables, hybridization_df)
    output_targets = ""
    for (i, output_target) in pairs(outputs_df.targetId)
        output_targets *= "\t$output_target = " * outputs_df.targetValue[i] * "\n"
    end

    inputs = "[" * prod(input_expressions .* ",") * "]"
    outputs_p = prod(output_variables .* ", ")
    outputs_net = "out, st_$(netid)"
    formula = "\n\tnnmodel_$(netid) = nnmodels[:$(netid)]\n"
    formula *= "\txnn_$(netid) = p[:$(netid)]\n"
    formula *= "\t$(outputs_net) = nnmodel_$(netid).nn($inputs, xnn_$(netid), nnmodel_$(netid).st)\n"
    formula *= "\tnnmodel_$(netid).st = st_$(netid)\n"
    formula *= "\t$(outputs_p) = out\n"
    formula *= "$(output_targets)\n\n"
    return formula
end
