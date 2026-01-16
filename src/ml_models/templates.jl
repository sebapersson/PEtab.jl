function _template_odeproblem(model_SBML_prob, model_SBML, ml_models_in_ode::Dict, petab_tables::PEtabTables)::String
    @unpack umodel, ps, odes = model_SBML_prob
    fode = "function f_$(model_SBML.name)(du, u, p, t, ml_models)::Nothing\n"
    fode *= "\t" * prod(umodel .* ", ") * " = u\n"
    fode *= "\t@unpack " * prod(ps .* ", ") * " = p\n"
    for ml_id in keys(ml_models_in_ode)
        fode *= _template_ml_in_ode(ml_id, petab_tables)
    end
    for ode in odes
        fode *= "\t" * ode
    end
    fode *= "\treturn nothing\n"
    fode *= "end"
    return fode
end

function _template_ml_in_ode(ml_id::Symbol, petab_tables::PEtabTables)::String
    hybridization_df = petab_tables[:hybridization]
    mappings_df = petab_tables[:mapping]

    input_arguments = _get_ml_model_io_petab_ids(mappings_df, ml_id, :inputs)
    inputs = "("
    for (i, input_argument) in pairs(input_arguments)
        df = filter(r -> r.targetId in input_argument, hybridization_df)
        inputs *= "[" * prod(df.targetValue .* ",") * "]"
        if i == length(input_arguments)
            inputs *= ')'
        else
            inputs *= ", "
        end
    end

    output_variables = _get_ml_model_io_petab_ids(mappings_df, ml_id, :outputs) |>
        Iterators.flatten
    outputs_df = filter(row -> row.targetValue in output_variables, hybridization_df)
    output_targets = ""
    for (i, output_target) in pairs(outputs_df.targetId)
        output_targets *= "\t$output_target = " * outputs_df.targetValue[i] * "\n"
    end

    outputs_p = prod(output_variables .* ", ")
    outputs_net = "out, st_$(ml_id)"
    formula = "\n\tml_model_$(ml_id) = ml_models[:$(ml_id)]\n"
    formula *= "\tx_ml_$(ml_id) = p[:$(ml_id)]\n"
    formula *= "\t$(outputs_net) = ml_model_$(ml_id).model($inputs, x_ml_$(ml_id), ml_model_$(ml_id).st)\n"
    formula *= "\tml_model_$(ml_id).st = st_$(ml_id)\n"
    formula *= "\t$(outputs_p) = out\n"
    formula *= "$(output_targets)\n\n"
    return formula
end

function _template_ml_observable(ml_id::Symbol, petab_tables::PEtabTables, state_ids::Vector{String},  sys_observable_ids::Vector{Symbol}, xindices::ParameterIndices, model_SBML::SBMLImporter.ModelSBML, type::Symbol)::String
    mappings_df = petab_tables[:mapping]
    hybridization_df = petab_tables[:hybridization]

    input_arguments = _get_ml_model_io_petab_ids(mappings_df, ml_id, :inputs)
    inputs = "("
    for (i, input_argument) in pairs(input_arguments)
        df = filter(r -> r.targetId in input_argument, hybridization_df)
        inputs *= "[" * prod(df.targetValue .* ",") * "]"
        if i == length(input_arguments)
            inputs *= ')'
        else
            inputs *= ", "
        end
    end
    inputs = _parse_formula(inputs, state_ids, sys_observable_ids, xindices, model_SBML, type)

    output_variables = _get_ml_model_io_petab_ids(mappings_df, Symbol(ml_id), :outputs) |>
        Iterators.flatten
    outputs = prod(output_variables .* ", ")

    formula = "\n\t\tml_model_$(ml_id) = ml_models[:$(ml_id)]\n"
    if ml_id in xindices.xids[:ml_in_ode]
        formula *= "\t\tx_ml_$(ml_id) = __p_model[:$(ml_id)]\n"
    elseif ml_id in xindices.xids[:ml_est]
        formula *= "\t\tx_ml_$(ml_id) = x_ml_models[:$(ml_id)]\n"
    else
        formula *= "\t\tx_ml_$(ml_id) = x_ml_models_constant[:$(ml_id)]\n"
    end
    formula *= "\t\tout, st_$(ml_id) = ml_model_$(ml_id).model($inputs, x_ml_$(ml_id), ml_model_$(ml_id).st)\n"
    formula *= "\t\t$(outputs) = out\n"
    formula *= "\t\tml_model_$(ml_id).st = st_$(ml_id)\n"
    return formula
end
