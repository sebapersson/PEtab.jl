function _template_odeproblem(model_SBML_prob, model_SBML, nnmodels_in_ode, mapping_table::DataFrame)::String
    @unpack umodel, ps, odes = model_SBML_prob
    fode = "function f_$(model_SBML.name)(du, u, p, t, nn)::Nothing\n"
    fode *= "\t" * prod(umodel .* ", ") * " = u\n"
    fode *= "\t@unpack " * prod(ps .* ", ") * " = p\n"
    for netid in keys(nnmodels_in_ode)
        fode *= _template_nn_in_ode(netid, mapping_table)
    end
    for ode in odes
        fode *= "\t" * ode
    end
    fode *= "\treturn nothing\n"
    fode *= "end"
    return fode
end

function _template_nn_in_ode(netid::Symbol, mapping_table)
    inputs = "[" * prod(PEtab._get_net_values(mapping_table, netid, :inputs) .* ",") * "]"
    outputs_p = prod(PEtab._get_net_values(mapping_table, netid, :outputs) .* ", ")
    outputs_net = "out, st_$(netid)"
    formula = "\n\tst_$(netid), net_$(netid) = nn[:$(netid)]\n"
    formula *= "\txnn_$(netid) = p[:p_$(netid)]\n"
    formula *= "\t$(outputs_net) = net_$(netid)($inputs, xnn_$(netid), st_$(netid))\n"
    formula *= "\tnn[:$(netid)][1] = st_$(netid)\n"
    formula *= "\t$(outputs_p) = out\n\n"
    return formula
end
