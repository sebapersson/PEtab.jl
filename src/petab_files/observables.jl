function parse_observables(modelname::String, paths::Dict{Symbol, String}, sys::ModelSystem,
                           observables_df::DataFrame, xindices::ParameterIndices, speciemap,
                           model_SBML::SBMLImporter.ModelSBML, mapping_table::DataFrame,
                           write_to_file::Bool)::NTuple{4, String}
    state_ids = _get_state_ids(sys)

    _hstr = _parse_h(state_ids, xindices, observables_df, model_SBML, mapping_table)
    _σstr = _parse_σ(state_ids, xindices, observables_df, model_SBML)
    _u0str = _parse_u0(speciemap, state_ids, xindices, model_SBML, false)
    _u0!str = _parse_u0(speciemap, state_ids, xindices, model_SBML, true)
    if write_to_file == true
        pathsave = joinpath(paths[:dirjulia], "$(modelname)_h_sd_u0.jl")
        strwrite = *(_hstr, _u0!str, _u0str, _σstr)
        open(pathsave, "w") do f
            write(f, strwrite)
        end
    end
    return _hstr, _u0!str, _u0str, _σstr
end

function _parse_h(state_ids::Vector{String}, xindices::ParameterIndices,
                  observables_df::DataFrame, model_SBML::SBMLImporter.ModelSBML,
                  mapping_table::DataFrame)::String
    hstr = "function compute_h(u::AbstractVector, t::Real, p::AbstractVector, \
            xobservable::AbstractVector, xnondynamic_mech::AbstractVector, xnn, \
            xnn_constant, nominal_values::Vector{Float64}, obsid::Symbol, \
            map::ObservableNoiseMap, nnmodels)::Real\n"

    observable_ids = string.(observables_df[!, :observableId])
    for (i, obsid) in pairs(observable_ids)
        formula = filter(x -> !isspace(x), observables_df[i, :observableFormula] |> string)
        formula = _parse_formula(formula, state_ids, xindices, model_SBML, :observable)
        obs_parameters = _get_observable_parameters(formula)
        formulas_nn = _get_formulas_nn(formula, mapping_table, state_ids, xindices,
                                       model_SBML, :observable)
        hstr *= "\tif obsid == :$(obsid)\n"
        hstr *= _template_obs_sd_parameters(obs_parameters; obs = true)
        hstr *= formulas_nn
        hstr *= "\t\treturn $formula \n"
        hstr *= "\tend\n"
    end
    hstr *= "end\n\n"
    return hstr
end

function _parse_σ(state_ids::Vector{String}, xindices::ParameterIndices,
                  observables_df::DataFrame, model_SBML::SBMLImporter.ModelSBML)::String
    σstr = "function compute_σ(u::AbstractVector, t::Real, p::AbstractVector, \
            xnoise::AbstractVector, xnondynamic_mech::AbstractVector, xnn, \
            xnn_constant, nominal_values::Vector{Float64}, obsid::Symbol, \
            map::ObservableNoiseMap, nn)::Real\n"

    # Write the formula for standard deviations to file
    observable_ids = string.(observables_df[!, :observableId])
    for (i, obsid) in pairs(observable_ids)
        formula = filter(x -> !isspace(x), observables_df[i, :noiseFormula] |> string)
        formula = _parse_formula(formula, state_ids, xindices, model_SBML, :noise)
        noise_parameters = _get_noise_parameters(formula)
        σstr *= "\tif obsid == :$(obsid)\n"
        σstr *= _template_obs_sd_parameters(noise_parameters; obs = false)
        σstr *= "\t\treturn $formula \n"
        σstr *= "\tend\n"
    end
    σstr *= "end\n\n"
    return σstr
end

function _parse_u0(speciemap, state_ids::Vector{String}, xindices::ParameterIndices,
                   model_SBML::SBMLImporter.ModelSBML, inplace::Bool)
    speciemap_ids = replace.(string.(first.(speciemap)), "(t)" => "")
    if inplace == true
        u0str = "function compute_u0!(u0::AbstractVector, p::AbstractVector)\n"
    else
        u0str = "function compute_u0(p::AbstractVector)::AbstractVector\n"
    end

    for id in state_ids
        im = findfirst(x -> x == id, speciemap_ids)
        u0formula = _parse_formula(string(speciemap[im].second), state_ids, xindices,
                                   model_SBML, :u0)
        u0str *= "\t$id = $u0formula\n"
    end

    if inplace == true
        u0str *= "\tu0 .= " * prod(state_ids .* ", ")[1:(end - 2)] * "\n"
    else
        inplace == false
        u0str *= "\treturn [" * prod(state_ids .* ", ")[1:(end - 2)] * "]\n"
    end
    u0str *= "end\n\n"
    return u0str
end

function _get_observable_parameters(formula::String)::Vector{String}
    obsp = [m.match for m in eachmatch(r"observableParameter[0-9]_\w+", formula)] |>
           unique |>
           sort .|>
           string
    return obsp
end

function _get_noise_parameters(formula::String)::Vector{String}
    noisep = [m.match for m in eachmatch(r"noiseParameter[0-9]_\w+", formula)] |>
             unique |>
             sort .|>
             string
    return noisep
end

function _template_obs_sd_parameters(parameters::Vector{String}; obs::Bool)::String
    xget = obs ? "xobservable" : "xnoise"
    if length(parameters) == 0
        return ""
    end
    if length(parameters) == 1
        return "\t\t" * parameters[1] * " = get_obs_sd_parameter($xget, map)[1]\n"
    end
    if length(parameters) > 1
        return "\t\t" * prod(parameters .* ", ") * "= get_obs_sd_parameter($xget, map)\n"
    end
end

function _parse_formula(formula::String, state_ids::Vector{String},
                        xindices::ParameterIndices, model_SBML::SBMLImporter.ModelSBML,
                        type::Symbol)::String
    formula = SBMLImporter.insert_functions(formula, PETAB_FUNCTIONS, PETAB_FUNCTIONS_NAMES)
    # It is possible to have math expressions on the form 3.0a instead of 3.0*a, this makes
    # the parsing harder and is fixed with this regex. The second regex is to adjust
    # for the first regex also incorrectly parsing sciencetific notation
    formula = replace(formula, r"(?<![\w])(\d+\.?\d*)([_a-zA-Z]+)" => s"\1*\2")
    formula = replace(formula, r"(?<![\w])(\d+\.?\d*)\*?([eE][+-]?\d+)" => s"\1*1\2")

    # SBML assignment rules can appear in observable and noise formulas, where they
    # should be inlined
    if type in [:noise, :observable]
        formula = SBMLImporter._inline_assignment_rules(formula, model_SBML)
    end
    if type == :observable
        ids_replace = [:observable => "xobservable", :sys => "p", :nondynamic_mech => "xnondynamic_mech", :petab => "nominal_values"]
    elseif type == :noise
        ids_replace = [:noise => "xnoise", :sys => "p", :nondynamic_mech => "xnondynamic_mech", :petab => "nominal_values"]
    elseif type == :u0
        ids_replace = [:sys => "p"]
    end
    for (idtype, varname) in ids_replace
        for (i, id) in pairs(string.(xindices.xids[idtype]))
            formula = SBMLImporter._replace_variable(formula, id, "$(varname)[$i]")
        end
    end
    for (i, id) in pairs(state_ids)
        formula = SBMLImporter._replace_variable(formula, id, "u[$i]")
    end
    # In PEtab time is given by time, but it is t in Julia. For u0 time is zero
    if type == :u0
        formula = SBMLImporter._replace_variable(formula, "time", "0.0")
    else
        formula = SBMLImporter._replace_variable(formula, "time", "t")
    end
    return formula
end

function _get_formulas_nn(formula, mapping_table::DataFrame, state_ids::Vector{String}, xindices::ParameterIndices, model_SBML::SBMLImporter.ModelSBML, type::Symbol)::String
    formula_nn = ""
    isempty(mapping_table) && return formula_nn
    for netid in unique(_get_netids(mapping_table))
        outputs = _get_net_petab_variables(mapping_table, Symbol(netid), :outputs)
        has_nn_output = false
        for output in outputs
            if SBMLImporter._replace_variable(formula, output, "") != formula
                has_nn_output = true
            end
        end
        has_nn_output == false && continue
        formula_nn *= _template_nn_formula(Symbol(netid), mapping_table, state_ids, xindices, model_SBML, type)
    end
    return formula_nn
end

function _template_nn_formula(netid::Symbol, mapping_table::DataFrame, state_ids::Vector{String}, xindices::ParameterIndices, model_SBML::SBMLImporter.ModelSBML, type::Symbol)::String
    inputs = "[" * prod(_get_net_petab_variables(mapping_table, netid, :inputs) .* ",") * "]"
    inputs = _parse_formula(inputs, state_ids, xindices, model_SBML, type)
    outputs = prod(_get_net_petab_variables(mapping_table, netid, :outputs) .* ", ")
    formula = "\n\t\tnnmodel_$(netid) = nnmodels[:$(netid)]\n"
    if netid in xindices.xids[:nn_est]
        formula *= "\t\txnn_$(netid) = xnn[:$(netid)]\n"
    else
        formula *= "\t\txnn_$(netid) = xnn_constant[:$(netid)]\n"
    end
    formula *= "\t\tout, st_$(netid) = nnmodel_$(netid).nn($inputs, xnn_$(netid), nnmodel_$(netid).st)\n"
    formula *= "\t\t$(outputs) = out\n"
    formula *= "\t\tnnmodel_$(netid).st = st_$(netid)\n"
    return formula
end
