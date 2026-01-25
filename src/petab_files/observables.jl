function parse_observables(
        modelname::String, paths::Dict{Symbol, String}, sys::ModelSystem,
        petab_tables::PEtabTables, xindices::ParameterIndices, speciemap_problem,
        speciemap_model, sys_observable_ids::Vector{Symbol},
        model_SBML::SBMLImporter.ModelSBML, ml_models::MLModels, write_to_file::Bool
    )::NTuple{4, String}
    state_ids = _get_state_ids(sys)

    _hstr = _parse_h(
        state_ids, sys_observable_ids, xindices, petab_tables, model_SBML, ml_models
    )
    _σstr = _parse_σ(
        state_ids, sys_observable_ids, xindices, petab_tables[:observables], model_SBML
    )
    _u0str = _parse_u0(
        speciemap_problem, speciemap_model, state_ids, xindices, model_SBML, false
    )
    _u0!str = _parse_u0(
        speciemap_problem, speciemap_model, state_ids, xindices, model_SBML, true
    )

    if write_to_file == true
        pathsave = joinpath(paths[:dirjulia], "$(modelname)_h_sd_u0.jl")
        strwrite = prod([_hstr, "\n\n", _u0!str, "\n\n", _u0str, "\n\n", _σstr])
        open(pathsave, "w") do f
            write(f, strwrite)
        end
    end
    return _hstr, _u0!str, _u0str, _σstr
end

function _parse_h(
        state_ids::Vector{String}, sys_observable_ids::Vector{Symbol},
        xindices::ParameterIndices, petab_tables::PEtabTables,
        model_SBML::SBMLImporter.ModelSBML, ml_models::MLModels
    )::String
    hstr = "function compute_h(__u_model::AbstractVector, t::Real, \
           __p_model::AbstractVector, xobservable::AbstractVector, \
           xnondynamic_mech::AbstractVector, x_ml_models, x_ml_models_constant, \
           nominal_values::Vector{Float64}, obsid::Symbol, \
           map::ObservableNoiseMap, __sys_observables, ml_models)::Real\n"

    observables_df = petab_tables[:observables]
    observable_ids = string.(observables_df[!, :observableId])
    for (i, obsid) in pairs(observable_ids)
        formula = filter(x -> !isspace(x), observables_df[i, :observableFormula] |> string)
        formula = _parse_formula(
            formula, state_ids, sys_observable_ids, xindices, model_SBML, :observable
        )
        obs_parameters = _get_observable_parameters(formula)
        formulas_nn = _get_ml_formulas(
            formula, petab_tables, state_ids, sys_observable_ids, xindices, model_SBML,
            ml_models, :observable
        )

        hstr *= "\tif obsid == :$(obsid)\n"
        hstr *= _template_obs_sd_parameters(obs_parameters; obs = true)
        hstr *= formulas_nn
        hstr *= "\t\treturn $formula \n"
        hstr *= "\tend\n"
    end
    hstr *= "end"
    return hstr
end

function _parse_σ(
        state_ids::Vector{String}, sys_observable_ids::Vector{Symbol},
        xindices::ParameterIndices, observables_df::DataFrame,
        model_SBML::SBMLImporter.ModelSBML
    )::String
    σstr = "function compute_σ(__u_model::AbstractVector, t::Real, \
            __p_model::AbstractVector, xnoise::AbstractVector, \
            xnondynamic_mech::AbstractVector, x_ml_models, x_ml_models_constant, \
            nominal_values::Vector{Float64}, obsid::Symbol, map::ObservableNoiseMap, \
            __sys_observables, nn)::Real\n"

    # Write the formula for standard deviations to file
    observable_ids = string.(observables_df[!, :observableId])
    for (i, obsid) in pairs(observable_ids)
        formula = filter(x -> !isspace(x), observables_df[i, :noiseFormula] |> string)
        formula = _parse_formula(
            formula, state_ids, sys_observable_ids, xindices, model_SBML, :noise
        )
        noise_parameters = _get_noise_parameters(formula)

        σstr *= "\tif obsid == :$(obsid)\n"
        σstr *= _template_obs_sd_parameters(noise_parameters; obs = false)
        σstr *= "\t\treturn $formula \n"
        σstr *= "\tend\n"
    end
    σstr *= "end"
    return σstr
end

function _parse_u0(
        speciemap_problem, speciemap_model, state_ids::Vector{String},
        xindices::ParameterIndices, model_SBML::SBMLImporter.ModelSBML, inplace::Bool
    )
    # As commented in PEtabModel files, for correct gradient the model must be mutated
    # if initial values are assigned in the conditions table. This corresponds to adding
    # an extra parameter. However, it is allowed for this parameter to map to NaN, in this
    # case, the original SBML or user provided initial value formula should be used.
    # Therefore, an isnan must be a part of the initial formulas, and this isnan is only
    # valid if post-equilibration is false, as is post-equilibration is true the value
    # from before a steady-state simulation should be used, which I can only detect if
    # NaN is set as initial value
    speciemap_problem_ids = replace.(string.(first.(speciemap_problem)), "(t)" => "")
    speciemap_model_ids = replace.(string.(first.(speciemap_model)), "(t)" => "")
    if inplace == true
        u0str = "function compute_u0!(__u0_model::AbstractVector, \
                __p_model::AbstractVector, __post_eq)\n"
    else
        u0str = "function compute_u0(__p_model::AbstractVector, __post_eq)::AbstractVector\n"
    end

    for id in state_ids
        im_problem = findfirst(x -> x == id, speciemap_problem_ids)
        im_model = findfirst(x -> x == id, speciemap_model_ids)
        u0formula_problem = _parse_formula(
            string(speciemap_problem[im_problem].second), state_ids, Symbol[], xindices,
            model_SBML, :u0
        )
        u0formula_model = _parse_formula(
            string(speciemap_model[im_model].second), state_ids, Symbol[], xindices,
            model_SBML, :u0
        )

        if u0formula_problem == u0formula_model
            u0str *= "\t$id = $(u0formula_problem)\n"
        else
            u0str *= "\tif isnan($(u0formula_problem)) && __post_eq == false\n"
            u0str *= "\t\t$id = $(u0formula_model)\n\telse\n"
            u0str *= "\t\t$id = $(u0formula_problem)\n\tend\n"
        end
    end

    if inplace == true
        u0str *= "\t__u0_model .= " * prod(state_ids .* ", ")[1:(end - 2)] * "\n"
    else
        inplace == false
        u0str *= "\treturn [" * prod(state_ids .* ", ")[1:(end - 2)] * "]\n"
    end
    u0str *= "end"
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

function _parse_formula(
        formula::String, state_ids::Vector{String}, sys_observable_ids::Vector{Symbol},
        xindices::ParameterIndices, model_SBML::SBMLImporter.ModelSBML, type::Symbol
    )::String
    # If the formula is defined as part of the observables in the system, the pre-built
    # symbolic funciton is the most efficient to use. This applies to SBML assignment
    # rules and observables defined via the PEtab interface
    if type in [:noise, :observable] && Symbol(formula) in sys_observable_ids
        formula = "__sys_observables[Symbol(\"$(formula)\")](__u_model, __p_model, t)"
        return formula
    end

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
        ids_replace = [
            :observable => "xobservable",
            :sys => "__p_model",
            :nondynamic_mech => "xnondynamic_mech",
            :petab => "nominal_values",
        ]
    elseif type == :noise
        ids_replace = [
            :noise => "xnoise",
            :sys => "__p_model",
            :nondynamic_mech => "xnondynamic_mech",
            :petab => "nominal_values",
        ]
    elseif type == :u0
        ids_replace = [:sys => "__p_model"]
    end
    for (idtype, varname) in ids_replace
        for (i, id) in pairs(string.(xindices.ids[idtype]))
            formula = SBMLImporter._replace_variable(formula, id, "$(varname)[$i]")
        end
    end
    for (i, id) in pairs(state_ids)
        formula = SBMLImporter._replace_variable(formula, id, "__u_model[$i]")
    end
    # In PEtab time is given by time, but it is t in Julia. For u0 time is zero
    if type == :u0
        formula = SBMLImporter._replace_variable(formula, "time", "0.0")
    else
        formula = SBMLImporter._replace_variable(formula, "time", "t")
    end
    return formula
end

function _get_ml_formulas(
        formula, petab_tables::PEtabTables, state_ids::Vector{String},
        sys_observable_ids::Vector{Symbol}, xindices::ParameterIndices,
        model_SBML::SBMLImporter.ModelSBML, ml_models::MLModels, type::Symbol
    )::String
    formula_nn = ""
    mappings_df = petab_tables[:mapping]
    isempty(mappings_df) && return formula_nn
    for ml_model in ml_models.ml_models
        ml_model.static == true && continue

        ml_id = ml_model.ml_id
        output_variables = _get_ml_model_io_petab_ids(mappings_df, ml_id, :outputs)
        has_nn_output = false
        for output_variable in Iterators.flatten(output_variables)
            if SBMLImporter._replace_variable(formula, output_variable, "") != formula
                has_nn_output = true
            end
        end
        has_nn_output == false && continue
        formula_nn *= _template_ml_observable(
            ml_id, petab_tables, state_ids, sys_observable_ids, xindices, model_SBML, type
        )
    end
    return formula_nn
end
