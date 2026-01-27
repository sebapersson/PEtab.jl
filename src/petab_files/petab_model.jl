"""
    PEtabModel(path_yaml; kwargs...)

Import a PEtab problem in the standard (YAML + tables) format from `path_yaml` as a
`PEtabModel` for parameter estimation.

# Keyword Arguments
- `ifelse_to_callback::Bool = true`: Rewrite `ifelse` (SBML piecewise) expressions as
  [callbacks](https://github.com/SciML/DiffEqCallbacks.jl). Typically improves simulation
  performance.
- `write_to_file::Bool = false`: Write generated Julia functions to
  `dirname(path_yaml)/Julia_model_files/` (useful for debugging).
- `verbose::Bool = false`: Print progress while building the model.
"""
function PEtabModel(
        path_yaml::String; build_julia_files::Bool = true, verbose::Bool = false,
        ifelse_to_callback::Bool = true, write_to_file::Bool = false,
        ml_models::Union{Nothing, MLModels, MLModel} = nothing
    )::PEtabModel
    petab_version = _get_version(path_yaml)

    if petab_version == "1.0.0"
        petab_tables = read_tables_v1(path_yaml)
        petab_events = PEtabEvent[]
    else
        petab_tables, petab_events = v2_to_v1_tables(path_yaml, ifelse_to_callback)
    end

    # Sanity check MLModels are correctly associated with the problem
    sciml_extension = _has_sciml_extension(path_yaml)
    if sciml_extension && isnothing(ml_models)
        throw(PEtabInputError("For PEtab problems loaded with the SciML extension, \
            `ml_models` must be provided to `PEtabModel` as \
            `PEtabModel(...; ml_models = ml_models)`, where `ml_models` associated with \
            the PEtab SciML problem can be loaded with `ml_models = MLModels(path_yaml)`"))
    end
    if petab_version == "1.0.0" && !isnothing(ml_models)
        throw(PEtabInputError("PEtab problems with SciML extension are only supported for \
            PEtab v2 problems."))
    end

    ml_models = if isnothing(ml_models)
        MLModels()
    elseif ml_models isa MLModel
        MLModels(ml_models)
    else
        ml_models
    end

    paths = _get_petab_paths(path_yaml)
    return _PEtabModel(
        paths, petab_tables, build_julia_files, verbose, ifelse_to_callback,
        write_to_file, petab_events, ml_models
    )
end

function _PEtabModel(
        paths::Dict{Symbol, String}, petab_tables::PEtabTables, build_julia_files::Bool,
        verbose::Bool, ifelse_to_callback::Bool, write_to_file::Bool,
        petab_events::Vector{PEtabEvent}, ml_models::MLModels
    )
    name = splitdir(paths[:dirmodel])[end]

    write_to_file && !isdir(paths[:dirjulia]) && mkdir(paths[:dirjulia])
    if write_to_file == false && build_julia_files == true
        paths[:dirjulia] = ""
    end
    _logging(:Build_PEtabModel, verbose; name = name)

    # Ensure correct type internally for ml_models
    ml_models = isnothing(ml_models) ? MLModels() : ml_models

    #=
        If the SBML model contains a neural-network it must be parsed as an ODEProblem, as
        MTK does not have good neural-net support yet. Otherwise, model should be parsed as
        ODESystem as usual. If parsed as ODEProblem assignment rules must be inlined

        In case one of the conditions in the PEtab table assigns an initial specie value,
        the SBML model must be mutated to add an initial value parameter in order for
        PEtab.jl to be able to correctly compute gradients
    =#
    btime = @elapsed begin
        ode_ml_models = _get_ode_ml_models(ml_models, paths[:SBML], petab_tables)

        inline_assignment_rules = !isempty(ode_ml_models)
        model_SBML = SBMLImporter.parse_SBML(
            paths[:SBML], false; model_as_string = false,
            ifelse_to_callback = ifelse_to_callback,
            inline_assignment_rules = inline_assignment_rules
        )
        speciemap_sbml = _get_sbml_speciemap(model_SBML)
        add_u0_parameters!(model_SBML, petab_tables, ml_models)

        pathmodel = joinpath(paths[:dirjulia], name * ".jl")
        exist = isfile(pathmodel)
        if isempty(ode_ml_models)
            odesystem, speciemap, parametermap = _get_odesys(
                model_SBML, paths, exist, build_julia_files, write_to_file, verbose
            )
        else
            odesystem, speciemap, parametermap = _get_odeproblem(
                model_SBML, ode_ml_models, petab_tables, verbose
            )
        end

        sys_observables = _get_sys_observables(odesystem)
        sys_observable_ids = collect(keys(sys_observables))
    end

    # Indices for mapping parameters and tracking which parameter to estimate, needed
    # to build observable functions, and callbacks
    xindices = ParameterIndices(
        petab_tables, paths, odesystem, parametermap, speciemap, ml_models
    )

    path_u0_h_σ = joinpath(paths[:dirjulia], "$(name)_h_sd_u0.jl")
    exist = isfile(path_u0_h_σ)
    _logging(:Build_u0_h_σ, verbose; buildfiles = build_julia_files, exist = exist)
    if !exist || build_julia_files == true
        btime = @elapsed begin
            hstr, u0!str, u0str, σstr = parse_observables(
                name, paths, odesystem, petab_tables, xindices, speciemap, speciemap_sbml,
                sys_observable_ids, model_SBML, ml_models, write_to_file
            )
        end
        _logging(:Build_u0_h_σ, verbose; time = btime)
    else
        hstr, u0!str, u0str, σstr = _get_functions_as_str(path_u0_h_σ, 4)
    end
    compute_h = @RuntimeGeneratedFunction(Meta.parse(hstr))
    compute_σ = @RuntimeGeneratedFunction(Meta.parse(σstr))
    # Compute u0 should be default have two inputs for being able to compute gradients.
    # __post_eq should only be provided to solve after a pre-equlibration, to track
    # potential NaN values in the conditions table
    _compute_u0! = @RuntimeGeneratedFunction(Meta.parse(u0!str))
    _compute_u0 = @RuntimeGeneratedFunction(Meta.parse(u0str))
    compute_u0! = let f_u0! = _compute_u0!
        (u0, p; __post_eq = false) -> f_u0!(u0, p, __post_eq)
    end
    compute_u0 = let f_u0 = _compute_u0
        (p; __post_eq = false) -> f_u0(p, __post_eq)
    end

    # SBMLImporter holds the callback building functionality. However, currently it needs
    # to know the ODESystem parameter order (psys), and whether or not any parameters
    # which are estimated are present in the event condition. For the latter, timespan
    # should not be converted to floats in case dual numbers (for gradients) are propagated
    _logging(:Build_callbacks, verbose)
    btime = @elapsed begin
        cbs, float_tspan = _parse_events(
            model_SBML, petab_events, odesystem, speciemap, parametermap, name, xindices,
            petab_tables
        )
    end
    _logging(:Build_callbacks, verbose; time = btime)

    return PEtabModel(
        name, compute_h, compute_u0!, compute_u0, compute_σ, float_tspan, paths, odesystem,
        deepcopy(odesystem), parametermap, speciemap, petab_tables, cbs, false,
        petab_events, sys_observables, ml_models
    )
end

function add_u0_parameters!(
        model_SBML::SBMLImporter.ModelSBML, petab_tables::PEtabTables, ml_models::MLModels
    )::Nothing
    conditions_df = petab_tables[:conditions]
    parameters_df = petab_tables[:parameters]
    mappings_df = petab_tables[:mapping]
    hybridization_df = petab_tables[:hybridization]

    specieids = keys(model_SBML.species)
    rateruleids = model_SBML.rate_rule_variables
    sbml_variables = Iterators.flatten((specieids, rateruleids)) |> unique
    condition_variables = names(conditions_df)

    # Neural net output variables can set initial values, in this case the initial
    # value must be converted to a parameter
    net_outputs = String[]
    for ml_model in ml_models.ml_models
        ml_id = ml_model.ml_id
        ml_model.static == false && continue
        output_variables = _get_ml_model_io_petab_ids(mappings_df, ml_id, :outputs) |>
            Iterators.flatten
        outputs_df = filter(row -> row.targetValue in output_variables, hybridization_df)
        net_outputs = vcat(net_outputs, outputs_df.targetId)
    end

    variables = Iterators.flatten((condition_variables, net_outputs))
    if any(x -> x in sbml_variables, variables) == false
        return nothing
    end

    for condition_variable in variables
        !(condition_variable in sbml_variables) && continue

        if condition_variable in specieids
            sbml_variable = model_SBML.species[condition_variable]
        else
            sbml_variable = model_SBML.parameters[condition_variable]
        end
        u0name = "__init__" .* sbml_variable.name .* "__"
        value = sbml_variable.initial_value
        u0parameter = SBMLImporter.ParameterSBML(
            u0name, true, value, "", false, false,
            false, false, false, false
        )
        model_SBML.parameters[u0name] = u0parameter
        sbml_variable.initial_value = u0name

        # Rename output in the hybridization table, to have the neural-net map to the
        # initial-value parameter instead
        if condition_variable in net_outputs
            ix = findall(x -> x == condition_variable, hybridization_df.targetId)
            hybridization_df[ix, :targetId] .= "__init__" .* sbml_variable.name .* "__"
        end

        # Check if any parameter in the PEtab tables maps to u0 in the conditions table,
        # because if this is the case this parameter must be added to the SBML model as
        # it should be treated as a dynamic parameter for indexing. A NaN value is allowed,
        # meaning the parameter should be set by the corresponding SBML formula
        !(condition_variable in names(conditions_df)) && continue
        for condition_value in conditions_df[!, condition_variable]
            ismissing(condition_value) && continue
            condition_value == "NaN" && continue
            condition_value isa Real && continue
            is_number(condition_value) && continue
            # Potential parameters case
            for parameter_id in parameters_df.parameterId
                # Whether the formula contains the parameter
                _formula = SBMLImporter._replace_variable(condition_value, parameter_id, "")
                _formula == condition_value && continue

                haskey(model_SBML.parameters, parameter_id) && continue

                parameter = SBMLImporter.ParameterSBML(
                    condition_value, true, "0.0", "",
                    false, false, false, false, false,
                    false
                )
                model_SBML.parameters[parameter_id] = parameter
            end
        end
    end
    return nothing
end

function _reorder_speciemap(speciemap, odesystem::ODESystem)
    statenames = unknowns(odesystem) .|> string
    speciemap_out = similar(speciemap, length(statenames))
    for (i, statename) in pairs(statenames)
        imap = findfirst(x -> x == statename, first.(speciemap) .|> string)
        speciemap_out[i] = speciemap[imap]
    end
    return speciemap_out
end

function _get_odesys(
        model_SBML::SBMLImporter.ModelSBML, paths::Dict{Symbol, String}, exist::Bool,
        build_julia_files::Bool, write_to_file::Bool, verbose::Bool
    )
    _logging(:Build_SBML, verbose; buildfiles = build_julia_files, exist = exist)

    btime = @elapsed begin
        if !exist || build_julia_files == true
            model_SBML_sys = SBMLImporter._to_system_syntax(model_SBML, false, false)
            modelstr = SBMLImporter.write_reactionsystem(
                model_SBML_sys, paths[:dirjulia], model_SBML; write_to_file = write_to_file
            )
        else
            modelstr = _get_functions_as_str(pathmodel, 1)[1]
        end
    end
    _logging(:Build_SBML, verbose; time = btime)

    _logging(:Build_ODESystem, verbose)
    btime = @elapsed begin
        get_rn = @RuntimeGeneratedFunction(Meta.parse(modelstr))
        # Argument needed by @RuntimeGeneratedFunction
        rn, speciemap, parametermap = get_rn("https://xkcd.com/303/")
        _odesystem = convert(ODESystem, Catalyst.complete(rn))
        # DAE requires special processing
        if isempty(model_SBML.algebraic_rules)
            odesystem = structural_simplify(_odesystem)
        else
            odesystem = structural_simplify(dae_index_lowering(_odesystem))
        end
    end
    _logging(:Build_ODESystem, verbose; time = btime)

    # The state-map is not in the same order as unknowns(system) so the former is reorded
    # to make it easier to build the u0 function
    speciemap = _reorder_speciemap(speciemap, odesystem)
    return odesystem, speciemap, parametermap
end

function _get_odeproblem(
        model_SBML::SBMLImporter.ModelSBML, ode_ml_models::MLModels,
        petab_tables::PEtabTables, verbose::Bool
    )
    hybridization_df, mappings_df = _get_petab_tables(
        petab_tables, [:hybridization, :mapping]
    )

    _logging(:Build_SBML_prob, verbose)
    btime = @elapsed begin
        for ml_id in ode_ml_models.ml_ids
            output_variables = Iterators.flatten(
                _get_ml_model_io_petab_ids(mappings_df, ml_id, :outputs)
            )
            outputs_df = filter(
                row -> row.targetValue in output_variables, hybridization_df
            )

            output_targets = outputs_df.targetId
            for output_target in output_targets
                delete!(model_SBML.parameters, output_target)
            end
        end

        # Parse the SBML model into an ODEProblem with the neueral networks inserted
        model_SBML_prob = SBMLImporter.ModelSBMLProb(model_SBML)
        __fode = _template_odeproblem(model_SBML_prob, model_SBML, ode_ml_models, petab_tables)
        _fode = @RuntimeGeneratedFunction(Meta.parse(__fode))
        fode = let ml_models = ode_ml_models
            (du, u, p, t) -> _fode(du, u, p, t, ml_models)
        end

        # Build the internal PEtab.jl ODEProblem parameter struct, which is a ComponentVector
        # with mechanistic and neural-network parameters. For internal PEtab.jl mapping,
        # neural-network must be last in this ComponentVector
        ps_mech = parse.(Float64, last.(model_SBML_prob.psmap))
        ps_ode = (; (Symbol.(first.(model_SBML_prob.psmap)) .=> ps_mech)...)
        for ml_model in ode_ml_models.ml_models
            ps_ml = _get_lux_ps(ml_model)
            ps_ode = merge(ps_ode, (; ml_model.ml_id => ps_ml))
        end
        ps_ode = ComponentArray(ps_ode)

        # For internal mapping, initial values need to be provided as a ComponentArray
        _u0tmp = zeros(Float64, length(model_SBML_prob.umodel))
        u0 = ComponentArray(; (Symbol.(model_SBML_prob.umodel) .=> _u0tmp)...)
        oprob = SciMLBase.ODEProblem(fode, u0, (0.0, 10.0), ps_ode)
    end
    _logging(:Build_SBML_prob, verbose; time = btime)

    # PEtab.jl needed maps for later processing
    u0map = model_SBML_prob.umap
    psmap = nothing
    return oprob, u0map, psmap
end

function _reorder_parametermap(parametermap, parameter_order::Vector{Symbol})
    parametermap_out = Vector{Pair{Symbolics.Num, Float64}}(undef, length(parametermap))
    for (i, pname) in pairs(parameter_order)
        imap = findfirst(x -> x == pname, Symbol.(first.(parametermap)))

        if !(parametermap[imap].second isa Symbolics.Num)
            parametermap_out[i] = parametermap[imap].first => parametermap[imap].second
        elseif SBMLImporter.is_number(string(parametermap[imap].second))
            parametermap_out[i] = parametermap[imap].first => parametermap[imap].second.val
        else
            parametermap_out[i] = parametermap[imap].first => 0.0
        end
    end
    return parametermap_out
end

function _get_sbml_speciemap(model_SBML::SBMLImporter.ModelSBML)
    model_SBML_sys = SBMLImporter._to_system_syntax(model_SBML, false, false)
    modelstr = SBMLImporter.write_reactionsystem(
        model_SBML_sys, "", model_SBML; write_to_file = false
    )
    get_rn = @RuntimeGeneratedFunction(Meta.parse(modelstr))
    _, speciemap, _ = get_rn("https://xkcd.com/303/")
    return speciemap
end

function _get_sys_observables(sys::ReactionSystem)::Dict{Symbol, Function}
    return _get_sys_observables(_get_system(sys))
end
function _get_sys_observables(sys::ODESystem)::Dict{Symbol, Function}
    sys_observables = Dict{Symbol, Function}()
    for _observable in observables(sys)
        index_obs = Symbol(replace(string(_observable), "(t)" => ""))
        sys_observables[index_obs] = SymbolicIndexingInterface.observed(sys, index_obs)
    end
    return sys_observables
end
function _get_sys_observables(::ODEProblem)::Dict{Symbol, Function}
    return Dict{Symbol, Function}()
end
