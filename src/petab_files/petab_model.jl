function PEtabModel(path_yaml::String; build_julia_files::Bool = true,
                    verbose::Bool = false, ifelse_to_callback::Bool = true,
                    write_to_file::Bool = false)::PEtabModel
    paths = _get_petab_paths(path_yaml)
    petab_tables = read_tables(path_yaml)
    return _PEtabModel(paths, petab_tables, build_julia_files, verbose, ifelse_to_callback,
                       write_to_file)
end

function _PEtabModel(paths::Dict{Symbol, String}, petab_tables::Dict{Symbol, DataFrame},
                     build_julia_files::Bool, verbose::Bool, ifelse_to_callback::Bool,
                     write_to_file::Bool)
    name = splitdir(paths[:dirmodel])[end]

    write_to_file && !isdir(paths[:dirjulia]) && mkdir(paths[:dirjulia])
    if write_to_file == false && build_julia_files == true
        paths[:dirjulia] = ""
    end
    _logging(:Build_PEtabModel, verbose; name = name)

    # Import SBML model with SBMLImporter
    # In case one of the conditions in the PEtab table assigns an initial specie value,
    # the SBML model must be mutated to add an initial value parameter to correctly
    # compute gradients. However, it is allowed for this parameter to take the value
    # NaN, in this case the importer should resort to the SBML file initial value
    # mapping. Therefore, for building the u0 function the original specie-map must
    # also be extracted
    model_SBML = SBMLImporter.parse_SBML(paths[:SBML], false; model_as_string = false,
                                         ifelse_to_callback = ifelse_to_callback,
                                         inline_assignment_rules = false)
    speciemap_sbml = _get_sbml_speciemap(model_SBML)
    add_u0_parameters!(model_SBML, petab_tables[:conditions], petab_tables[:parameters])
    pathmodel = joinpath(paths[:dirjulia], name * ".jl")
    exist = isfile(pathmodel)
    _logging(:Build_SBML, verbose; buildfiles = build_julia_files, exist = exist)
    if !exist || build_julia_files == true
        btime = @elapsed begin
            model_SBML_sys = SBMLImporter._to_system_syntax(model_SBML, false, false)
            modelstr = SBMLImporter.write_reactionsystem(model_SBML_sys, paths[:dirjulia],
                                                         model_SBML;
                                                         write_to_file = write_to_file)
        end
        _logging(:Build_SBML, verbose; time = btime)
    else
        modelstr = _get_functions_as_str(pathmodel, 1)[1]
    end

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
    # The state-map is not in the same order as unknowns(system) so the former is reordered
    # to make it easier to build the u0 function
    speciemap = _reorder_speciemap(speciemap, odesystem)

    # Indices for mapping parameters and tracking which parameter to estimate, useful
    # when building the coming PEtab functions
    xindices = ParameterIndices(petab_tables, odesystem, parametermap, speciemap)

    _logging(:Build_ODESystem, verbose; time = btime)

    path_u0_h_σ = joinpath(paths[:dirjulia], "$(name)_h_sd_u0.jl")
    exist = isfile(path_u0_h_σ)
    _logging(:Build_u0_h_σ, verbose; buildfiles = build_julia_files, exist = exist)
    if !exist || build_julia_files == true
        btime = @elapsed begin
            hstr, u0!str, u0str, σstr = parse_observables(name, paths, odesystem,
                                                          petab_tables[:observables],
                                                          xindices, speciemap,
                                                          speciemap_sbml, model_SBML,
                                                          write_to_file)
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
        float_tspan = _xdynamic_in_event_cond(model_SBML, xindices, petab_tables) |> !
        psys = _get_sys_parameters(odesystem, speciemap, parametermap) .|> string
        cbset = SBMLImporter.create_callbacks(odesystem, model_SBML, name;
                                              p_PEtab = psys, float_tspan = float_tspan)
    end
    _logging(:Build_callbacks, verbose; time = btime)

    return PEtabModel(name, compute_h, compute_u0!, compute_u0, compute_σ, float_tspan,
                      paths, odesystem, deepcopy(odesystem), parametermap, speciemap,
                      petab_tables, cbset, false)
end

function add_u0_parameters!(model_SBML::SBMLImporter.ModelSBML, conditions_df::DataFrame,
                            parameters_df::DataFrame)::Nothing
    specieids = keys(model_SBML.species)
    rateruleids = model_SBML.rate_rule_variables
    sbml_variables = Iterators.flatten((specieids, rateruleids)) |> unique

    condition_variables = names(conditions_df)
    if any(x -> x in sbml_variables, condition_variables) == false
        return nothing
    end

    for condition_variable in condition_variables
        !(condition_variable in sbml_variables) && continue

        if condition_variable in specieids
            sbml_variable = model_SBML.species[condition_variable]
        else
            sbml_variable = model_SBML.parameters[condition_variable]
        end
        u0name = "__init__" .* sbml_variable.name .* "__"
        value = sbml_variable.initial_value
        u0parameter = SBMLImporter.ParameterSBML(u0name, true, value, "", false, false,
                                                 false, false, false, false)
        model_SBML.parameters[u0name] = u0parameter
        sbml_variable.initial_value = u0name

        # Check if any parameter in the PEtab tables maps to u0 in the conditions table,
        # because if this is the case this parameter must be added to the SBML model as
        # it should be treated as a dynamic parameter for indexing. A NaN value is allowed,
        # meaning the parameter should be set by the corresponding SBML formula
        for condition_value in conditions_df[!, condition_variable]
            condition_value == "NaN" && continue
            condition_value isa Real && continue
            is_number(condition_value) && continue
            ismissing(condition_value) && continue
            condition_value in keys(model_SBML.parameters) && continue
            if !(condition_value in parameters_df[!, :parameterId])
                throw(PEtabFileError("The condition table value $condition_variable does \
                                      not correspond to any parameter in the SBML file \
                                      parameters file"))
            end
            parameter = SBMLImporter.ParameterSBML(condition_value, true, "0.0", "", false,
                                                   false, false, false, false, false)
            model_SBML.parameters[condition_value] = parameter
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
    modelstr = SBMLImporter.write_reactionsystem(model_SBML_sys, "", model_SBML;
                                                 write_to_file = false)
    get_rn = @RuntimeGeneratedFunction(Meta.parse(modelstr))
    _, speciemap, _ = get_rn("https://xkcd.com/303/")
    return speciemap
end
