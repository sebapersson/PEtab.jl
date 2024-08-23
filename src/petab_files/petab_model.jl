function PEtabModel(path_yaml::String; build_julia_files::Bool = false, verbose::Bool = true, ifelse_to_event::Bool = true, custom_values::Union{Nothing, Dict} = nothing, write_to_file::Bool = true)::PEtabModel
    paths = _get_petab_paths(path_yaml)
    conditions_df, measurements_df, parameters_df, observables_df = read_tables(path_yaml)
    modelname = splitdir(paths[:dirmodel])[end]
    dirjulia = joinpath(paths[:dirmodel], "Julia_model_files")
    !isdir(dirjulia) && write_to_file && mkdir(dirjulia)
    _logging(:Build_PEtabModel, verbose; name=modelname)

    # Import SBML model with SBMLImporter
    # In case one of the conditions in the PEtab table assigns an initial specie value,
    # the SBML model must be mutated to add an iniitial value parameter to correctly
    # compute gradients
    model_SBML = SBMLImporter.build_SBML_model(paths[:SBML]; model_as_string = false, ifelse_to_callback = ifelse_to_event, inline_assignment_rules = false)
    _addu0_parameters!(model_SBML, conditions_df, parameters_df)
    pathmodel = joinpath(dirjulia, modelname * ".jl")
    exist = isfile(pathmodel)
    _logging(:Build_SBML, verbose; buildfiles = build_julia_files, exist = exist)
    if !exist || build_julia_files == true
        btime = @elapsed begin
            parsed_model_SBML = SBMLImporter._reactionsystem_from_SBML(model_SBML; check_massaction=false)
            modelstr = SBMLImporter.reactionsystem_to_string(parsed_model_SBML,
                                                              write_to_file,
                                                              pathmodel,
                                                              model_SBML)
        end
        _logging(:Build_SBML, verbose; time = btime)
    else
        modelstr = _get_functions_as_str(pathmodel, 1)[1]
    end

    _logging(:Build_ODESystem, verbose)
    btime = @elapsed begin
        get_rn = @RuntimeGeneratedFunction(Meta.parse(modelstr))
        # Argument needed by @RuntimeGeneratedFunction
        rn, statemap, parametermap = get_rn("https://xkcd.com/303/")
        _odesystem = convert(ODESystem, rn)
        # DAE requires special processing
        if isempty(model_SBML.algebraic_rules)
            odesystem = structural_simplify(_odesystem)
        else
            odesystem = structural_simplify(dae_index_lowering(_odesystem))
        end
        parameter_names = parameters(odesystem)
        state_names = states(odesystem)
    end
    # The state-map is not in the same order as states(system) so the former is reorded
    # to make it easier to build the u0 function
    _reorder_statemap!(statemap, odesystem)
    _logging(:Build_ODESystem, verbose; time = btime)

    path_u0_h_σ = joinpath(dirjulia, modelname * "_u0_h_sd.jl")
    exist = isfile(path_u0_h_σ)
    _logging(:Build_u0_h_σ, verbose; buildfiles = build_julia_files, exist = exist)
    if !exist || build_julia_files == true
        # TODO: Change after refactoring observable file
        btime = @elapsed begin
            h_str, u0!_str, u0_str, σ_str = create_u0_h_σ_file(modelname, path_yaml, dirjulia, odesystem, parametermap, statemap, model_SBML, custom_values = custom_values, write_to_file = write_to_file)
        end
        _logging(:Build_u0_h_σ, verbose; time = btime)
    else
        h_str, u0!_str, u0_str, σ_str = _get_functions_as_str(path_u0_h_σ, 4)
    end
    compute_h = @RuntimeGeneratedFunction(Meta.parse(h_str))
    compute_u0! = @RuntimeGeneratedFunction(Meta.parse(u0!_str))
    compute_u0 = @RuntimeGeneratedFunction(Meta.parse(u0_str))
    compute_σ = @RuntimeGeneratedFunction(Meta.parse(σ_str))

    path_∂_h_σ = joinpath(dirjulia, modelname * "_d_h_sd.jl")
    exist = isfile(path_∂_h_σ)
    _logging(:Build_∂_h_σ, verbose; buildfiles = build_julia_files, exist = exist)
    if !exist || build_julia_files == true
        btime = @elapsed begin
            ∂h∂u_str, ∂h∂p_str, ∂σ∂u_str, ∂σ∂p_str = create_∂_h_σ_file(modelname, path_yaml, dirjulia, odesystem, parametermap, statemap, model_SBML, custom_values = custom_values, write_to_file = write_to_file)
        end
        _logging(:Build_∂_h_σ, verbose; time = btime)
    else
        ∂h∂u_str, ∂h∂p_str, ∂σ∂u_str, ∂σ∂p_str = _get_functions_as_str(path_D_h_sd, 4)
    end
    compute_∂h∂u! = @RuntimeGeneratedFunction(Meta.parse(∂h∂u_str))
    compute_∂h∂p! = @RuntimeGeneratedFunction(Meta.parse(∂h∂p_str))
    compute_∂σ∂σu! = @RuntimeGeneratedFunction(Meta.parse(∂σ∂u_str))
    compute_∂σ∂σp! = @RuntimeGeneratedFunction(Meta.parse(∂σ∂p_str))

    # TODO: Refactor later to only use SBMLImporter functionality here
    _logging(:Build_callbacks, verbose)
    btime = @elapsed begin
        cbset, compute_tstops, check_cb_active, convert_tspan = create_callbacks_SBML(odesystem,
                                                                                      parametermap,
                                                                                      statemap,
                                                                                      model_SBML,
                                                                                      modelname,
                                                                                      path_yaml,
                                                                                      dirjulia,
                                                                                      custom_values = custom_values,
                                                                                      write_to_file = write_to_file)
    end
    _logging(:Build_callbacks, verbose; time = btime)

    petab_model = PEtabModel(modelname,
                             compute_h,
                             compute_u0!,
                             compute_u0,
                             compute_σ,
                             compute_∂h∂u!,
                             compute_∂σ∂σu!,
                             compute_∂h∂p!,
                             compute_∂σ∂σp!,
                             compute_tstops,
                             convert_tspan,
                             odesystem,
                             deepcopy(odesystem),
                             parametermap,
                             statemap,
                             parameter_names,
                             state_names,
                             paths[:dirmodel],
                             dirjulia,
                             measurements_df,
                             conditions_df,
                             observables_df,
                             parameters_df,
                             paths[:SBML],
                             path_yaml,
                             cbset,
                             check_cb_active,
                             false)
    return petab_model
end

function _addu0_parameters!(model_SBML::SBMLImporter.ModelSBML, conditions_df::DataFrame, parameters_df::DataFrame)::Nothing
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
        u0parameter = SBMLImporter.ParameterSBML(u0name, true, value, "", false, false, false)
        model_SBML.parameters[u0name] = u0parameter
        sbml_variable.initial_value = u0name

        # Check if any parameter in the PEtab tables maps to u0 in the conditions table,
        # because if this is the case this parameter must be added to the SBML model as
        # it should be treated as a dynamic parameter for indexing
        for condition_value in conditions_df[!, condition_variable]
            condition_value isa Real && continue
            ismissing(condition_value) && continue
            condition_value in keys(model_SBML.parameters) && continue
            if !(condition_variable in parameters_df[!, :parameterId])
                throw(PEtabFileError("The condition table value $condition_variable does not
                                      correspond to any parameter in the SBML file
                                      parameters file"))
            end
            parameter = SBMLImporter.ParameterSBML(condition_value, true, "0.0", "", false, false, false)
            model_SBML.parameters[condition_value] = parameter
        end
    end
    return nothing
end

function _reorder_statemap!(statemap, odesystem::ODESystem)::Nothing
    statenames = states(odesystem) .|> string
    for (i, statename) in pairs(statenames)
        string(statemap[i].first) == statename && continue
        imap = findfirst(x -> x == statename, first.(statemap) .|> string)
        tmp = statemap[i]
        statemap[i] = statemap[imap]
        statemap[imap] = tmp
    end
    return nothing
end
