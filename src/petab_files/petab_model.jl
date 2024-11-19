function PEtabModel(path_yaml::String; build_julia_files::Bool = true,
                    verbose::Bool = false, ifelse_to_callback::Bool = true,
                    write_to_file::Bool = false,
                    nnmodels::Union{Dict, Nothing} = nothing)::PEtabModel
    paths = _get_petab_paths(path_yaml)
    petab_tables = read_tables(path_yaml)
    name = splitdir(paths[:dirmodel])[end]

    write_to_file && !isdir(paths[:dirjulia]) && mkdir(paths[:dirjulia])
    if write_to_file == false && build_julia_files == true
        paths[:dirjulia] = ""
    end
    _logging(:Build_PEtabModel, verbose; name = name)

    # Build the internal PEtab.jl nn-dict. The same that users provide via the PEtab
    # interface
    rng = Random.default_rng()
    nn = Dict()
    for (netid, nnmodel) in nnmodels
        _, _st = Lux.setup(rng, nnmodel[:net])
        nn[netid] = [_st, nnmodel[:net]]
    end

    # Import SBML model with SBMLImporter
    # In case one of the conditions in the PEtab table assigns an initial specie value,
    # the SBML model must be mutated to add an iniitial value parameter to correctly
    # compute gradients
    nnmodels_in_ode = _get_in_ode_nets(nnmodels)
    if !isempty(nnmodels_in_ode)
        inline_assignment_rules = true
    else
        inline_assignment_rules = false
    end
    model_SBML = SBMLImporter.parse_SBML(paths[:SBML], false; model_as_string = false,
                                         ifelse_to_callback = ifelse_to_callback,
                                         inline_assignment_rules = inline_assignment_rules)
    _addu0_parameters!(model_SBML, petab_tables[:conditions], petab_tables[:parameters])
    pathmodel = joinpath(paths[:dirjulia], name * ".jl")
    exist = isfile(pathmodel)
    # If the model contains a neural-network it must be parsed as an ODEProblem, as MTK
    # does not have good neural-net support yet. Otherwise, model should be parsed as
    # ODESystem as usual
    if isempty(nnmodels_in_ode)
        odesystem, speciemap, parametermap = _get_odesys(model_SBML, paths, exist, build_julia_files, write_to_file, verbose)
    else
        odesystem, speciemap, parametermap = _get_odeproblem(model_SBML, nnmodels_in_ode, petab_tables[:mapping_table], nn)
    end

    # Indices for mapping parameters and tracking which parameter to estimate, useful
    # when building the comig PEtab functions
    xindices = ParameterIndices(petab_tables, odesystem, parametermap, speciemap, nn)

    path_u0_h_σ = joinpath(paths[:dirjulia], "$(name)_h_sd_u0.jl")
    exist = isfile(path_u0_h_σ)
    _logging(:Build_u0_h_σ, verbose; buildfiles = build_julia_files, exist = exist)
    if !exist || build_julia_files == true
        btime = @elapsed begin
            hstr, u0!str, u0str, σstr = parse_observables(name, paths, odesystem,
                                                          petab_tables[:observables],
                                                          xindices, speciemap, model_SBML,
                                                          petab_tables[:mapping_table],
                                                          write_to_file)
        end
        _logging(:Build_u0_h_σ, verbose; time = btime)
    else
        hstr, u0!str, u0str, σstr = _get_functions_as_str(path_u0_h_σ, 4)
    end
    compute_h = @RuntimeGeneratedFunction(Meta.parse(hstr))
    compute_u0! = @RuntimeGeneratedFunction(Meta.parse(u0!str))
    compute_u0 = @RuntimeGeneratedFunction(Meta.parse(u0str))
    compute_σ = @RuntimeGeneratedFunction(Meta.parse(σstr))

    # SBMLImporter holds the callback building functionality. However, currently it needs
    # to know the ODESystem parameter order (psys), and whether or not any parameters
    # which are estimated are present in the event condition. For the latter, timespan
    # should not be converted to floats in case dual numbers (for gradients) are propegated
    _logging(:Build_callbacks, verbose)
    btime = @elapsed begin
        float_tspan = _xdynamic_in_event_cond(model_SBML, xindices, petab_tables) |> !
        psys = _get_xids_sys_order(odesystem, speciemap, parametermap) .|> string
        specie_ids = _get_state_ids(odesystem)
        cbset = SBMLImporter.create_callbacks(odesystem, model_SBML, name;
                                              p_PEtab = psys, float_tspan = float_tspan,
                                              _specie_ids = specie_ids)
    end
    _logging(:Build_callbacks, verbose; time = btime)

    return PEtabModel(name, compute_h, compute_u0!, compute_u0, compute_σ, float_tspan,
                      paths, odesystem, deepcopy(odesystem), parametermap, speciemap,
                      petab_tables, cbset, false, nn)
end

function _addu0_parameters!(model_SBML::SBMLImporter.ModelSBML, conditions_df::DataFrame,
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
        # it should be treated as a dynamic parameter for indexing
        for condition_value in conditions_df[!, condition_variable]
            condition_value isa Real && continue
            ismissing(condition_value) && continue
            condition_value in keys(model_SBML.parameters) && continue
            if !(condition_value in parameters_df[!, :parameterId])
                throw(PEtabFileError("The condition table value $condition_variable does " *
                                     "not correspond to any parameter in the SBML file " *
                                     "parameters file"))
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

function _get_odesys(model_SBML::SBMLImporter.ModelSBML, paths::Dict{Symbol, String}, exist::Bool, build_julia_files::Bool, write_to_file::Bool, verbose::Bool)
    _logging(:Build_SBML, verbose; buildfiles = build_julia_files, exist = exist)
    btime = @elapsed begin
        if !exist || build_julia_files == true
            model_SBML_sys = SBMLImporter._to_system_syntax(model_SBML, false, false)
            modelstr = SBMLImporter.write_reactionsystem(model_SBML_sys, paths[:dirjulia], model_SBML; write_to_file = write_to_file)
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
    # The state-map is not in the same order as unknowns(system) so the former is reorded
    # to make it easier to build the u0 function
    speciemap = _reorder_speciemap(speciemap, odesystem)
    return odesystem, speciemap, parametermap
end

function _get_odeproblem(model_SBML::SBMLImporter.ModelSBML, nnmodels_in_ode::Dict, mapping_table::DataFrame, nn::Dict)
    for id in keys(nnmodels_in_ode)
        if !(string(id) in mapping_table[!, :netId])
            throw(PEtab.PEtabInputError("Neural net $id defined in the net.yaml file is \
                                         is not defined in the mapping table, which it must \
                                         in order to understand how the net interacts with \
                                        the model."))
        end
        # Inputs can be any variable
        for variable_id in PEtab._get_net_values(mapping_table, id, :inputs)
            if SBMLImporter._is_model_variable(variable_id, model_SBML) == false
                throw(PEtab.PEtabInputError("Input $variable_id to neural net $id inside the
                                            ODE is not as required a valid SBML variable."))
            end
        end
        # Outputs can only be parameters, as implications of setting a specie are unclear
        for variable_id in PEtab._get_net_values(mapping_table, id, :outputs)
            if haskey(model_SBML.parameters, variable_id)
                delete!(model_SBML.parameters, variable_id)
                continue
            end
            if occursin(r"_dot$", variable_id)
                specie_id = replace(variable_id, r"_dot$" => "")
                @assert haskey(model_SBML.species, specie_id) "$(specie_id) is not a valid \
                                                               model SBML specie"
                model_SBML.species[specie_id].formula = variable_id
                continue
            end
            throw(PEtab.PEtabInputError("Ouput $variable_id to neural net $id inside the
                                        ODE is not as required a valid SBML parameter or
                                        a specie derivative on the form specie_id_dot"))
        end
    end

    # Build the internal PEtab.jl nn-dict used by the ODEProblem
    rng = Random.default_rng()
    pnns = Dict()
    for (netid, nnmodel) in nnmodels_in_ode
        _pnn, _ = Lux.setup(rng, nnmodel[:net])
        pnns[Symbol("p_$netid")] = _pnn
    end

    # Parse ODEProblem, with nndict via closure
    model_SBML_prob = SBMLImporter.ModelSBMLProb(model_SBML)
    __fode = _template_odeproblem(model_SBML_prob, model_SBML, nnmodels_in_ode, mapping_table)
    _fode = @RuntimeGeneratedFunction(Meta.parse(__fode))
    fode = let nndict = nn
        (du, u, p, t) -> _fode(du, u, p, t, nndict)
    end

    # Parametermap needs to include neural net parameters
    psvals = parse.(Float64, last.(model_SBML_prob.psmap))
    psvec = (; (Symbol.(first.(model_SBML_prob.psmap)) .=> psvals)...)
    for (pid, pnn) in pnns
        psvec = merge(psvec, (;pid => pnn, ))
    end
    psvec = ComponentArray(psvec)
    _u0tmp = zeros(Float64, length(model_SBML_prob.umodel))
    u0 = ComponentArray(; (Symbol.(model_SBML_prob.umodel) .=> _u0tmp)...)
    oprob = SciMLBase.ODEProblem(fode, u0, (0.0, 10.0), psvec)

    # Speciemap (initial values)
    u0map = model_SBML_prob.umap
    psmap = nothing

    # Remove nn from mapping table, to later help distinguish from where a neural net
    # is provided
    for netid in keys(nnmodels_in_ode)
        filter!(:netId => !=(string(netid)), mapping_table)
    end
    return oprob, u0map, psmap
end

function load_nets(path_yaml::String)::Dict
    yaml_file = YAML.load_file(path_yaml)
    sciml_info = yaml_file["extensions"]["petab_sciml"]
    nnmodels = Dict()
    for _netfile in sciml_info["net_files"]
        nnmodel = Dict()
        netfile = joinpath(dirname(path_yaml), _netfile)
        net, id = PEtab.parse_to_lux(netfile)
        for (id, nninfo) in sciml_info["hybridization"][id]
            if id == "input"
                nnmodel[:input] = nninfo
            elseif id == "output"
                nnmodel[:output] = nninfo
            end
        end
        nnmodel[:net] = net
        nnmodels[Symbol(id)] = nnmodel
    end
    return nnmodels
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

function _get_in_ode_nets(nnmodels::Union{Dict, Nothing})::Dict
    out = Dict()
    isnothing(nnmodels) && return out
    for (id, nnmodel) in nnmodels
        if nnmodel[:input] == "ode" && nnmodel[:output] == "ode"
            out[id] = nnmodel
        end
    end
    return out
end
