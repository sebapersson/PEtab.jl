# TODO: Write check to test a non-dynamic variable appears in observable or noise  formula

"""
    create_u0_h_σ_file(model_name::String,
                       dirmodel::String,
                       system::ODESystem,
                       statemap,
                       model_SBML::SBMLImporter.ModelSBML;
                       verbose::Bool=false)

    For a PeTab model with name model_name with all PeTab-files in dirmodel and associated
    ModellingToolkit ODESystem (with its statemap) build a file containing a functions for
    i) computing the observable model value (h) ii) compute the initial value u0 (by using the
    statemap) and iii) computing the standard error (σ) for each observableFormula in the
    observables PeTab file.

    Note - The produced Julia file will go via the JIT-compiler. The SBML-dict is needed as
    sometimes variables are encoded via explicit-SBML rules.
"""
function create_u0_h_σ_file(model_name::String,
                            path_yaml::String,
                            dirjulia::String,
                            system::ODESystem,
                            parametermap,
                            statemap,
                            model_SBML::SBMLImporter.ModelSBML;
                            custom_values::Union{Nothing, Dict} = nothing,
                            write_to_file::Bool = true)
    p_ode_problem_names = _get_sys_parameters(system, statemap, parametermap) .|> string
    model_state_names = replace.(string.(unknowns(system)), "(t)" => "")

    petab_tables = read_tables(path_yaml)
    measurements_data, observables_data, parameters_data, experimental_conditions, = collect(values(petab_tables))
    parameter_info = parse_parameters(parameters_data, custom_values = custom_values)
    measurement_info = parse_measurements(measurements_data, observables_data)

    # Indices for keeping track of parameters in θ
    θ_indices = parse_conditions(parameter_info, measurement_info, system, parametermap,
                                 statemap, experimental_conditions)

    h_str = create_h_function(model_name, dirjulia, model_state_names, parameter_info,
                              p_ode_problem_names,
                              string.(θ_indices.xids[:nondynamic]), observables_data,
                              model_SBML, write_to_file)

    u0!_str = create_u0_function(model_name, dirjulia, parameter_info, p_ode_problem_names,
                                 statemap, write_to_file, model_SBML, inplace = true)

    u0_str = create_u0_function(model_name, dirjulia, parameter_info, p_ode_problem_names,
                                statemap, write_to_file, model_SBML, inplace = false)

    σ_str = create_σ_function(model_name, dirjulia, parameter_info, model_state_names,
                              p_ode_problem_names, string.(θ_indices.xids[:nondynamic]),
                              observables_data, model_SBML, write_to_file)

    return h_str, u0!_str, u0_str, σ_str
end
"""
    When parsed from Julia input.
"""
function create_u0_h_σ_file(model_name::String,
                            system,
                            experimental_conditions::DataFrame,
                            measurements_data::DataFrame,
                            parameters_data::DataFrame,
                            observables_data::DataFrame,
                            statemap)::NTuple{4, String}
    model_state_names = replace.(string.(unknowns(system)), "(t)" => "")
    parametermap = [p => 0.0 for p in parameters(system)]
    p_ode_problem_names = _get_sys_parameters(system, statemap, parametermap) .|> string

    parameter_info = PEtab.parse_parameters(parameters_data)
    measurement_info = PEtab.parse_measurements(measurements_data, observables_data)

    # Indices for keeping track of parameters in θ
    θ_indices = PEtab.parse_conditions(parameter_info, measurement_info, system,
                                       parametermap, statemap, experimental_conditions)

    # Dummary variables to keep PEtab importer happy even as we are not providing any PEtab files
    model_SBML = SBMLImporter.ModelSBML("")

    h_str = PEtab.create_h_function(model_name, @__DIR__, model_state_names, parameter_info,
                                    p_ode_problem_names,
                                    string.(θ_indices.xids[:nondynamic]),
                                    observables_data, model_SBML, false)
    u0!_str = PEtab.create_u0_function(model_name, @__DIR__, parameter_info,
                                       p_ode_problem_names, statemap, false,
                                       model_SBML, inplace = true)
    u0_str = PEtab.create_u0_function(model_name, @__DIR__, parameter_info,
                                      p_ode_problem_names, statemap, false,
                                      model_SBML, inplace = false)
    σ_str = PEtab.create_σ_function(model_name, @__DIR__, parameter_info, model_state_names,
                                    p_ode_problem_names,
                                    string.(θ_indices.xids[:nondynamic]),
                                    observables_data, model_SBML, false)

    return h_str, u0!_str, u0_str, σ_str
end

"""
    create_h_function(model_name::String,
                      dirmodel::String,
                      model_state_names::Vector{String},
                      parameter_info::ParametersInfo,
                      namesParamDyn::Vector{String},
                      namesNonDynParam::Vector{String},
                      observables_data::DataFrame,
                      model_SBML::SBMLImporter.ModelSBML)

    For model_name create a function for computing y_model by translating the observables_data
    PeTab-file into Julia syntax.
"""
function create_h_function(model_name::String,
                           dirmodel::String,
                           model_state_names::Vector{String},
                           parameter_info::ParametersInfo,
                           p_ode_problem_names::Vector{String},
                           xnondynamic_names::Vector{String},
                           observables_data::DataFrame,
                           model_SBML::SBMLImporter.ModelSBML,
                           write_to_file::Bool)
    io = IOBuffer()
    path_save = joinpath(dirmodel, model_name * "_h_sd_u0.jl")
    model_state_str, xdynamic_str, xnondynamic_str, constant_parameters_str = create_top_function_h(model_state_names,
                                                                                                       parameter_info,
                                                                                                       p_ode_problem_names,
                                                                                                       xnondynamic_names)

    # Write the formula for each observable in Julia syntax
    observable_ids = string.(observables_data[!, :observableId])
    observable_str = ""
    for i in eachindex(observable_ids)
        _formula = filter(x -> !isspace(x),
                          string(observables_data[!, :observableFormula][i]))
        observable_parameters = get_observable_parameters(_formula)
        observable_str *= "\tif observableId === " * ":" * observable_ids[i] * " \n"
        if !isempty(observable_parameters)
            observable_str *= "\t\t" * observable_parameters *
                              " = get_obs_sd_parameter(xobservable, parametermap)\n"
        end

        formula = replace_explicit_variable_rule(_formula, model_SBML)

        # Translate the formula for the observable to Julia syntax
        _julia_formula = petab_formula_to_Julia(formula, model_state_names, parameter_info,
                                                p_ode_problem_names, xnondynamic_names)
        julia_formula = variables_to_array_index(_julia_formula, model_state_names,
                                                 parameter_info, p_ode_problem_names,
                                                 xnondynamic_names, p_ode_problem = true)
        observable_str *= "\t\t" * "return " * julia_formula * "\n" * "\tend\n\n"
    end

    # Create h function
    if write_to_file == true
        write(io, model_state_str)
        write(io, xdynamic_str)
        write(io, xnondynamic_str)
        write(io, constant_parameters_str)
        write(io, "\n")
    end
    write(io,
          "function compute_h(u::AbstractVector, t::Real, p_ode_problem::AbstractVector, xobservable::AbstractVector,
               xnondynamic::AbstractVector, nominal_values::Vector{Float64}, observableId::Symbol,
                  parametermap::ObservableNoiseMap)::Real \n")
    write(io, observable_str)
    write(io, "end")
    h_str = String(take!(io))
    if write_to_file == true
        str_write = h_str * "\n\n"
        open(path_save, "w") do f
            write(f, str_write)
        end
    end
    close(io)
    return h_str
end

"""
    create_top_function_h(model_state_names::Vector{String},
                          parameter_info::ParametersInfo,
                          namesParamODEProb::Vector{String},
                          namesNonDynParam::Vector{String})

    Extracts all variables needed for the observable h function.
"""
function create_top_function_h(model_state_names::Vector{String},
                               parameter_info::ParametersInfo,
                               p_ode_problem_names::Vector{String},
                               xnondynamic_names::Vector{String})
    model_state_str = "#"
    for i in eachindex(model_state_names)
        model_state_str *= "u[" * string(i) * "] = " * model_state_names[i] * ", "
    end
    model_state_str = model_state_str[1:(end - 2)] # Remove last non needed ", "
    model_state_str *= "\n"

    xdynamic_str = "#"
    for i in eachindex(p_ode_problem_names)
        xdynamic_str *= "p_ode_problem_names[" * string(i) * "] = " *
                         p_ode_problem_names[i] * ", "
    end
    xdynamic_str = xdynamic_str[1:(end - 2)] # Remove last non needed ", "
    xdynamic_str *= "\n"

    xnondynamic_str = "#"
    if !isempty(xnondynamic_names)
        for i in eachindex(xnondynamic_names)
            xnondynamic_str *= "xnondynamic[" * string(i) * "] = " *
                                 xnondynamic_names[i] * ", "
        end
        xnondynamic_str = xnondynamic_str[1:(end - 2)] # Remove last non needed ", "
        xnondynamic_str *= "\n"
    end

    constant_parameters_str = ""
    for i in eachindex(parameter_info.parameter_id)
        if parameter_info.estimate[i] == false
            constant_parameters_str *= "#parameter_info.nominalValue[" * string(i) *
                                       "] = " * string(parameter_info.parameter_id[i]) *
                                       "_C \n"
        end
    end
    constant_parameters_str *= "\n"

    return model_state_str, xdynamic_str, xnondynamic_str, constant_parameters_str
end

"""
    create_u0_function(model_name::String,
                       dirmodel::String,
                       parameter_info::ParametersInfo,
                       p_ode_problem_names::Vector{String},
                       statemap,
                       model_SBML;
                       inplace::Bool=true)

    For model_name create a function for computing initial value by translating the statemap
    into Julia syntax.

    To correctly create the function the name of all parameters, parameter_info (to get constant parameters)
    are required.
"""
function create_u0_function(model_name::String,
                            dirmodel::String,
                            parameter_info::ParametersInfo,
                            p_ode_problem_names::Vector{String},
                            statemap,
                            write_to_file::Bool,
                            model_SBML::SBMLImporter.ModelSBML;
                            inplace::Bool = true)
    path_save = joinpath(dirmodel, model_name * "_h_sd_u0.jl")
    io = IOBuffer()

    if inplace == true
        write(io,
              "function u0!(u0::AbstractVector, p_ode_problem::AbstractVector) \n\n")
    else
        write(io, "function u0(p_ode_problem::AbstractVector)::AbstractVector \n\n")
    end

    # Write named list of parameter to file
    p_ode_problem_str = "\t#"
    for i in eachindex(p_ode_problem_names)
        p_ode_problem_str *= "p_ode_problem[" * string(i) * "] = " *
                             p_ode_problem_names[i] * ", "
    end
    p_ode_problem_str = p_ode_problem_str[1:(end - 2)]
    p_ode_problem_str *= "\n\n"
    write(io, p_ode_problem_str)

    write(io, "\tt = 0.0 # u at time zero\n\n")

    # Write the formula for each initial condition to file
    _model_state_names = [replace.(string.(statemap[i].first), "(t)" => "")
                          for i in eachindex(statemap)]
    # If we create from Julia model (e.g.) Catalyst this is not applicable and step is skipped
    if !isempty(model_SBML.parameters)
        model_state_names1 = filter(x -> x ∈ keys(model_SBML.species) &&
                                        model_SBML.species[x].assignment_rule == false,
                                    _model_state_names)
        model_state_names2 = filter(x -> x ∈ keys(model_SBML.parameters) &&
                                        model_SBML.parameters[x].rate_rule == true,
                                    _model_state_names)
        model_state_names = vcat(model_state_names1, model_state_names2)
    else
        model_state_names = _model_state_names
    end
    model_state_str = ""
    for i in eachindex(model_state_names)
        stateName = model_state_names[i]
        _stateExpression = replace(string(statemap[i].second), " " => "")
        stateFormula = petab_formula_to_Julia(_stateExpression, model_state_names,
                                              parameter_info, p_ode_problem_names, String[])
        for i in eachindex(p_ode_problem_names)
            stateFormula = SBMLImporter._replace_variable(stateFormula,
                                                         p_ode_problem_names[i],
                                                         "p_ode_problem[" * string(i) * "]")
        end
        model_state_str *= "\t" * stateName * " = " * stateFormula * "\n"
    end
    write(io, model_state_str * "\n")

    # Ensure the states in correct order are written to u0
    if inplace == true
        model_state_str = "\tu0 .= "
        for i in eachindex(model_state_names)
            model_state_str *= model_state_names[i] * ", "
        end
        model_state_str = model_state_str[1:(end - 2)]
        write(io, model_state_str)

        # Where we return the entire initial value vector
    elseif inplace == false
        model_state_str = "\t return ["
        for i in eachindex(model_state_names)
            model_state_str *= model_state_names[i] * ", "
        end
        model_state_str = model_state_str[1:(end - 2)]
        model_state_str *= "]"
        write(io, model_state_str)
    end

    write(io, "\nend")
    u0_str = String(take!(io))
    if write_to_file == true
        str_write = u0_str * "\n\n"
        open(path_save, "a") do f
            write(f, str_write)
        end
    end
    close(io)

    return u0_str
end

"""
    create_σ_function(model_name::String,
                      dirmodel::String,
                      parameter_info::ParametersInfo,
                      model_state_names::Vector{String},
                      p_ode_problem_names::Vector{String},
                      xnondynamic_names::Vector{String},
                      observables_data::DataFrame,
                      model_SBML::SBMLImporter.ModelSBML)

    For model_name create a function for computing the standard deviation σ by translating the observables_data
    PeTab-file into Julia syntax.
"""
function create_σ_function(model_name::String,
                           dirmodel::String,
                           parameter_info::ParametersInfo,
                           model_state_names::Vector{String},
                           p_ode_problem_names::Vector{String},
                           xnondynamic_names::Vector{String},
                           observables_data::DataFrame,
                           model_SBML::SBMLImporter.ModelSBML,
                           write_to_file::Bool)
    path_save = joinpath(dirmodel, model_name * "_h_sd_u0.jl")
    io = IOBuffer()

    # Write the formula for standard deviations to file
    observable_ids = string.(observables_data[!, :observableId])
    observable_str = ""
    for i in eachindex(observable_ids)
        _formula = filter(x -> !isspace(x), string(observables_data[!, :noiseFormula][i]))
        noise_parameters = get_noise_parameters(_formula)
        observable_str *= "\tif observableId === " * ":" * observable_ids[i] * " \n"
        if !isempty(noise_parameters)
            observable_str *= "\t\t" * noise_parameters *
                              " = get_obs_sd_parameter(xnoise, parametermap)\n"
        end

        formula = replace_explicit_variable_rule(_formula, model_SBML)

        # Translate the formula for the observable to Julia syntax
        _julia_formula = petab_formula_to_Julia(formula, model_state_names, parameter_info,
                                                p_ode_problem_names, xnondynamic_names)
        julia_formula = variables_to_array_index(_julia_formula, model_state_names,
                                                 parameter_info, p_ode_problem_names,
                                                 xnondynamic_names, p_ode_problem = true)
        observable_str *= "\t\t" * "return " * julia_formula * "\n" * "\tend\n\n"
    end

    write(io,
          "function compute_σ(u::AbstractVector, t::Real, xnoise::AbstractVector, p_ode_problem::AbstractVector,  xnondynamic::AbstractVector,
               nominal_values::Vector{Float64}, observableId::Symbol, parametermap::ObservableNoiseMap)::Real \n")
    write(io, observable_str)
    write(io, "\nend")
    σ_str = String(take!(io))
    if write_to_file == true
        str_write = σ_str * "\n\n"
        open(path_save, "a") do f
            write(f, str_write)
        end
    end
    close(io)

    return σ_str
end
