
"""
    createFileDYmodSdU0(model_name::String,
                       dir_model::String,
                       odeSys::ODESystem,
                       state_map,
                       model_SBML::SBMLImporter.ModelSBML)

    For a PeTab model with name model_name with all PeTab-files in dir_model and associated
    ModellingToolkit ODESystem (with its state_map) build a file containing a functions for
    i) computing the observable model value (y_model) ii) compute the initial value u0 (by using the
    state_map) and iii) computing the standard error (sd) for each observableFormula in the
    observables PeTab file.
    Note - The produced Julia file will go via the JIT-compiler.
"""
function create_derivative_σ_h_file(model_name::String,
                                    path_yaml::String,
                                    dir_julia::String,
                                    system::ODESystem,
                                    parameter_map,
                                    state_map,
                                    model_SBML::SBMLImporter.ModelSBML;
                                    custom_parameter_values::Union{Nothing, Dict} = nothing,
                                    write_to_file::Bool = true)
    p_ode_problem_names = string.(parameters(system))
    model_state_names = replace.(string.(states(system)), "(t)" => "")

    experimental_conditions, measurements_data, parameters_data, observables_data = read_petab_files(path_yaml)
    parameter_info = process_parameters(parameters_data,
                                        custom_parameter_values = custom_parameter_values)
    measurement_info = process_measurements(measurements_data, observables_data)

    # Indices for keeping track of parameters in θ
    θ_indices = compute_θ_indices(parameter_info, measurement_info, system, parameter_map,
                                  state_map, experimental_conditions)

    ∂h∂u_str, ∂h∂p_str = create∂h∂_function(model_name, dir_julia, model_state_names,
                                            parameter_info, p_ode_problem_names,
                                            string.(θ_indices.θ_non_dynamic_names),
                                            observables_data, model_SBML, write_to_file)
    ∂σ∂u_str, ∂σ∂p_str = create∂σ∂_function(model_name, dir_julia, parameter_info,
                                            model_state_names, p_ode_problem_names,
                                            string.(θ_indices.θ_non_dynamic_names),
                                            observables_data, model_SBML, write_to_file)

    return ∂h∂u_str, ∂h∂p_str, ∂σ∂u_str, ∂σ∂p_str
end
function create_derivative_σ_h_file(model_name::String,
                                    system,
                                    experimental_conditions::CSV.File,
                                    measurements_data::CSV.File,
                                    parameters_data::CSV.File,
                                    observables_data::CSV.File,
                                    state_map)
    p_ode_problem_names = string.(parameters(system))
    model_state_names = replace.(string.(states(system)), "(t)" => "")
    parameter_map = [p => 0.0 for p in parameters(system)]

    parameter_info = PEtab.process_parameters(parameters_data)
    measurement_info = PEtab.process_measurements(measurements_data, observables_data)

    # Indices for keeping track of parameters in θ
    θ_indices = PEtab.compute_θ_indices(parameter_info, measurement_info, system,
                                        parameter_map, state_map, experimental_conditions)

    # Dummary variables to keep PEtab importer happy even as we are not providing any PEtab files
    model_SBML = SBMLImporter.ModelSBML("")

    ∂h∂u_str, ∂h∂p_str = PEtab.create∂h∂_function(model_name, @__DIR__, model_state_names,
                                                  parameter_info, p_ode_problem_names,
                                                  string.(θ_indices.θ_non_dynamic_names),
                                                  observables_data, model_SBML, false)
    ∂σ∂u_str, ∂σ∂p_str = PEtab.create∂σ∂_function(model_name, @__DIR__, parameter_info,
                                                  model_state_names, p_ode_problem_names,
                                                  string.(θ_indices.θ_non_dynamic_names),
                                                  observables_data, model_SBML, false)

    return ∂h∂u_str, ∂h∂p_str, ∂σ∂u_str, ∂σ∂p_str
end

"""
    create∂h∂_function(model_name::String,
                       dir_model::String,
                       model_state_names::Vector{String},
                       parameter_info::ParametersInfo,
                       p_ode_problem_names::Vector{String},
                       θ_non_dynamic_names::Vector{String},
                       observables_data::CSV.File,
                       model_SBML::SBMLImporter.ModelSBML)

    For model_name create using Symbolics function for computing ∂h/∂u and ∂h/∂p where
    u = modelStates and p = p_ode_problem (parameters for ODE problem)
"""
function create∂h∂_function(model_name::String,
                            dir_model::String,
                            model_state_names::Vector{String},
                            parameter_info::ParametersInfo,
                            p_ode_problem_names::Vector{String},
                            θ_non_dynamic_names::Vector{String},
                            observables_data::CSV.File,
                            model_SBML::SBMLImporter.ModelSBML,
                            write_to_file::Bool)
    path_save = joinpath(dir_model, model_name * "_D_h_sd.jl")
    io1 = IOBuffer()
    io2 = IOBuffer()

    model_state_str, p_ode_problem_str, θ_non_dynamic_str = create_top_∂h∂_function(model_state_names,
                                                                                    p_ode_problem_names,
                                                                                    θ_non_dynamic_names,
                                                                                    observables_data)

    # Store the formula of each observable in string
    observable_ids = string.(observables_data[:observableId])
    p_observeble_str = ""
    u_observeble_str = ""
    for i in eachindex(observable_ids)

        # Each observebleID falls below its own if-statement
        p_observeble_str *= "\tif observableId == " * ":" * observable_ids[i] * "" * " \n"
        u_observeble_str *= "\tif observableId == " * ":" * observable_ids[i] * "" * " \n"

        _formula = filter(x -> !isspace(x), string(observables_data[:observableFormula][i]))
        formula = replace_explicit_variable_rule(_formula, model_SBML)
        julia_formula = petab_formula_to_Julia(formula, model_state_names, parameter_info,
                                               p_ode_problem_names, θ_non_dynamic_names)

        enter_observable = true # Only extract observable parameter once
        for iState in eachindex(model_state_names)
            if occursin(Regex("\\b" * model_state_names[iState] * "\\b"), julia_formula)
                # Extract observable parameters
                observable_parameters = get_observable_parameters(formula)
                if !isempty(observable_parameters) && enter_observable == true
                    u_observeble_str *= "\t\t" * observable_parameters *
                                        " = get_obs_sd_parameter(θ_observable, parameter_map)\n"
                    enter_observable = false
                end

                julia_formula_symbolic = eval(Meta.parse(julia_formula))
                ui_symbolic = eval(Meta.parse(model_state_names[iState]))
                _∂h∂ui = string(Symbolics.derivative(julia_formula_symbolic, ui_symbolic;
                                                     simplify = true))
                ∂h∂ui = variables_to_array_index(_∂h∂ui, model_state_names, parameter_info,
                                                 p_ode_problem_names, θ_non_dynamic_names,
                                                 p_ode_problem = true)

                u_observeble_str *= "\t\tout[" * string(iState) * "] = " * ∂h∂ui * "\n"
            end
        end

        enter_observable = true # Only extract observable parameter once
        for ip in eachindex(p_ode_problem_names)
            if occursin(Regex("\\b" * p_ode_problem_names[ip] * "\\b"), julia_formula)
                # Extract observable parameters
                observable_parameters = get_observable_parameters(formula)
                if !isempty(observable_parameters) && enter_observable == true
                    p_observeble_str *= "\t\t" * observable_parameters *
                                        " = get_obs_sd_parameter(θ_observable, parameter_map)\n"
                    enter_observable = false
                end

                julia_formula_symbolic = eval(Meta.parse(julia_formula))
                pi_symbolic = eval(Meta.parse(p_ode_problem_names[ip]))
                _∂h∂pi = string(Symbolics.derivative(julia_formula_symbolic, pi_symbolic;
                                                     simplify = true))
                ∂h∂pi = variables_to_array_index(_∂h∂pi, model_state_names, parameter_info,
                                                 p_ode_problem_names, θ_non_dynamic_names,
                                                 p_ode_problem = true)
                p_observeble_str *= "\t\tout[" * string(ip) * "] = " * ∂h∂pi * "\n"
            end
        end

        u_observeble_str *= "\t\t" * "return nothing\n" * "\tend\n\n"
        p_observeble_str *= "\t\t" * "return nothing\n" * "\tend\n\n"
    end

    if write_to_file == true
        write(io1, model_state_str)
        write(io1, p_ode_problem_str)
        write(io1, θ_non_dynamic_str)
        write(io1, "\n")
    end
    write(io1,
          "function compute_∂h∂u!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                  θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) \n")
    write(io1, u_observeble_str)
    write(io1, "end")
    ∂h∂u_str = String(take!(io1))
    if write_to_file
        str_write = ∂h∂u_str * "\n\n"
        open(path_save, "w") do f
            write(f, str_write)
        end
    end

    write(io2,
          "function compute_∂h∂p!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                  θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) \n")
    write(io2, p_observeble_str)
    write(io2, "end")
    ∂h∂p_str = String(take!(io2))
    if write_to_file
        str_write = ∂h∂p_str * "\n\n"
        open(path_save, "a") do f
            write(f, str_write)
        end
    end
    close(io1)
    close(io2)

    return ∂h∂u_str, ∂h∂p_str
end

"""
    create_top_∂h∂_function(model_state_names::Vector{String},
                            p_ode_problem_names::Vector{String},
                            θ_non_dynamic_names::Vector{String},
                            observables_data::CSV.File)

    Extracts all variables needed for the functions and add them as variables for Symbolics.
"""
function create_top_∂h∂_function(model_state_names::Vector{String},
                                 p_ode_problem_names::Vector{String},
                                 θ_non_dynamic_names::Vector{String},
                                 observables_data::CSV.File)

    # We formulate the string in a format accepatble for symbolics so that we later
    # can differentative the observable function with respect to the h formula
    variables_str = "@variables "

    model_state_str = "#"
    for i in eachindex(model_state_names)
        model_state_str *= "u[" * string(i) * "] = " * model_state_names[i] * ", "
        variables_str *= model_state_names[i] * ", "
    end
    model_state_str = model_state_str[1:(end - 2)] # Remove last non needed ", "
    model_state_str *= "\n"

    # Extract name of dynamic parameter
    p_ode_problem_str = "#"
    for i in eachindex(p_ode_problem_names)
        p_ode_problem_str *= "p_ode_problem[" * string(i) * "] = " *
                             p_ode_problem_names[i] * ", "
        variables_str *= p_ode_problem_names[i] * ", "
    end
    p_ode_problem_str = p_ode_problem_str[1:(end - 2)]
    p_ode_problem_str *= "\n"

    θ_non_dynamic_str = "#"
    if !isempty(θ_non_dynamic_names)
        for i in eachindex(θ_non_dynamic_names)
            θ_non_dynamic_str *= "θ_non_dynamic[" * string(i) * "] = " *
                                 θ_non_dynamic_names[i] * ", "
            variables_str *= θ_non_dynamic_names[i] * ", "
        end
        θ_non_dynamic_str = θ_non_dynamic_str[1:(end - 2)] # Remove last non needed ", "
        θ_non_dynamic_str *= "\n"
    end

    # Extracts all observable- and noise-parameters to add them to the symbolics variable string
    observable_ids = string.(observables_data[:observableId])
    for i in eachindex(observable_ids)

        # Extract observable parameters
        _formula = filter(x -> !isspace(x), string(observables_data[:observableFormula][i]))
        observable_parameters = get_observable_parameters(_formula)
        if !isempty(observable_parameters)
            variables_str *= observable_parameters * ", "
        end

        # Extract noise parameters
        _formula = filter(x -> !isspace(x), string(observables_data[:noiseFormula][i]))
        noise_parameters = get_noise_parameters(_formula)
        if !isempty(noise_parameters)
            variables_str *= noise_parameters * ", "
        end
    end

    # Remove last "," and Run @variables ... string to bring symbolic parameters into the scope
    variables_str = variables_str[1:(end - 2)]
    eval(Meta.parse(variables_str))

    return model_state_str, p_ode_problem_str, θ_non_dynamic_str
end

"""
    create∂σ∂_function(model_name::String,
                            dir_model::String,
                            parameter_info::ParametersInfo,
                            model_state_names::Vector{String},
                            p_ode_problem_names::Vector{String},
                            θ_non_dynamic_names::Vector{String},
                            observables_data::CSV.File,
                            model_SBML::SBMLImporter.ModelSBML)

    For model_name create a function for computing the standard deviation by translating the observables_data
"""
function create∂σ∂_function(model_name::String,
                            dir_model::String,
                            parameter_info::ParametersInfo,
                            model_state_names::Vector{String},
                            p_ode_problem_names::Vector{String},
                            θ_non_dynamic_names::Vector{String},
                            observables_data::CSV.File,
                            model_SBML::SBMLImporter.ModelSBML,
                            write_to_file::Bool)
    path_save = joinpath(dir_model, model_name * "_D_h_sd.jl")
    io1 = IOBuffer()
    io2 = IOBuffer()

    observable_ids = string.(observables_data[:observableId])
    p_observeble_str = ""
    u_observeble_str = ""
    for i in eachindex(observable_ids)

        # Each observebleID falls below its own if-statement
        p_observeble_str *= "\tif observableId == " * ":" * observable_ids[i] * "" * " \n"
        u_observeble_str *= "\tif observableId == " * ":" * observable_ids[i] * "" * " \n"

        _formula = filter(x -> !isspace(x), string(observables_data[:noiseFormula][i]))
        formula = replace_explicit_variable_rule(_formula, model_SBML)
        julia_formula = petab_formula_to_Julia(formula, model_state_names, parameter_info,
                                               p_ode_problem_names, θ_non_dynamic_names)

        enter_observable = true
        for iState in eachindex(model_state_names)
            if occursin(Regex("\\b" * model_state_names[iState] * "\\b"), julia_formula)
                noise_parameters = get_noise_parameters(formula)
                if !isempty(noise_parameters) && enter_observable == true
                    u_observeble_str *= "\t\t" * noise_parameters *
                                        " = get_obs_sd_parameter(θ_sd, parameter_map)\n"
                    enter_observable = false
                end

                julia_formula_symbolic = eval(Meta.parse(julia_formula))
                ui_symbolic = eval(Meta.parse(model_state_names[iState]))
                _∂σ∂ui = string(Symbolics.derivative(julia_formula_symbolic, ui_symbolic;
                                                     simplify = true))
                ∂σ∂ui = variables_to_array_index(_∂σ∂ui, model_state_names, parameter_info,
                                                 p_ode_problem_names, θ_non_dynamic_names,
                                                 p_ode_problem = true)
                u_observeble_str *= "\t\tout[" * string(iState) * "] = " * ∂σ∂ui * "\n"
            end
        end

        enter_observable = true
        for ip in eachindex(p_ode_problem_names)
            if occursin(Regex("\\b" * p_ode_problem_names[ip] * "\\b"), julia_formula)
                noise_parameters = get_noise_parameters(formula)
                if !isempty(noise_parameters) && enter_observable == true
                    p_observeble_str *= "\t\t" * noise_parameters *
                                        " = get_obs_sd_parameter(θ_sd, parameter_map)\n"
                    enter_observable = false
                end

                julia_formula_symbolic = eval(Meta.parse(julia_formula))
                pi_symbolic = eval(Meta.parse(p_ode_problem_names[ip]))
                _∂σ∂pi = string(Symbolics.derivative(julia_formula_symbolic, pi_symbolic;
                                                     simplify = true))
                ∂σ∂pi = variables_to_array_index(_∂σ∂pi, model_state_names, parameter_info,
                                                 p_ode_problem_names, θ_non_dynamic_names,
                                                 p_ode_problem = true)
                p_observeble_str *= "\t\tout[" * string(ip) * "] = " * ∂σ∂pi * "\n"
            end
        end

        u_observeble_str *= "\t\t" * "return nothing\n" * "\tend\n\n"
        p_observeble_str *= "\t\t" * "return nothing\n" * "\tend\n\n"
    end

    write(io1,
          "function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                   parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) \n")
    write(io1, u_observeble_str)
    write(io1, "end")
    ∂σ∂σuStr = String(take!(io1))
    if write_to_file == true
        str_write = ∂σ∂σuStr * "\n\n"
        open(path_save, "a") do f
            write(f, str_write)
        end
    end

    write(io2,
          "function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                   parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) \n")
    write(io2, p_observeble_str)
    write(io2, "end")
    ∂σ∂σpStr = String(take!(io2))
    if write_to_file == true
        str_write = ∂σ∂σpStr * "\n\n"
        open(path_save, "a") do f
            write(f, str_write)
        end
    end

    close(io1)
    close(io2)
    return ∂σ∂σuStr, ∂σ∂σpStr
end
