"""
    create_σ_h_u0_file(model_name::String,
                       dir_model::String,
                       system::ODESystem,
                       state_map,
                       SBML_dict::Dict;
                       verbose::Bool=false)

    For a PeTab model with name model_name with all PeTab-files in dir_model and associated
    ModellingToolkit ODESystem (with its state_map) build a file containing a functions for
    i) computing the observable model value (h) ii) compute the initial value u0 (by using the
    state_map) and iii) computing the standard error (σ) for each observableFormula in the
    observables PeTab file.

    Note - The produced Julia file will go via the JIT-compiler. The SBML-dict is needed as
    sometimes variables are encoded via explicit-SBML rules.
"""
function create_σ_h_u0_file(model_name::String,
                            path_yaml::String,
                            dir_julia::String,
                            system::ODESystem,
                            parameter_map,
                            state_map,
                            SBML_dict::Dict;
                            custom_parameter_values::Union{Nothing, Dict}=nothing, 
                            write_to_file::Bool=true)

    p_ode_problem_names = string.(parameters(system))
    model_state_names = replace.(string.(states(system)), "(t)" => "")

    experimental_conditions, measurements_data, parameters_data, observables_data = read_petab_files(path_yaml)
    parameter_info = process_parameters(parameters_data, custom_parameter_values=custom_parameter_values)
    measurement_info = process_measurements(measurements_data, observables_data)

    # Indices for keeping track of parameters in θ
    θ_indices = compute_θ_indices(parameter_info, measurement_info, system, parameter_map, state_map, experimental_conditions)

    h_str = create_h_function(model_name, dir_julia, model_state_names, parameter_info, p_ode_problem_names,
                              string.(θ_indices.θ_non_dynamic_names), observables_data, SBML_dict, write_to_file)

    u0!_str = create_u0_function(model_name, dir_julia, parameter_info, p_ode_problem_names, state_map, write_to_file, SBML_dict, inplace=true)

    u0_str = create_u0_function(model_name, dir_julia, parameter_info, p_ode_problem_names, state_map, write_to_file, SBML_dict, inplace=false)

    σ_str = create_σ_function(model_name, dir_julia, parameter_info, model_state_names, p_ode_problem_names, string.(θ_indices.θ_non_dynamic_names), observables_data, SBML_dict, write_to_file)

    return h_str, u0!_str, u0_str, σ_str
end
"""
    When parsed from Julia input.
"""
function create_σ_h_u0_file(model_name::String,
                            system,
                            experimental_conditions::CSV.File,
                            measurements_data::CSV.File,
                            parameters_data::CSV.File,
                            observables_data::CSV.File,
                            state_map)::NTuple{4, String}

    p_ode_problem_names = string.(parameters(system))
    model_state_names = replace.(string.(states(system)), "(t)" => "")
    parameter_map = [p => 0.0 for p in parameters(system)]

    parameter_info = PEtab.process_parameters(parameters_data)
    measurement_info = PEtab.process_measurements(measurements_data, observables_data)

    # Indices for keeping track of parameters in θ
    θ_indices = PEtab.compute_θ_indices(parameter_info, measurement_info, system, parameter_map, state_map, experimental_conditions)

    # Dummary variables to keep PEtab importer happy even as we are not providing any PEtab files
    SBML_dict = Dict(); SBML_dict["species"] = Dict()

    h_str = PEtab.create_h_function(model_name, @__DIR__, model_state_names, parameter_info, p_ode_problem_names,
                                   string.(θ_indices.θ_non_dynamic_names), observables_data, SBML_dict, false)
    u0!_str = PEtab.create_u0_function(model_name, @__DIR__, parameter_info, p_ode_problem_names, state_map, false,
                                      SBML_dict, inplace=true)
    u0_str = PEtab.create_u0_function(model_name, @__DIR__, parameter_info, p_ode_problem_names, state_map, false,
                                     SBML_dict, inplace=false)
    σ_str = PEtab.create_σ_function(model_name, @__DIR__, parameter_info, model_state_names, p_ode_problem_names,
                                   string.(θ_indices.θ_non_dynamic_names), observables_data, SBML_dict, false)

    return h_str, u0!_str, u0_str, σ_str
end


"""
    create_h_function(model_name::String,
                      dir_model::String,
                      model_state_names::Vector{String},
                      paramData::ParametersInfo,
                      namesParamDyn::Vector{String},
                      namesNonDynParam::Vector{String},
                      observables_data::CSV.File,
                      SBML_dict::Dict)

    For model_name create a function for computing y_model by translating the observables_data
    PeTab-file into Julia syntax.
"""
function create_h_function(model_name::String,
                           dir_model::String,
                           model_state_names::Vector{String},
                           parameter_info::ParametersInfo,
                           p_ode_problem_names::Vector{String},
                           θ_non_dynamic_names::Vector{String},
                           observables_data::CSV.File,
                           SBML_dict::Dict, 
                           write_to_file::Bool)

    io = IOBuffer()
    path_save = joinpath(dir_model, model_name * "_h_sd_u0.jl")
    model_state_str, θ_dynamic_str,  θ_non_dynamic_str, constant_parameters_str = create_top_function_h(model_state_names, 
        parameter_info, p_ode_problem_names,  θ_non_dynamic_names)

    # Write the formula for each observable in Julia syntax
    observable_ids = string.(observables_data[:observableId])
    observable_str = ""
    for i in eachindex(observable_ids)

        _formula = filter(x -> !isspace(x), string(observables_data[:observableFormula][i]))
        observable_parameters = get_observable_parameters(_formula)
        observable_str *= "\tif observableId === " * ":" * observable_ids[i] * " \n"
        if !isempty(observable_parameters)
            observable_str *= "\t\t" * observable_parameters * " = get_obs_sd_parameter(θ_observable, parameter_map)\n"
        end

        formula = replace_explicit_variable_rule(_formula, SBML_dict)

        # Translate the formula for the observable to Julia syntax
        _julia_formula = petab_formula_to_Julia(formula, model_state_names, parameter_info, p_ode_problem_names,  θ_non_dynamic_names)
        julia_formula = variables_to_array_index(_julia_formula, model_state_names, parameter_info, p_ode_problem_names,  θ_non_dynamic_names, p_ode_problem=true)
        observable_str *= "\t\t" * "return " * julia_formula * "\n" * "\tend\n\n"
    end

    # Create h function
    if write_to_file == true
        write(io, model_state_str)
        write(io, θ_dynamic_str)
        write(io,  θ_non_dynamic_str)
        write(io, constant_parameters_str)
        write(io, "\n")
    end
    write(io, "function compute_h(u::AbstractVector, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                   θ_non_dynamic::AbstractVector, parameter_info::ParametersInfo, observableId::Symbol,
                      parameter_map::θObsOrSdParameterMap)::Real \n")
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
                          paramData::ParametersInfo,
                          namesParamODEProb::Vector{String},
                          namesNonDynParam::Vector{String})

    Extracts all variables needed for the observable h function.
"""
function create_top_function_h(model_state_names::Vector{String},
                               parameter_info::ParametersInfo,
                               p_ode_problem_names::Vector{String},
                               θ_non_dynamic_names::Vector{String})

    model_state_str = "#"
    for i in eachindex(model_state_names)
        model_state_str *= "u[" * string(i) * "] = " * model_state_names[i] * ", "
    end
    model_state_str = model_state_str[1:end-2] # Remove last non needed ", "
    model_state_str *= "\n"

    θ_dynamic_str = "#"
    for i in eachindex(p_ode_problem_names)
        θ_dynamic_str *= "p_ode_problem_names[" * string(i) * "] = " * p_ode_problem_names[i] * ", "
    end
    θ_dynamic_str = θ_dynamic_str[1:end-2] # Remove last non needed ", "
    θ_dynamic_str *= "\n"

    θ_non_dynamic_str = "#"
    if !isempty(θ_non_dynamic_names)
        for i in eachindex(θ_non_dynamic_names)
            θ_non_dynamic_str *= "θ_non_dynamic[" * string(i)* "] = " *  θ_non_dynamic_names[i] * ", "
        end
        θ_non_dynamic_str =  θ_non_dynamic_str[1:end-2] # Remove last non needed ", "
        θ_non_dynamic_str *= "\n"
    end

    constant_parameters_str = ""
    for i in eachindex(parameter_info.parameter_id)
        if parameter_info.estimate[i] == false
            constant_parameters_str *= "#parameter_info.nominalValue[" * string(i) *"] = " * string(parameter_info.parameter_id[i]) * "_C \n"
        end
    end
    constant_parameters_str *= "\n"

    return model_state_str, θ_dynamic_str,  θ_non_dynamic_str, constant_parameters_str
end


"""
    create_u0_function(model_name::String,
                       dir_model::String,
                       parameter_info::ParametersInfo,
                       p_ode_problem_names::Vector{String},
                       state_map, 
                       SBML_dict;
                       inplace::Bool=true)

    For model_name create a function for computing initial value by translating the state_map
    into Julia syntax.

    To correctly create the function the name of all parameters, paramData (to get constant parameters)
    are required.
"""
function create_u0_function(model_name::String,
                            dir_model::String,
                            parameter_info::ParametersInfo,
                            p_ode_problem_names::Vector{String},
                            state_map,
                            write_to_file::Bool, 
                            SBML_dict;
                            inplace::Bool=true)

    path_save = joinpath(dir_model, model_name * "_h_sd_u0.jl")                            
    io = IOBuffer()

    if inplace == true
        write(io, "function compute_u0!(u0::AbstractVector, p_ode_problem::AbstractVector) \n\n")
    else
        write(io, "function compute_u0(p_ode_problem::AbstractVector)::AbstractVector \n\n")
    end

    # Write named list of parameter to file
    p_ode_problem_str = "\t#"
    for i in eachindex(p_ode_problem_names)
        p_ode_problem_str *= "p_ode_problem[" * string(i) * "] = " * p_ode_problem_names[i] * ", "
    end
    p_ode_problem_str = p_ode_problem_str[1:end-2]
    p_ode_problem_str *= "\n\n"
    write(io, p_ode_problem_str)

    write(io, "\tt = 0.0 # u at time zero\n\n")

    # Write the formula for each initial condition to file
    _model_state_names = [replace.(string.(state_map[i].first), "(t)" => "") for i in eachindex(state_map)]
    # If we create from Julia model (e.g.) Catalyst this is not applicable and step is skipped 
    if "parameters" ∈ keys(SBML_dict) # When built from SBML 
        model_state_names = filter(x -> x ∈ keys(SBML_dict["species"]) && SBML_dict["species"][x].assignment_rule == false, _model_state_names)
    else
        model_state_names = _model_state_names
    end
    model_state_str = ""
    for i in eachindex(model_state_names)
        stateName = model_state_names[i]
        _stateExpression = replace(string(state_map[i].second), " " => "")
        stateFormula = petab_formula_to_Julia(_stateExpression, model_state_names, parameter_info, p_ode_problem_names, String[])
        for i in eachindex(p_ode_problem_names)
            stateFormula = replace_variable(stateFormula, p_ode_problem_names[i], "p_ode_problem["*string(i)*"]")
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
        model_state_str = model_state_str[1:end-2]
        write(io, model_state_str)

    # Where we return the entire initial value vector
    elseif inplace == false
        model_state_str = "\t return ["
        for i in eachindex(model_state_names)
            model_state_str *= model_state_names[i] * ", "
        end
        model_state_str = model_state_str[1:end-2]
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
                      dir_model::String,
                      parameter_info::ParametersInfo,
                      model_state_names::Vector{String},
                      p_ode_problem_names::Vector{String},
                      θ_non_dynamic_names::Vector{String},
                      observables_data::CSV.File,
                      SBML_dict::Dict)

    For model_name create a function for computing the standard deviation σ by translating the observables_data
    PeTab-file into Julia syntax.
"""
function create_σ_function(model_name::String,
                           dir_model::String,
                           parameter_info::ParametersInfo,
                           model_state_names::Vector{String},
                           p_ode_problem_names::Vector{String},
                           θ_non_dynamic_names::Vector{String},
                           observables_data::CSV.File,
                           SBML_dict::Dict, 
                           write_to_file::Bool)

    path_save = joinpath(dir_model, model_name * "_h_sd_u0.jl")
    io = IOBuffer()

    # Write the formula for standard deviations to file
    observable_ids = string.(observables_data[:observableId])
    observable_str = ""
    for i in eachindex(observable_ids)

        _formula = filter(x -> !isspace(x), string(observables_data[:noiseFormula][i]))
        noise_parameters = get_noise_parameters(_formula)
        observable_str *= "\tif observableId === " * ":" * observable_ids[i] * " \n"
        if !isempty(noise_parameters)
            observable_str *= "\t\t" * noise_parameters * " = get_obs_sd_parameter(θ_sd, parameter_map)\n"
        end

        formula = replace_explicit_variable_rule(_formula, SBML_dict)

        # Translate the formula for the observable to Julia syntax
        _julia_formula = petab_formula_to_Julia(formula, model_state_names, parameter_info, p_ode_problem_names,  θ_non_dynamic_names)
        julia_formula = variables_to_array_index(_julia_formula, model_state_names, parameter_info, p_ode_problem_names,  θ_non_dynamic_names, p_ode_problem=true)
        observable_str *= "\t\t" * "return " * julia_formula * "\n" * "\tend\n\n"
    end

    write(io, "function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                   parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap)::Real \n")
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
