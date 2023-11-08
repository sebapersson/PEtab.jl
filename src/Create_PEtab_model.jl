"""
    PEtabModel(path_yaml::String;
               build_julia_files::Bool=false,
               verbose::Bool=true,
               ifelse_to_event::Bool=true,
               write_to_file::Bool=true,
               jlfile_path::String="")::PEtabModel

Create a PEtabModel from a PEtab specified problem with a YAML-file located at `path_yaml`.

When parsing a PEtab problem, several things happen under the hood:

1. The SBML file is translated into `ModelingToolkit.jl` format to allow for symbolic computations of the ODE-model Jacobian. Piecewise and model events are further written into `DifferentialEquations.jl` callbacks.
2. The observable PEtab table is translated into a Julia file with functions for computing the observable (`h`), noise parameter (`σ`), and initial values (`u0`).
3. To allow gradients via adjoint sensitivity analysis and/or forward sensitivity equations, the gradients of `h` and `σ` are computed symbolically with respect to the ODE model's states (`u`) and parameters (`ode_problem.p`).

All of this happens automatically, and resulting files are stored under `petab_model.dir_julia` assuming write_to_file=true. To save time, `forceBuildJlFiles=false` by default, which means that Julia files are not rebuilt if they already exist.

# Arguments
- `path_yaml::String`: Path to the PEtab problem YAML file.
- `build_julia_files::Bool=false`: If `true`, forces the creation of Julia files for the problem even if they already exist.
- `verbose::Bool=true`: If `true`, displays verbose output during parsing.
- `ifelse_to_event::Bool=true`: If `true`, rewrites `if-else` statements in the SBML model as event-based callbacks.
- `write_to_file::Bool=true`: If `true`, writes built Julia files to disk (recomended)

# Example
```julia
petab_model = PEtabModel("path_to_petab_problem_yaml")
```
"""
function PEtabModel(path_yaml::String;
                    build_julia_files::Bool=false,
                    verbose::Bool=true,
                    ifelse_to_event::Bool=true,
                    custom_parameter_values::Union{Nothing, Dict}=nothing, 
                    write_to_file::Bool=true)::PEtabModel

    path_SBML, path_parameters, path_conditions, path_observables, path_measurements, dir_julia, dir_model, model_name = read_petab_yaml(path_yaml)

    verbose == true && @info "Building PEtabModel for $model_name"

    path_model_jl_file = joinpath(dir_julia, model_name * ".jl")
    if !isfile(path_model_jl_file) || build_julia_files == true
        verbose == true && printstyled("[ Info:", color=123, bold=true)
        verbose == true && build_julia_files && print(" By user option rebuilds Julia ODE model ...")
        verbose == true && !build_julia_files && print(" Building Julia model file as it does not exist ...")

        b_build = @elapsed model_dict, model_str = SBML_to_ModellingToolkit(path_SBML, path_model_jl_file, model_name, 
            ifelse_to_event=ifelse_to_event, write_to_file=write_to_file)
        verbose == true && @printf(" done. Time = %.1es\n", b_build)
    end

    if isfile(path_model_jl_file) && build_julia_files == false
        verbose == true && printstyled("[ Info:", color=123, bold=true)
        verbose == true && print(" Julia model file exists and will not be rebuilt\n")
        model_str = get_function_str(path_model_jl_file, 1)[1]
    end

    model_str = add_parameter_condition_initial_values(model_str, path_conditions, path_parameters, path_model_jl_file, write_to_file)

    # For down the line processing model dict is required 
    if !@isdefined(model_dict)
        model_dict, _ = SBML_to_ModellingToolkit(path_SBML, path_model_jl_file, model_name, write_to_file=false, 
                                                 only_extract_model_dict=true, ifelse_to_event=ifelse_to_event)
    end

    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Symbolically processes ODE-system ...")
    timeTake = @elapsed begin
        _get_ode_system = @RuntimeGeneratedFunction(Meta.parse(model_str))
        _ode_system, state_map, parameter_map = _get_ode_system("https://xkcd.com/303/") # Argument needed by @RuntimeGeneratedFunction
        if "algebraic_rules" ∉ keys(model_dict) || isempty(model_dict["algebraic_rules"])
            ode_system = structural_simplify(_ode_system)
        # DAE requires special processing
        else
            ode_system = structural_simplify(dae_index_lowering(_ode_system))
        end
        parameter_names = parameters(ode_system)
        state_names = states(ode_system)
    end
    verbose == true && @printf(" done. Time = %.1es\n", timeTake)

    path_u0_h_sigma = joinpath(dir_julia, model_name * "_h_sd_u0.jl")
    if !isfile(path_u0_h_sigma) || build_julia_files == true
        verbose == true && printstyled("[ Info:", color=123, bold=true)
        verbose == true && !isfile(path_u0_h_sigma) && print(" Building u0, h and σ file as it does not exist ...")
        verbose == true && isfile(path_u0_h_sigma) && print(" By user option rebuilds u0, h and σ file ...")
        if !@isdefined(model_dict)
            model_dict, _ = SBML_to_ModellingToolkit(path_SBML, path_model_jl_file, model_name, write_to_file=false, 
                                                     only_extract_model_dict=true, ifelse_to_event=ifelse_to_event)
        end
        b_build = @elapsed h_str, u0!_str, u0_str, σ_str = create_σ_h_u0_file(model_name, path_yaml, dir_julia, ode_system, 
                                                                              parameter_map, state_map, model_dict, 
                                                                              custom_parameter_values=custom_parameter_values, 
                                                                              write_to_file=write_to_file)
        verbose == true && @printf(" done. Time = %.1es\n", b_build)
    else
        verbose == true && printstyled("[ Info:", color=123, bold=true)
        verbose == true && print(" u0, h and σ file exists and will not be rebuilt\n")
        h_str, u0!_str, u0_str, σ_str = get_function_str(path_u0_h_sigma, 4)
    end
    compute_h = @RuntimeGeneratedFunction(Meta.parse(h_str))
    compute_u0! = @RuntimeGeneratedFunction(Meta.parse(u0!_str))
    compute_u0 = @RuntimeGeneratedFunction(Meta.parse(u0_str))
    compute_σ = @RuntimeGeneratedFunction(Meta.parse(σ_str))
    

    path_D_h_sd = joinpath(dir_julia, model_name * "_D_h_sd.jl")
    if !isfile(path_D_h_sd) || build_julia_files == true
        verbose == true && printstyled("[ Info:", color=123, bold=true)
        verbose == true && !isfile(path_u0_h_sigma) && print(" Building ∂h∂p, ∂h∂u, ∂σ∂p and ∂σ∂u file as it does not exist ...")
        verbose == true && isfile(path_u0_h_sigma) && print(" By user option rebuilds ∂h∂p, ∂h∂u, ∂σ∂p and ∂σ∂u file ...")
        if !@isdefined(model_dict)
            model_dict, _ = SBML_to_ModellingToolkit(path_SBML, path_model_jl_file, model_name, write_to_file=false, 
                                                     only_extract_model_dict=true, ifelse_to_event=ifelse_to_event)
        end
        b_build = @elapsed ∂h∂u_str, ∂h∂p_str, ∂σ∂u_str, ∂σ∂p_str = create_derivative_σ_h_file(model_name, path_yaml, 
                                                                                          dir_julia, ode_system, 
                                                                                          parameter_map, state_map, 
                                                                                          model_dict, 
                                                                                          custom_parameter_values=custom_parameter_values, 
                                                                                          write_to_file=write_to_file)
        verbose == true && @printf(" done. Time = %.1es\n", b_build)
    else verbose == true
        verbose == true && printstyled("[ Info:", color=123, bold=true)
        verbose == true && print(" ∂h∂p, ∂h∂u, ∂σ∂p and ∂σ∂u file exists and will not be rebuilt\n")
        ∂h∂u_str, ∂h∂p_str, ∂σ∂u_str, ∂σ∂p_str = get_function_str(path_D_h_sd, 4)
    end
    compute_∂h∂u! = @RuntimeGeneratedFunction(Meta.parse(∂h∂u_str))
    compute_∂h∂p! = @RuntimeGeneratedFunction(Meta.parse(∂h∂p_str))
    compute_∂σ∂σu! = @RuntimeGeneratedFunction(Meta.parse(∂σ∂u_str))
    compute_∂σ∂σp! = @RuntimeGeneratedFunction(Meta.parse(∂σ∂p_str))

    path_callback = joinpath(dir_julia, model_name * "_callbacks.jl")
    if !isfile(path_callback) || build_julia_files == true
        verbose == true && printstyled("[ Info:", color=123, bold=true)
        verbose == true && !isfile(path_callback) && print(" Building callback file as it does not exist ...")
        verbose == true && isfile(path_callback) && print(" By user option rebuilds callback file ...")
        if !@isdefined(model_dict)
            model_dict, _ = SBML_to_ModellingToolkit(path_SBML, path_model_jl_file, model_name, write_to_file=false, 
                only_extract_model_dict=true, ifelse_to_event=ifelse_to_event)
        end
        b_build = @elapsed callback_str, tstops_str = create_callbacks_SBML(ode_system, parameter_map, 
            state_map, model_dict, model_name, path_yaml, dir_julia, custom_parameter_values=custom_parameter_values, 
            write_to_file=write_to_file)
        verbose == true && @printf(" done. Time = %.1es\n", b_build)
    else
        verbose == true && printstyled("[ Info:", color=123, bold=true)
        verbose == true && print(" Callback file exists and will not be rebuilt\n")
        callback_str, tstops_str = get_function_str(path_callback, 2)
    end
    get_callback_function = @RuntimeGeneratedFunction(Meta.parse(callback_str))
    cbset, check_cb_active, convert_tspan = get_callback_function("https://xkcd.com/2694/") # Argument needed by @RuntimeGeneratedFunction
    compute_tstops = @RuntimeGeneratedFunction(Meta.parse(tstops_str))

    petab_model = PEtabModel(model_name,
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
                             ode_system,
                             parameter_map,
                             state_map,
                             parameter_names,
                             state_names,
                             dir_model,
                             dir_julia,
                             CSV.File(path_measurements, stringtype=String),
                             CSV.File(path_conditions, stringtype=String),
                             CSV.File(path_observables, stringtype=String),
                             CSV.File(path_parameters, stringtype=String),
                             path_SBML,
                             path_yaml,
                             cbset,
                             check_cb_active, 
                             false)

    return petab_model
end


# For reading the run-time generated PEtab-related functions which via Meta.parse are passed
# on to @RuntimeGeneratedFunction to build the PEtab related functions without world-problems.
function get_function_str(file_path::AbstractString, n_functions::Int64)::Vector{String}

    f_start, f_end = zeros(Int64, n_functions), zeros(Int64, n_functions)
    i_function = 1
    in_function::Bool = false
    n_lines = open(file_path, "r") do f countlines(f) end
    body_str = Vector{String}(undef, n_lines)

    f = open(file_path, "r")
    for (i_line, line) in pairs(readlines(f))

        if length(line) ≥ 8 && line[1:8] == "function"
            f_start[i_function] = i_line
            in_function = true
        end

        if length(line) ≥ 3 && line[1] != '#' && line[1:3] == "end"
            f_end[i_function] = i_line
            in_function = false
            i_function += 1
        end

        body_str[i_line] = string(line)
    end
    close(f)

    out = Vector{String}(undef, n_functions)
    for i in eachindex(out)

        # Runtime generated functions requrie at least on function argument input, hence if missing we
        # add a foo argument
        if body_str[f_start[i]][end-1:end] == "()"
            body_str[f_start[i]] = body_str[f_start[i]][1:end-2] * "(foo)"
        end

        out[i] = prod([body_str[j] * '\n' for j in f_start[i]:f_end[i]])
    end
    return out
end


# The PEtab standard allows the condition table to have headers which corresponds to states. In order for this to
# be compatible with gradient compuations we add such initial values as an additional parameter in ode_problem.p
# by overwriting the Julia-model file
function add_parameter_condition_initial_values(model_str::String,
                                                path_conditions::String,
                                                path_parameters::String, 
                                                path_julia_file::String,
                                                write_to_file::Bool)

    # Load necessary data
    experimental_conditions_file = CSV.File(path_conditions)
    parameters_file = CSV.File(path_parameters)

    # Extract return line to find names of trueParameterValues and ODESystem
    return_line = filter(line -> occursin(r"\s*return",line), split(model_str, "\n"))[1]
    return_line = replace(return_line, r"\s*return"=>"")
    return_line = replace(return_line, " "=>"")
    return_outputs = split(return_line,",")
    ode_system_name = return_outputs[1]
    true_parameter_values_name = return_outputs[end]

    # Extract line with  ODESystem to find name of stateArray and parameterArray
    ode_line = filter(line -> occursin(Regex(ode_system_name * "\\s*=\\s*ODESystem\\("), line), split(model_str, "\n"))[1]    
    # Finds the offset after the parenthesis in "ODESystem("
    function_start_regex = Regex("\\bODESystem\\(\\K")
    # Matches parentheses pairs to grab the arguments of the "ODESystem(" function
    match_parentheses_regex = Regex("\\((?:[^)(]*(?R)?)*+\\)")
    function_start = match(function_start_regex, ode_line)
    function_start_position = function_start.offset
    inside_function = match(match_parentheses_regex, ode_line[function_start_position-1:end]).match
    inside_function = inside_function[2:end-1]
    inside_function = replace(inside_function, " "=>"")
    return_outputs = split(inside_function,",")
    parameter_array_name = string(return_outputs[end])
    state_array_name = string(return_outputs[end-1])

    # Extract state and parameter names
    state_names = get_state_parameter_names_jl_function(model_str, state_array_name, parameter_array_name, get_states=true)
    parameter_names = get_state_parameter_names_jl_function(model_str, state_array_name, parameter_array_name, get_states=false)
    
    # Check if the condition table contains states to map initial values
    column_names = string.(experimental_conditions_file.names)
    length(column_names) == 1 && return model_str
    i_start = column_names[2] == "conditionName" ? 3 : 2 # Sometimes PEtab file does not include column conditionName
    # Only change model file in case on of the experimental conditions map to a state (that is add an init parameter)
    if any(name -> name ∈ state_names, column_names[i_start:end]) == false
        return model_str
    end

    # Find states and create new parameter names and values
    which_states = (column_names[i_start:end])[findall(x -> x ∈ state_names, column_names[i_start:end])]
    new_parameter_names = "__init__" .* which_states .* "__"
    new_parameter_values = Vector{String}(undef, length(new_parameter_names))

    # Check if the columns for which the states are assigned contain parameters. If these parameters are not a part
    # of the ODE-system they have to be assigned to the ODE-system (since they determine an initial value they must
    # be considered dynamic parameters).
    for state in which_states
        for row_value in experimental_conditions_file[Symbol(state)]
            if typeof(row_value) <: Real
                continue
            elseif ismissing(row_value)
                continue
            elseif is_number(row_value) == true || string(row_value) ∈ parameter_names
                continue
            elseif row_value ∈ parameters_file[:parameterId]
                # Must be a parameter which did not appear in the SBML file
                new_parameter_names = vcat(new_parameter_names, row_value)
                new_parameter_values = vcat(new_parameter_values, "0.0")
            else
                @error "The condition table value $row_value does not correspond to any parameter in the SBML file parameters file"
            end
        end
    end

    # Check if the function has already been rewritten
    if any(x -> x ∈ parameter_names, new_parameter_names)
        return model_str
    end

    # Update function lines with new parameters and values
    function_line_by_line = split(model_str, '\n')
    lines_add = 0:0
    for i in eachindex(function_line_by_line)
        line_no_whitespace = replace(function_line_by_line[i], r"\s+" => "")

        # Check which lines new initial value parameters should be added to the parametersMap
        if length(line_no_whitespace) ≥ 13 && line_no_whitespace[1:13] == true_parameter_values_name
            lines_add = (i+1):(i+length(new_parameter_names))
        end

        # Add new parameters for ModelingToolkit.@parameters line
        if length(line_no_whitespace) ≥ 27 && line_no_whitespace[1:27] == "ModelingToolkit.@parameters"
            function_line_by_line[i] *= (" "*prod([str * " " for str in new_parameter_names]))[1:end-1]
        end

        # Add new parameters in parameterArray
        if length(line_no_whitespace) ≥ 10 && line_no_whitespace[1:10] == parameter_array_name
            function_line_by_line[i] = function_line_by_line[i][1:end-1] * ", " * (" "*prod([str * ", " for str in new_parameter_names]))[1:end-2] * "]"
        end

        # Move through state array
        for j in eachindex(which_states)
            if starts_with_x(line_no_whitespace, which_states[j])
                # Extract the default value
                _, default_value = split(line_no_whitespace, "=>")
                new_parameter_values[j] = default_value[end] == ',' ? default_value[1:end-1] : default_value[:]
                function_line_by_line[i] = "\t" * which_states[j] * " => " * new_parameter_names[j] * ","
            end
        end
    end

    function_line_by_line_new = Vector{String}(undef, length(function_line_by_line) + length(new_parameter_names))
    k = 1
    for i in eachindex(function_line_by_line_new)
        if i ∈ lines_add
            continue
        end
        function_line_by_line_new[i] = function_line_by_line[k]
        k += 1
    end
    # We need to capture default values
    function_line_by_line_new[lines_add] .= "\t" .* new_parameter_names .* " => " .* new_parameter_values .* ","

    new_function_string = function_line_by_line_new[1]
    new_function_string *= prod(row * "\n" for row in function_line_by_line_new[2:end])
    # Write the new function to the Julia file
    if write_to_file == true
        open(path_julia_file, "w") do f
            write(f, new_function_string)
            flush(f)
        end
    end

    return get_function_str(path_julia_file, 1)[1]
end


# Extract model state names from stateArray in the JL-file (and also parameter names)
function get_state_parameter_names_jl_function(fAsString::String, state_array_name::String, parameter_array_name::String; get_states::Bool=false)

    function_line_by_line = split(fAsString, '\n')
    for i in eachindex(function_line_by_line)
        line_no_whitespace = replace(function_line_by_line[i], " " => "")
        line_no_whitespace = replace(line_no_whitespace, "\t" => "")

        # Add new parameters in parameterArray
        if get_states == true
            len_statearray_name = length(state_array_name)
            if length(line_no_whitespace) ≥ len_statearray_name && line_no_whitespace[1:len_statearray_name] == state_array_name
                loc_array_start = findfirst("=[",line_no_whitespace)[end]
                return split(line_no_whitespace[loc_array_start+1:end-1], ",")
            end
        end

        if get_states == false
            len_parameterarray_name = length(parameter_array_name)
            if length(line_no_whitespace) ≥ len_parameterarray_name && line_no_whitespace[1:len_parameterarray_name] == parameter_array_name
                loc_array_start = findfirst("=[",line_no_whitespace)[end]
                return split(line_no_whitespace[loc_array_start+1:end-1], ",")
            end
        end
    end

end


# Check if a str starts with x
function starts_with_x(str, x)
    if length(str) < length(x)
        return false
    end

    if str[1:length(x)] == x && str[length(x)+1] ∈ [' ', '=']
        return true
    end
    return false
end
