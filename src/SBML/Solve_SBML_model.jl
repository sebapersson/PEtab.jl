"""
solve_SBML(path_SBML, solver, tspan; abstol=1e-8, reltol=1e-8, saveat=Float64[], verbose=true)

Solve an ODE SBML model at the values reported in the SBML file over the specified time span (t0::Float, tend::Float).

Solvers from the OrdinaryDiffEq.jl package are supported. If you want to save the ODE solution at specific time-points, 
e.g., [1.0, 3.0], provide the `saveat` argument as `saveat=[1.0, 3.0]`. The output is provided in the format of OrdinaryDiffEq.jl. 
The Julia model files are saved in the same directory as the SBML file, in a subdirectory named "SBML".

!!! note
    This function is primarily intended for testing the SBML importer.
"""
function solve_SBML(path_SBML, solver, tspan; abstol=1e-8, reltol=1e-8, saveat::Vector{Float64}=Float64[], verbose::Bool=true)

    @assert isfile(path_SBML) "SBML file does not exist"

    verbose && @info "Building ODE system for file at $path_SBML"
    model_name = splitdir(path_SBML)[2][1:end-4]
    dir_save = joinpath(splitdir(path_SBML)[1], "SBML")
    if !isdir(dir_save)
        mkdir(dir_save)
    end
    pathODE = joinpath(dir_save, "ODE_" * model_name * ".jl")
    SBML_dict, _ = SBML_to_ModellingToolkit(path_SBML, pathODE, model_name, ifelse_to_event=true)

    #println("get_function_str(pathODE, 1)[1] = ", get_function_str(pathODE, 1)[1])

    verbose && @info "Symbolically processing system"
    _get_ode_system = @RuntimeGeneratedFunction(Meta.parse(get_function_str(pathODE, 1)[1]))
    _ode_system, state_map, parameter_map = _get_ode_system("https://xkcd.com/303/") # Argument needed by @RuntimeGeneratedFunction
    if isempty(SBML_dict["algebraicRules"])
        ode_system = structural_simplify(_ode_system)
    # DAE requires special processing
    else
        ode_system = structural_simplify(dae_index_lowering(_ode_system))
    end

    # Build callback function 
    p_ode_problem_names = string.(parameters(ode_system))
    model_state_names = replace.(string.(states(ode_system)), "(t)" => "")
    model_name = replace(model_name, "-" => "_")
    write_callbacks_str = "function getCallbacks_" * model_name * "()\n"
    write_tstops_str = "\nfunction computeTstops(u::AbstractVector, p::AbstractVector)\n"

    # In case we do not have any events
    verbose && @info "Building callbacks"
    if isempty(SBML_dict["boolVariables"]) && isempty(SBML_dict["events"])
        callback_names = ""
        check_activated_t0_names = ""
        write_tstops_str *= "\t return Float64[]\nend\n"
    else
        model_state_names = isempty(model_state_names) ? String[] : model_state_names
        for key in keys(SBML_dict["boolVariables"])
            function_str, callback_str =  create_callback(key, SBML_dict, p_ode_problem_names, string.(model_state_names))
            write_callbacks_str *= function_str * "\n"
            write_callbacks_str *= callback_str * "\n"
        end
        for key in keys(SBML_dict["events"])
            function_str, callback_str = create_callback_event(key, SBML_dict, p_ode_problem_names, string.(model_state_names))
            write_callbacks_str *= function_str * "\n"
            write_callbacks_str *= callback_str * "\n"
        end

        _callback_names = vcat([key for key in keys(SBML_dict["boolVariables"])], [key for key in keys(SBML_dict["events"])])
        callback_names = prod(["cb_" * name * ", " for name in _callback_names])[1:end-2]
        # Only relevant for picewise expressions 
        if !isempty(SBML_dict["boolVariables"])
            check_activated_t0_names = prod(["is_active_t0_" * key * "!, " for key in keys(SBML_dict["boolVariables"])])[1:end-2]
        else
            check_activated_t0_names = ""
        end
        _write_tstops_str, convert_tspan = create_tstops_function(SBML_dict, model_state_names, p_ode_problem_names, nothing)
        write_tstops_str *= "\treturn" * _write_tstops_str * "\n" * "end" * "\n"
    end
    convert_tspan = false
    write_callbacks_str *= "\treturn CallbackSet(" * callback_names * "), Function[" * check_activated_t0_names * "], " * string(convert_tspan)  * "\nend"
    fileWrite = dir_save * "/" * model_name * "_callbacks.jl"
    if isfile(fileWrite)
        rm(fileWrite)
    end
    io = open(fileWrite, "w")
    write(io, write_callbacks_str * "\n\n")
    write(io, write_tstops_str)
    close(io)

    strGetCallbacks = get_function_str(fileWrite, 2)
    get_callback_function = @RuntimeGeneratedFunction(Meta.parse(strGetCallbacks[1]))
    cbset, check_cb_active, convert_tspan = get_callback_function("https://xkcd.com/2694/") # Argument needed by @RuntimeGeneratedFunction
    computeTstops = @RuntimeGeneratedFunction(Meta.parse(strGetCallbacks[2]))

    verbose && @info "Solving ODE"

    ode_problem = ODEProblem(ode_system, state_map, tspan, parameter_map, jac=true)
    tstops = computeTstops(ode_problem.u0, ode_problem.p)
    for f! in check_cb_active
        f!(ode_problem.u0, ode_problem.p)
    end

    return solve(ode_problem, solver, abstol=abstol, reltol=reltol, saveat=saveat, tstops=tstops, callback=cbset)
end