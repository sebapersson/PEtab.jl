"""
solve_SBML(path_SBML, solver, tspan; abstol=1e-8, reltol=1e-8, saveat=Float64[], verbose=true)

Solve an ODE SBML model at the values reported in the SBML file over the specified time span (t0::Float, tend::Float).

Solvers from the OrdinaryDiffEq.jl package are supported. If you want to save the ODE solution at specific time-points, 
e.g., [1.0, 3.0], provide the `saveat` argument as `saveat=[1.0, 3.0]`. The output is provided in the format of OrdinaryDiffEq.jl. 


The Julia model files are saved in the same directory as the SBML file, in a subdirectory named "SBML".

!!! note
    This function is primarily intended for testing the SBML importer.
"""
function solve_SBML(path_SBML, solver, tspan; write_to_file::Bool=false, abstol=1e-8, reltol=1e-8, saveat::Vector{Float64}=Float64[], verbose::Bool=true)

    @assert isfile(path_SBML) "SBML file does not exist"
    model_SBML = build_SBML_model(path_SBML, ifelse_to_event=true)

    verbose && @info "Building ODE system for file at $path_SBML"
    model_name = splitdir(path_SBML)[2][1:end-4]
    
    path_save_ODE = joinpath(splitdir(path_SBML)[1], "SBML", "ODE_" * model_name * ".jl")
    if write_to_file == true
        dir_save = joinpath(splitdir(path_SBML)[1], "SBML")
        !isdir(dir_save) && mkdir(dir_save)
        open(path_save_ODE, "w") do f
            write(f, model_str)
        end
    end
    model_str = odesystem_from_SBML(model_SBML, path_save_ODE, model_name, write_to_file)

    verbose && @info "Symbolically processing system"
    _get_ode_system = @RuntimeGeneratedFunction(Meta.parse(model_str))
    _ode_system, state_map, parameter_map = _get_ode_system("https://xkcd.com/303/") # Argument needed by @RuntimeGeneratedFunction
    # For a DAE we need special processing to rewrite it into an ODE 
    if isempty(model_SBML.algebraic_rules)
        ode_system = structural_simplify(_ode_system)
    else
        ode_system = structural_simplify(dae_index_lowering(_ode_system))
    end

    # Build potential callback functions
    verbose && @info "Building callbacks"
    p_ode_problem_names = isempty(parameters(ode_system)) ? String[] : string.(parameters(ode_system))
    model_state_names = replace.(string.(states(ode_system)), "(t)" => "")
    model_name = replace(model_name, "-" => "_")
    write_callbacks_str = "function getCallbacks_" * model_name * "()\n"
    write_tstops_str = "\nfunction computeTstops(u::AbstractVector, p::AbstractVector)\n"
    # In case we do not have any events
    if isempty(model_SBML.ifelse_parameters) && isempty(model_SBML.events)
        callback_names = ""
        check_activated_t0_names = ""
        write_tstops_str *= "\t return Float64[]\nend\n"
    
    # In case of eevents
    else
        model_state_names = isempty(model_state_names) ? String[] : model_state_names
        for parameter in keys(model_SBML.ifelse_parameters)
            function_str, callback_str =  create_callback_ifelse(parameter, model_SBML, p_ode_problem_names, string.(model_state_names))
            write_callbacks_str *= function_str * "\n" * callback_str * "\n"
        end
        for key in keys(model_SBML.events)
            function_str, callback_str = create_callback_SBML_event(key, model_SBML, p_ode_problem_names, string.(model_state_names))
            write_callbacks_str *= function_str * "\n" * callback_str * "\n"
        end
        _callback_names = vcat([key for key in keys(model_SBML.ifelse_parameters)], [key for key in keys(model_SBML.events)])
        callback_names = prod(["cb_" * name * ", " for name in _callback_names])[1:end-2]
        # Only relevant for picewise expressions 
        if !isempty(model_SBML.ifelse_parameters)
            check_activated_t0_names = prod(["is_active_t0_" * key * "!, " for key in keys(model_SBML.ifelse_parameters)])[1:end-2]
        else
            check_activated_t0_names = ""
        end
        _write_tstops_str, convert_tspan = create_tstops_function(model_SBML, model_state_names, p_ode_problem_names, nothing)
        write_tstops_str *= "\treturn" * _write_tstops_str * "\n" * "end" * "\n"
    end
    convert_tspan = false
    write_callbacks_str *= "\treturn CallbackSet(" * callback_names * "), Function[" * check_activated_t0_names * "], " * string(convert_tspan)  * "\nend"
    if write_to_file == true
        path_save = joinpath(dir_save, model_name * "_callbacks.jl")
        isfile(path_save) && rm(path_save)
        open(path_save, "w") do f
            write(f, write_callbacks_str * "\n\n")
            write(f, write_tstops_str)
        end
    end

    get_callback_function = @RuntimeGeneratedFunction(Meta.parse(write_callbacks_str))
    cbset, check_cb_active, convert_tspan = get_callback_function("https://xkcd.com/2694/") # Argument needed by @RuntimeGeneratedFunction
    compute_tstops = @RuntimeGeneratedFunction(Meta.parse(write_tstops_str))

    # Build the ODEProblem and check if callbacks are active at time zero
    ode_problem = ODEProblem(ode_system, state_map, tspan, parameter_map, jac=true)
    for f! in check_cb_active
        f!(ode_problem.u0, ode_problem.p)
    end

    # In order for an event to trigger it must transition from false to true, which is checked under the 
    # time-stepping. However, sometimes the integrator can step directly to the event time so the flag 
    # stating the condition is false has not been triggered yet. Solution is that we trigger a tstop 
    # early on in the simulations 
    tstops = compute_tstops(ode_problem.u0, ode_problem.p)
    tstops = isempty(tstops) ? tstops : vcat(minimum(tstops) / 2.0, tstops)
    verbose && @info "Solving ODE"

    return solve(ode_problem, solver, abstol=abstol, reltol=reltol, saveat=saveat, tstops=tstops, callback=cbset)
end