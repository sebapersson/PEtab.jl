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
    SBMLDict, _ = XmlToModellingToolkit(path_SBML, pathODE, model_name, ifelse_to_event=true)

    #println("getFunctionsAsString(pathODE, 1)[1] = ", getFunctionsAsString(pathODE, 1)[1])

    verbose && @info "Symbolically processing system"
    _getODESystem = @RuntimeGeneratedFunction(Meta.parse(getFunctionsAsString(pathODE, 1)[1]))
    _odeSystem, state_map, parameter_map = _getODESystem("https://xkcd.com/303/") # Argument needed by @RuntimeGeneratedFunction
    if isempty(SBMLDict["algebraicRules"])
        odeSystem = structural_simplify(_odeSystem)
    # DAE requires special processing
    else
        odeSystem = structural_simplify(dae_index_lowering(_odeSystem))
    end

    # Build callback function 
    pODEProblemNames = string.(parameters(odeSystem))
    modelStateNames = replace.(string.(states(odeSystem)), "(t)" => "")
    model_name = replace(model_name, "-" => "_")
    stringWriteCallbacks = "function getCallbacks_" * model_name * "()\n"
    stringWriteTstops = "\nfunction computeTstops(u::AbstractVector, p::AbstractVector)\n"

    # In case we do not have any events
    verbose && @info "Building callbacks"
    if isempty(SBMLDict["boolVariables"]) && isempty(SBMLDict["events"])
        callbackNames = ""
        checkIfActivatedT0Names = ""
        stringWriteTstops *= "\t return Float64[]\nend\n"
    else
        modelStateNames = isempty(modelStateNames) ? String[] : modelStateNames
        for key in keys(SBMLDict["boolVariables"])
            functionsStr, callbackStr =  createCallback(key, SBMLDict, pODEProblemNames, string.(modelStateNames))
            stringWriteCallbacks *= functionsStr * "\n"
            stringWriteCallbacks *= callbackStr * "\n"
        end
        for key in keys(SBMLDict["events"])
            functionsStr, callbackStr = createCallbackForEvent(key, SBMLDict, pODEProblemNames, string.(modelStateNames))
            stringWriteCallbacks *= functionsStr * "\n"
            stringWriteCallbacks *= callbackStr * "\n"
        end

        _callbackNames = vcat([key for key in keys(SBMLDict["boolVariables"])], [key for key in keys(SBMLDict["events"])])
        callbackNames = prod(["cb_" * name * ", " for name in _callbackNames])[1:end-2]
        # Only relevant for picewise expressions 
        if !isempty(SBMLDict["boolVariables"])
            checkIfActivatedT0Names = prod(["isActiveAtTime0_" * key * "!, " for key in keys(SBMLDict["boolVariables"])])[1:end-2]
        else
            checkIfActivatedT0Names = ""
        end
        stringWriteTstops *= "\treturn" * createFuncionForTstops(SBMLDict, modelStateNames, pODEProblemNames, nothing) * "\n" * "end" * "\n"
    end
    convert_tspan = false
    stringWriteCallbacks *= "\treturn CallbackSet(" * callbackNames * "), Function[" * checkIfActivatedT0Names * "], " * string(convert_tspan)  * "\nend"
    fileWrite = dir_save * "/" * model_name * "_callbacks.jl"
    if isfile(fileWrite)
        rm(fileWrite)
    end
    io = open(fileWrite, "w")
    write(io, stringWriteCallbacks * "\n\n")
    write(io, stringWriteTstops)
    close(io)

    strGetCallbacks = getFunctionsAsString(fileWrite, 2)
    getCallbackFunction = @RuntimeGeneratedFunction(Meta.parse(strGetCallbacks[1]))
    cbSet, checkCbActive, convert_tspan = getCallbackFunction("https://xkcd.com/2694/") # Argument needed by @RuntimeGeneratedFunction
    computeTstops = @RuntimeGeneratedFunction(Meta.parse(strGetCallbacks[2]))

    verbose && @info "Solving ODE"

    odeProblem = ODEProblem(odeSystem, state_map, tspan, parameter_map, jac=true)
    tStops = computeTstops(odeProblem.u0, odeProblem.p)
    for f! in checkCbActive
        f!(odeProblem.u0, odeProblem.p)
    end

    return solve(odeProblem, solver, abstol=abstol, reltol=reltol, saveat=saveat, tstops=tStops, callback=cbSet)
end