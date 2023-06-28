"""
solveSBMLModel(pathSBML, solver, timeSpan; abstol=1e-8, reltol=1e-8, saveat=Float64[], verbose=true)

Solve an ODE SBML model at the values reported in the SBML file over the specified time span (t0::Float, tend::Float).

Solvers from the OrdinaryDiffEq.jl package are supported. If you want to save the ODE solution at specific time-points, 
e.g., [1.0, 3.0], provide the `saveat` argument as `saveat=[1.0, 3.0]`. The output is provided in the format of OrdinaryDiffEq.jl. 
The Julia model files are saved in the same directory as the SBML file, in a subdirectory named "SBML".

!!! note
    This function is primarily intended for testing the SBML importer.
"""
function solveSBMLModel(pathSBML, solver, timeSpan; abstol=1e-8, reltol=1e-8, saveat::Vector{Float64}=Float64[], verbose::Bool=true)

    @assert isfile(pathSBML) "SBML file does not exist"

    verbose && @info "Building ODE system"
    modelName = splitdir(pathSBML)[2][1:end-4]
    dirSave = joinpath(splitdir(pathSBML)[1], "SBML")
    if !isdir(dirSave)
        mkdir(dirSave)
    end
    pathODE = joinpath(dirSave, "ODE_" * modelName * ".jl")
    SBMLDict = XmlToModellingToolkit(pathSBML, pathODE, modelName, ifElseToEvent=true)

    verbose && @info "Symbolically processing system"
    _getODESystem = @RuntimeGeneratedFunction(Meta.parse(getFunctionsAsString(pathODE, 1)[1]))
    _odeSystem, stateMap, parameterMap = _getODESystem("https://xkcd.com/303/") # Argument needed by @RuntimeGeneratedFunction
    odeSystem = structural_simplify(_odeSystem)

    # Build callback function 
    pODEProblemNames = string.(parameters(odeSystem))
    modelStateNames = replace.(string.(states(odeSystem)), "(t)" => "")

    stringWriteCallbacks = "function getCallbacks_" * modelName * "()\n"
    stringWriteTstops = "\nfunction computeTstops(u::AbstractVector, p::AbstractVector)\n"

    # In case we do not have any events
    verbose && @info "Building callbacks"
    if isempty(SBMLDict["boolVariables"]) && isempty(SBMLDict["events"])
        callbackNames = ""
        checkIfActivatedT0Names = ""
        stringWriteTstops *= "\t return Float64[]\nend\n"
    else
        for key in keys(SBMLDict["boolVariables"])
            functionsStr, callbackStr =  createCallback(key, SBMLDict, pODEProblemNames, modelStateNames)
            stringWriteCallbacks *= functionsStr * "\n"
            stringWriteCallbacks *= callbackStr * "\n"
        end
        for key in keys(SBMLDict["events"])
            functionsStr, callbackStr = createCallbackForEvent(key, SBMLDict, pODEProblemNames, modelStateNames)
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
    convertTspan = false
    stringWriteCallbacks *= "\treturn CallbackSet(" * callbackNames * "), Function[" * checkIfActivatedT0Names * "], " * string(convertTspan)  * "\nend"
    fileWrite = dirSave * "/" * modelName * "_callbacks.jl"
    if isfile(fileWrite)
        rm(fileWrite)
    end
    io = open(fileWrite, "w")
    write(io, stringWriteCallbacks * "\n\n")
    write(io, stringWriteTstops)
    close(io)

    strGetCallbacks = getFunctionsAsString(fileWrite, 2)
    getCallbackFunction = @RuntimeGeneratedFunction(Meta.parse(strGetCallbacks[1]))
    cbSet, checkCbActive, convertTspan = getCallbackFunction("https://xkcd.com/2694/") # Argument needed by @RuntimeGeneratedFunction
    computeTstops = @RuntimeGeneratedFunction(Meta.parse(strGetCallbacks[2]))

    verbose && @info "Solving ODE"

    odeProblem = ODEProblem(odeSystem, stateMap, timeSpan, parameterMap, jac=true)
    tStops = computeTstops(odeProblem.u0, odeProblem.p)
    for f! in checkCbActive
        f!(odeProblem.u0, odeProblem.p)
    end

    return solve(odeProblem, solver, abstol=abstol, reltol=reltol, saveat=saveat, tstops=tStops, callback=cbSet)
end