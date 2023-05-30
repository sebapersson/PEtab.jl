# Function generating callbacksets for time-depedent SBML piecewise expressions, as callbacks are more efficient than
# using ifelse (e.g better integration stabillity)
function createCallbacksForTimeDepedentPiecewise(odeSystem::ODESystem,
                                                 parameterMap,
                                                 stateMap,
                                                 SBMLDict::Dict,
                                                 modelName::String,
                                                 pathYAML::String,
                                                 dirJulia::String;
                                                 jlFile::Bool=false,
                                                 customParameterValues::Union{Nothing, Dict}=nothing)

    pODEProblemNames = string.(parameters(odeSystem))
    modelStateNames = replace.(string.(states(odeSystem)), "(t)" => "")

    # Compute indices tracking parameters (needed as down the line we need to know if a parameter should be estimated
    # or not, as if such a parameter triggers a callback we must let it be a continious callback)
    experimentalConditions, measurementsData, parametersData, observablesData = readPEtabFiles(pathYAML, jlFile = jlFile)
    parameterInfo = processParameters(parametersData, customParameterValues=customParameterValues)
    measurementInfo = processMeasurements(measurementsData, observablesData)
    θ_indices = computeIndicesθ(parameterInfo, measurementInfo, odeSystem, parameterMap, stateMap, experimentalConditions)

    # In case of no-callbacks the function for getting callbacks will be empty, likewise for the function
    # which compute tstops (callback-times)
    stringWriteCallbacks = "function getCallbacks_" * modelName * "()\n"
    stringWriteFunctions = ""
    stringWriteTstops = "\nfunction computeTstops(u::AbstractVector, p::AbstractVector)\n"

    # In case we do not have any events
    if isempty(SBMLDict["boolVariables"])
        callbackNames = ""
        checkIfActivatedT0Names = ""
        stringWriteTstops *= "\t return Float64[]\nend\n"
    else
        for key in keys(SBMLDict["boolVariables"])
            functionsStr, callbackStr =  createCallback(key, SBMLDict, pODEProblemNames, modelStateNames)
            stringWriteCallbacks *= functionsStr * "\n"
            stringWriteCallbacks *= callbackStr * "\n"
        end
        callbackNames = prod(["cb_" * key * ", " for key in keys(SBMLDict["boolVariables"])])[1:end-2]
        checkIfActivatedT0Names = prod(["isActiveAtTime0_" * key * "!, " for key in keys(SBMLDict["boolVariables"])])[1:end-2]

        stringWriteTstops *= "\treturn" * createFuncionForTstops(SBMLDict, modelStateNames, pODEProblemNames, θ_indices) * "\n" * "end" * "\n"
    end

    # Check whether or not the trigger for a discrete callback depends on a parameter or not. If true then the time-span
    # must be converted to dual when computing the gradient using ForwardDiff.
    convertTspan = shouldConvertTspan(SBMLDict, odeSystem, θ_indices, jlFile)::Bool

    stringWriteCallbacks *= "\treturn CallbackSet(" * callbackNames * "), Function[" * checkIfActivatedT0Names * "], " * string(convertTspan)  * "\nend"
    fileWrite = dirJulia * "/" * modelName * "_callbacks.jl"
    if isfile(fileWrite)
        rm(fileWrite)
    end
    io = open(fileWrite, "w")
    write(io, stringWriteCallbacks * "\n\n")
    write(io, stringWriteTstops)

    close(io)
end


function createCallback(callbackName::String,
                        SBMLDict::Dict,
                        pODEProblemNames::Vector{String},
                        modelStateNames::Vector{String})

    # Check if the event trigger depend on parameters which are to be i) estimated, or ii) if it depend on models state.
    # For i) it must be a cont. event in order for us to be able to compute the gradient. For ii) we cannot compute
    # tstops (the event times) prior to starting to solve the ODE so it most be cont. callback
    _conditionFormula = SBMLDict["boolVariables"][callbackName][1]
    hasModelStates = conditionHasStates(_conditionFormula, modelStateNames)
    discreteEvent = hasModelStates == true ? false : true

    # Replace any state or parameter with their corresponding index in the ODE system to be comaptible with event
    # syntax
    for i in eachindex(modelStateNames)
        _conditionFormula = replaceWholeWord(_conditionFormula, modelStateNames[i], "u["*string(i)*"]")
    end
    for i in eachindex(pODEProblemNames)
        _conditionFormula = replaceWholeWord(_conditionFormula, pODEProblemNames[i], "integrator.p["*string(i)*"]")
    end

    # Replace inequality with - (root finding cont. event) or with == in case of
    # discrete event
    replaceWith = discreteEvent == true ? "==" : "-"
    conditionFormula = replace(_conditionFormula, "<=" => replaceWith)
    conditionFormula = replace(conditionFormula, ">=" => replaceWith)
    conditionFormula = replace(conditionFormula, ">" => replaceWith)
    conditionFormula = replace(conditionFormula, "<" => replaceWith)

    # Build the condition statement used in the jl function
    conditionStr = "\n\tfunction condition_" * callbackName * "(u, t, integrator)\n"
    conditionStr *= "\t\t" * conditionFormula * "\n\tend\n"

    # Build the affect function
    whichParameter = findfirst(x -> x == callbackName, pODEProblemNames)
    affectStr = "\tfunction affect_" * callbackName * "!(integrator)\n"
    affectStr *= "\t\tintegrator.p[" * string(whichParameter) * "] = 1.0\n\tend\n"

    # Build the callback
    if discreteEvent == false
        callbackStr = "\tcb_" * callbackName * " = ContinuousCallback(" * "condition_" * callbackName * ", " * "affect_" * callbackName * "!, "
    else
        callbackStr = "\tcb_" * callbackName * " = DiscreteCallback(" * "condition_" * callbackName * ", " * "affect_" * callbackName * "!, "
    end
    callbackStr *= "save_positions=(false, false))\n" # So we do not get problems with saveat in the ODE solver

    # Building a function which check if a callback is activated at time zero (as this is not something Julia will
    # check for us)
    sideInequality = SBMLDict["boolVariables"][callbackName][2] == "right" ? "!" : "" # Check if true or false evaluates expression to true
    activeAtT0Str = "\tfunction isActiveAtTime0_" * callbackName * "!(u, p)\n"
    activeAtT0Str *= "\t\tt = 0.0 # Used to check conditions activated at t0=0\n" * "\t\tp[" * string(whichParameter) * "] = 0.0 # Default to being off\n"
    conditionFormula = replace(_conditionFormula, "integrator." => "")
    conditionFormula = replace(conditionFormula, "<=" => "≤")
    conditionFormula = replace(conditionFormula, ">=" => "≥")
    activeAtT0Str *= "\t\tif " * sideInequality *"(" * conditionFormula * ")\n" * "\t\t\tp[" * string(whichParameter) * "] = 1.0\n\t\tend\n\tend\n"

    functionsStr = conditionStr * '\n' * affectStr * '\n' * activeAtT0Str * '\n'

    return functionsStr, callbackStr
end


# Function computing t-stops (time for events) for piecewise expressions using the symbolics package
# to symboically solve for where the condition is zero.
function createFuncionForTstops(SBMLDict::Dict,
                                modelStateNames::Vector{String},
                                pODEProblemNames::Vector{String},
                                θ_indices::ParameterIndices)

    tstopsStr = Vector{String}(undef, length(keys(SBMLDict["boolVariables"])))
    tstopsStrAlt = Vector{String}(undef, length(keys(SBMLDict["boolVariables"])))
    convertTspan = false
    i = 1
    for key in keys(SBMLDict["boolVariables"])

        conditionFormula = SBMLDict["boolVariables"][key][1]
        # In case the activation formula contains a state we cannot precompute the t-stop time as it depends on
        # the actual ODE solution.
        if conditionHasStates(conditionFormula, modelStateNames)
            tstopsStr[i] = ""
            tstopsStrAlt[i] = ""
            i += 1
            continue
        end
        if conditionHasParametersToEstimate(conditionFormula, pODEProblemNames, θ_indices)
            convertTspan = true
        end

        # We need to make the parameters and states symbolic in order to solve the condition expression
        # using the Symbolics package.
        variablesStr = "@variables t, "
        variablesStr *= prod(string.(collect(keys(SBMLDict["parameters"]))) .* ", " )[1:end-2] * " "
        variablesStr *= prod(string.(collect(keys(SBMLDict["states"]))) .* ", " )[1:end-2]
        variablesSymbolic = eval(Meta.parse(variablesStr))

        # Note - below order counts (e.g having < first results in ~= incase what actually stands is <=)
        conditionFormula = replace(conditionFormula, "<=" => "~")
        conditionFormula = replace(conditionFormula, ">=" => "~")
        conditionFormula = replace(conditionFormula, "<" => "~")
        conditionFormula = replace(conditionFormula, ">" => "~")
        conditionSymbolic = eval(Meta.parse(conditionFormula))

        # Expression for the time at which the condition is triggered
        expressionForTime = string.(Symbolics.solve_for(conditionSymbolic, variablesSymbolic[1], simplify=true))

        # Make compatible with the PEtab importer syntax
        for i in eachindex(modelStateNames)
            expressionForTime = replaceWholeWord(expressionForTime, modelStateNames[i], "u["*string(i)*"]")
        end
        for i in eachindex(pODEProblemNames)
            expressionForTime = replaceWholeWord(expressionForTime, pODEProblemNames[i], "p["*string(i)*"]")
        end
        # dualToFloat is needed as tstops for the integrator cannot be of type Dual
        tstopsStr[i] = "dualToFloat(" * expressionForTime * ")"
        tstopsStrAlt[i] = expressionForTime # Used when we convert timespan
        i += 1
    end

    if convertTspan == true
        return"[" * prod([isempty(tstopsStrAlt[i]) ? "" : tstopsStrAlt[i] * ", " for i in eachindex(tstopsStrAlt)])[1:end-2] * "]"
    else
        return " Float64[" * prod([isempty(tstopsStr[i]) ? "" : tstopsStr[i] * ", " for i in eachindex(tstopsStr)])[1:end-2] * "]"
    end
end


function conditionHasStates(conditionFormula::AbstractString, modelStateNames::Vector{String})::Bool
    for i in eachindex(modelStateNames)
        _conditionFormula = replaceWholeWord(conditionFormula, modelStateNames[i], "")
        if _conditionFormula != conditionFormula
            return true
        end
    end
    return false
end


function conditionHasParametersToEstimate(conditionFormula::AbstractString,
                                          pODEProblemNames::Vector{String},
                                          θ_indices::ParameterIndices)::Bool

    # Parameters which are present for each experimental condition, and condition specific parameters
    iODEProblemθConstantDynamic = θ_indices.mapODEProblem.iODEProblemθDynamic
    iODEProblemθDynamicCondition = reduce(vcat, [θ_indices.mapsConiditionId[i].iODEProblemθDynamic for i in keys(θ_indices.mapsConiditionId)])

    for i in eachindex(pODEProblemNames)
        _conditionFormula = replaceWholeWord(conditionFormula, pODEProblemNames[i], "integrator.p["*string(i)*"]")
        if _conditionFormula != conditionFormula
            if i ∈ iODEProblemθConstantDynamic || i ∈ iODEProblemθDynamicCondition
                return true
            end
        end
    end
    return false
end


# Function checking if the condition of the picewise conditions depends on a parameter which is to be estimated.
# In case of true the time-span need to be converted to Dual when computing the gradient (or hessian) of the
# the model via automatic differentitation
function shouldConvertTspan(SBMLDict::Dict, odeSystem::ODESystem, θ_indices, jlFile::Bool)::Bool

    pODEProblemNames = string.(parameters(odeSystem))
    for key in keys(SBMLDict["boolVariables"])

        conditionFormula = SBMLDict["boolVariables"][key][1]
        if conditionHasParametersToEstimate(conditionFormula, pODEProblemNames, θ_indices)
            return true
        end
    end
    return false
end
