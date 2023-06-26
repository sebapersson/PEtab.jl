# TODO: Refactor code and document functions. Check SBMLToolkit if can be used.



"""
    XmlToModellingToolkit(pathXml::String, modelName::String, dirModel::String)
    Convert a SBML file in pathXml to a Julia ModelingToolkit file and store
    the resulting file in dirModel with name modelName.jl.
"""
function XmlToModellingToolkit(pathXml::String, pathJlFile::AbstractString, modelName::AbstractString; writeToFile::Bool=true, ifElseToEvent::Bool=true)

    modelSBML = readSBML(pathXml)
    modelDict = buildODEModelDictionary(modelSBML, ifElseToEvent)

    if writeToFile
        writeODEModelToFile(modelDict, pathJlFile, modelName)
    end

    return modelDict
end


"""
    JLToModellingToolkit(jlFilePath::String, dirJulia::String, modelName::String; ifElseToEvent::Bool=true)
    Loads the Julia ModelingToolkit file located in jlFilePath.
    If the file contains ifelse statements and ifElseToEvent=true
    a fixed file will be stored in the Julia_model_files folder
    with the suffix _fix in its filename.
"""
function JLToModellingToolkit(jlFilePath::String, dirJulia::String, modelName::String; ifElseToEvent::Bool=true)

    # Some parts of the modelDict are needed to create the other julia files for the model.
    modelDict = Dict()
    modelDict["boolVariables"] = Dict()
    modelDict["modelRuleFunctions"] = Dict()
    modelDict["inputFunctions"] = Dict()
    modelDict["parameters"] = Dict()
    modelDict["states"] = Dict()

    # Read modelFile to work with it
    odefun = include(jlFilePath)
    expr = Expr(:call, Symbol(odefun))
    odeSys, stateMap, paramMap = eval(expr)

    for eq in odeSys.eqs
        key = replace(string(eq.lhs),"(t)" => "")
        if !occursin("Differential", key)
            modelDict["inputFunctions"][key] = key * "~" * string(eq.rhs)
        end
    end

    for par in paramMap
        modelDict["parameters"][string(par.first)] = string(par.second)
    end

    for stat in stateMap
        modelDict["states"][string(stat.first)] = string(stat.second)
    end

    #Initialize output model file path to input path
    modelFileJl = jlFilePath

    if ifElseToEvent == true
        # Rewrite any time-dependent ifelse to boolean statements such that we can express these as events.
        # This is recomended, as it often increases the stabillity when solving the ODE, and decreases run-time
        timeDependentIfElseToBool!(modelDict)
        if length(modelDict["boolVariables"]) > 0
            # changes final .jl in path to _fix.jl
            # and changes output model file path to the fixed one.
            fileName = splitpath(jlFilePath)[end]
            fileNameFix = replace(fileName, Regex(".jl\$") => "_fix.jl")
            modelFileJl = joinpath(dirJulia, fileNameFix)
            # Create a new "fixed" julia file
            io = open(modelFileJl, "w")
            println(io, "# Model name: " * modelName)
            println(io, "# Number of parameters: " * string(length(paramMap)))
            println(io, "# Number of species: " * string(length(stateMap)))
            println(io, "function getODEModel_" * modelName * "()")
            println(io, "")

            tmpLine = "    ModelingToolkit.@variables t "
            tmpLineArray = "    stateArray = ["
            for key in keys(modelDict["states"])
                tmpLine *= key * " "
                tmpLineArray *= replace(key,"(t)"=>"") * ", "
            end
            tmpLineArray = tmpLineArray[1:end-2] * "]"

            println(io, "    ### Define independent and dependent variables")
            println(io, tmpLine)
            println(io, "")
            println(io, "    ### Store dependent variables in array for ODESystem command")
            println(io, tmpLineArray)
            println(io, "")
            println(io, "    ### Define variable parameters")
            println(io, "")
            println(io, "    ### Define potential algebraic variables")

            if length(modelDict["inputFunctions"]) > 0
                tmpLine = "    ModelingToolkit.@variables "
                for key in keys(modelDict["inputFunctions"])
                    tmpLine *= key * "(t) "
                end
                println(io, tmpLine)
            end

            tmpLine = "    ModelingToolkit.@parameters "
            tmpLineArray = "    parameterArray = ["
            for key in keys(modelDict["parameters"])
                tmpLine *= key * " "
                tmpLineArray *= replace(key,"(t)"=>"") * ", "
            end
            tmpLineArray = tmpLineArray[1:end-2] * "]"

            println(io, "")
            println(io, "    ### Define parameters")
            println(io, tmpLine)
            println(io, "")
            println(io, "    ### Store parameters in array for ODESystem command")
            println(io, tmpLineArray)
            println(io, "")
            println(io, "    ### Define an operator for the differentiation w.r.t. time")
            println(io, "    D = Differential(t)")
            println(io, "")
            println(io, "    ### Continious events ###")
            println(io, "")
            println(io, "    ### Discrete events ###")

            equationList = string.(equations(odeSys))
            equationList = replace.(equationList, "Differential" => "D")
            equationList = replace.(equationList, "(t)" => "")

            tmpLine = "    eqs = [\n"

            for eq in equationList
                # Fixes equations containing ifelse
                if occursin("ifelse", eq)
                    # The equation starts either with "D(key) ~" or just "key ~"
                    tildePos = findfirst("~",eq)[1]
                    key = eq[1:tildePos-1]
                    key = replace(key,"D(" => "")
                    key = replace(key,")" => "")
                    key = replace(key," " => "")
                    tmpLine *= "    " * modelDict["inputFunctions"][key] * ",\n"
                else
                    tmpLine *= "    " * eq * ",\n"
                end
            end
            tmpLine = tmpLine[1:end-2] * "\n    ]"

            println(io, "")
            println(io, "    ### Derivatives ###")
            println(io, tmpLine)
            println(io, "    @named sys = ODESystem(eqs, t, stateArray, parameterArray)")
            println(io, "")
            println(io, "    ### Initial species concentrations ###")
            tmpLineArray = "    initialSpeciesValues = [\n"
            for stat in stateMap
                statN = replace(string(stat.first),"(t)"=>"")
                statV = string(stat.second)
                tmpLineArray *= "        " * statN * " => " * statV * ", \n"
            end
            tmpLineArray = tmpLineArray[1:end-3] * "\n    ]"
            println(io, tmpLineArray)

            println(io, "")
            println(io, "    ### SBML file parameter values ###")
            parameterList = string.(paramMap)
            tmpLineArray = "    trueParameterValues = [\n"
            for par in parameterList
                tmpLineArray *= "        " * par * ", \n"
            end

            # Adds boolVariables to trueParameterValues
            for par in keys(modelDict["boolVariables"])
                tmpLineArray *= "        " * par * " => 0.0, \n"
            end
            tmpLineArray = tmpLineArray[1:end-3] * "\n    ]"
            println(io, tmpLineArray)

            println(io, "")
            println(io, "    return sys, initialSpeciesValues, trueParameterValues")

            println(io, "")
            println(io, "end")
            close(io)
        end
    end

    return modelDict, modelFileJl

end


# Rewrites triggers in events to propper form for ModelingToolkit
function asTrigger(triggerFormula)
    if "geq" == triggerFormula[1:3]
        strippedFormula = triggerFormula[5:end-1]
        separatorUse = "≥"
    elseif "gt" == triggerFormula[1:2]
        strippedFormula = triggerFormula[4:end-1]
        separatorUse = "≥"
    elseif "leq" == triggerFormula[1:3]
        strippedFormula = triggerFormula[5:end-1]
        separatorUse = "≤"
    elseif "lt" == triggerFormula[1:2]
        strippedFormula = triggerFormula[4:end-1]
        separatorUse = "≤"
    end
    parts = splitBetween(strippedFormula, ',')
    if occursin("time", parts[1])
        parts[1] = replaceWholeWord(parts[1], "time", "t")
    end
    expression = parts[1] * " " * separatorUse * " " * parts[2] 
    return expression
end


# Rewrites derivatives for a model by replacing functions, any lagging piecewise, and power functions.
function rewriteDerivatives(derivativeAsString, modelDict, baseFunctions)
    
    newDerivativeAsString = replaceFunctionWithFormula(derivativeAsString, modelDict["modelFunctions"])
    newDerivativeAsString = replaceFunctionWithFormula(newDerivativeAsString, modelDict["modelRuleFunctions"])
    
    if occursin("pow(", newDerivativeAsString)
        newDerivativeAsString = removePowFunctions(newDerivativeAsString)
    end
    if occursin("piecewise(", newDerivativeAsString)
        newDerivativeAsString = rewritePiecewiseToIfElse(newDerivativeAsString, "foo", modelDict, baseFunctions, retFormula=true)
    end

    newDerivativeAsString = replaceWholeWordDict(newDerivativeAsString, modelDict["modelFunctions"])
    newDerivativeAsString = replaceWholeWordDict(newDerivativeAsString, modelDict["modelRuleFunctions"])

    return newDerivativeAsString
end


function processInitialAssignment(modelSBML, modelDict::Dict, baseFunctions::Array{String, 1})

    initallyAssignedVariable = Dict{String, String}()
    initallyAssignedParameter = Dict{String, String}()
    for (assignId, initialAssignment) in modelSBML.initial_assignments
        
        _formula = mathToString(initialAssignment)
        formula = rewriteDerivatives(_formula, modelDict, baseFunctions)

        # Figure out wheter parameters or state is affected by the initial assignment
        if assignId ∈ keys(modelDict["states"])
            modelDict["states"][assignId] = formula
            initallyAssignedVariable[assignId] = "states"

        elseif assignId ∈ keys(modelDict["nonConstantParameters"])
            modelDict["nonConstantParameters"][assignId] = formula
            initallyAssignedVariable[assignId] = "nonConstantParameters"

        elseif assignId ∈ keys(modelDict["parameters"])
            modelDict["parameters"][assignId] = formula
            initallyAssignedVariable[assignId] = "parameters"

        else
            @error "Could not identify assigned variable $assignId in list of states or parameters"
        end
    end

    # If the initial assignment for a state is the value of another state apply recursion until continue looping
    # until we have the initial assignment expressed as non state variables
    while true
        nestedVariables = false
        for (variable, dictName) in initallyAssignedVariable
            if dictName == "states"
                variableValue = modelDict["states"][variable]
                args = split(getArguments(variableValue, baseFunctions))
                for arg in args
                    if arg in keys(modelDict["states"])
                        nestedVariables = true
                        variableValue = replaceWholeWord(variableValue, arg, modelDict["states"][arg])
                    end
                end
                modelDict["states"][variable] = variableValue
            end
        end
        nestedVariables || break
    end

    # If the initial assignment for a parameters is the value of another parameters apply recursion
    # until we have the initial assignment expressed as non parameters
    while true
        nestedParameter = false
        for (parameter, dictName) in initallyAssignedParameter
            parameterValue = modelDict["parameters"][parameter]
            args = split(getArguments(parameterValue, baseFunctions))
            for arg in args
                if arg in keys(modelDict["parameters"])
                    nestedParameter = true
                    parameterValue = replaceWholeWord(parameterValue, arg, modelDict["parameters"][arg])
                end
            end
            modelDict["parameters"][parameter] = parameterValue
        end
        nestedParameter || break
    end

end


function buildODEModelDictionary(modelSBML, ifElseToEvent::Bool)

    # Nested dictionaries to store relevant model data:
    # i) Model parameters (constant during for a simulation)
    # ii) Model parameters that are nonConstant (e.g due to events) during a simulation
    # iii) Model states
    # iv) Model function (functions in the SBML file we rewrite to Julia syntax)
    # v) Model rules (rules defined in the SBML model we rewrite to Julia syntax)
    # vi) Model derivatives (derivatives defined by the SBML model)
    modelDict = Dict()
    modelDict["states"] = Dict()
    modelDict["hasOnlySubstanceUnits"] = Dict()
    modelDict["isBoundaryCondition"] = Dict()
    modelDict["parameters"] = Dict()
    modelDict["nonConstantParameters"] = Dict()
    modelDict["modelFunctions"] = Dict()
    modelDict["modelRuleFunctions"] = Dict()
    modelDict["modelRules"] = Dict()
    modelDict["derivatives"] = Dict()
    modelDict["eventDict"] = Dict()
    modelDict["discreteEventDict"] = Dict()
    modelDict["inputFunctions"] = Dict()
    modelDict["stringOfEvents"] = Dict()
    modelDict["discreteEventString"] = Dict()
    modelDict["numOfParameters"] = Dict()
    modelDict["numOfSpecies"] = Dict()
    modelDict["boolVariables"] = Dict()
    modelDict["events"] = Dict()
    modelDict["reactions"] = Dict()
    # Mathemathical base functions (can be expanded if needed)
    baseFunctions = ["exp", "log", "log2", "log10", "sin", "cos", "tan", "pi"]
    stringOfEvents = ""
    discreteEventString = ""

    for (stateId, state) in modelSBML.species
        # If initial amount is zero or nothing (default) should use initial-concentration if non-empty 
        if (state.initial_amount == 0 || isnothing(state.initial_amount)) && isnothing(state.initial_concentration)
            modelDict["states"][stateId] = "0.0"
        elseif !isnothing(state.initial_concentration)
            modelDict["states"][stateId] = string(state.initial_concentration)
        else 
            modelDict["states"][stateId] = string(state.initial_amount)
        end

        # Setup for downstream processing 
        modelDict["hasOnlySubstanceUnits"][stateId] = state.only_substance_units
        modelDict["isBoundaryCondition"][stateId] = state.boundary_condition 
        modelDict["derivatives"][stateId] = "D(" * stateId * ") ~ " # ModellingToolkitSyntax

        # In case being a boundary condition the state can only be changed by the user 
        if modelDict["isBoundaryCondition"][stateId] == true
           modelDict["derivatives"][stateId] *= "0.0"
        end
    end

    # Extract model parameters and their default values
    for (parameterId, parameter) in modelSBML.parameters
        modelDict["parameters"][parameterId] = string(parameter.value)
    end
    for (compartmentId, compartment) in modelSBML.compartments
        modelDict["parameters"][compartmentId] = string(compartment.size)
    end

    # Rewrite SBML functions into Julia syntax functions and store in dictionary to allow them to
    # be inserted into equation formulas downstream
    for (functionName, SBMLFunction) in modelSBML.function_definitions
        args = getSBMLFunctionArgs(SBMLFunction)
        functionFormula = mathToString(SBMLFunction.body.body)
        modelDict["modelFunctions"][functionName] = [args, functionFormula]
    end

    # Later by the process callback function these events are rewritten to 
    # DiscreteCallback:s if possible 
    eIndex = 1
    for (eventName, event) in modelSBML.events
        _triggerFormula = mathToString(event.trigger.math)
        triggerFormula = asTrigger(_triggerFormula)
        eventFormulas = Vector{String}(undef, length(event.event_assignments))
        eventAssignTo = similar(eventFormulas)
        for (i, eventAssignment) in pairs(event.event_assignments)
            eventFormulas[i] = mathToString(eventAssignment.math)
            eventAssignTo[i] = eventAssignment.variable
        end
        eventName = isempty(eventName) ? "event" * string(eIndex) : eventName
        modelDict["events"][eventName] = [triggerFormula, eventAssignTo .* " = " .* eventFormulas]
        eIndex += 1
    end

    for rule in modelSBML.rules
        if rule isa SBML.AssignmentRule
            ruleFormula = extractRuleFormula(rule)
            processAssignmentRule!(modelDict, ruleFormula, rule.variable, baseFunctions)
        end

        if rule isa SBML.RateRule
            ruleFormula = extractRuleFormula(rule)
            processRateRule!(modelDict, ruleFormula, rule.variable, baseFunctions)
        end

        if rule isa SBML.AlgebraicRule
            @error "Currently we do not support algebraic rules"
        end
    end

    # Positioned after rules since some assignments may include functions
    processInitialAssignment(modelSBML, modelDict, baseFunctions)

    # Process chemical reactions 
    for (id, reaction) in modelSBML.reactions
        # Process kinetic math into Julia syntax 
        _formula = mathToString(reaction.kinetic_math)
               
        # Add values for potential kinetic parameters (where-statements)
        for (parameterId, parameter) in reaction.kinetic_parameters
            _formula = replaceWholeWord(_formula, parameterId, parameter.value)
        end

        formula = rewriteDerivatives(_formula, modelDict, baseFunctions)
        modelDict["reactions"][reaction.name] = formula
        
        for reactant in reaction.reactants
            modelDict["isBoundaryCondition"][reactant.species] == true && continue # Constant state  
            compartment = modelSBML.species[reactant.species].compartment
            stoichiometry = isnothing(reactant.stoichiometry) ? "1" : string(reactant.stoichiometry)
            compartmentScaling = modelDict["hasOnlySubstanceUnits"][reactant.species] == true ? " * " : " * ( 1 /" * compartment * " ) * "
            modelDict["derivatives"][reactant.species] *= "-" * stoichiometry * compartmentScaling * "(" * formula * ")"
        end
        for product in reaction.products
            modelDict["isBoundaryCondition"][product.species] == true && continue # Constant state  
            compartment = modelSBML.species[product.species].compartment
            stoichiometry = isnothing(product.stoichiometry) ? "1" : string(product.stoichiometry)
            compartmentScaling = modelDict["hasOnlySubstanceUnits"][product.species] == true ? " * " : " * ( 1 /" * compartment * " ) * "
            modelDict["derivatives"][product.species] *= "+" * stoichiometry * compartmentScaling * "(" * formula * ")"
        end
    end

    # Check which parameters are a part derivatives or input function. If a parameter is not a part, e.g is an initial
    # assignment parameters, add to dummy variable to keep it from being simplified away.
    isInODESys = falses(length(modelDict["parameters"]))
    for du in values(modelDict["derivatives"])
        for (i, pars) in enumerate(keys(modelDict["parameters"]))
            if replaceWholeWord(du, pars, "") !== du
                isInODESys[i] = true
            end
        end
    end
    for inputFunc in values(modelDict["inputFunctions"])
        for (i, pars) in enumerate(keys(modelDict["parameters"]))
            if replaceWholeWord(inputFunc, pars, "") !== inputFunc
                isInODESys[i] = true
            end
        end
    end

    # Rewrite any time-dependent ifelse to boolean statements such that we can express these as events.
    # This is recomended, as it often increases the stabillity when solving the ODE, and decreases run-time
    if ifElseToEvent == true
        timeDependentIfElseToBool!(modelDict)
    end

    modelDict["stringOfEvents"] = stringOfEvents
    modelDict["discreteEventString"] = discreteEventString
    modelDict["numOfParameters"] = string(length(keys(modelDict["parameters"])))
    modelDict["numOfSpecies"] = string(length(keys(modelDict["states"])))
    return modelDict
end


"""
    writeODEModelToFile(modelDict, modelName, dirModel)
    Takes a modelDict as defined by buildODEModelDictionary
    and creates a Julia ModelingToolkit file and stores
    the resulting file in dirModel with name modelName.jl.
"""
function writeODEModelToFile(modelDict, pathJlFile, modelName)
    ### Writing to file
    modelFile = open(pathJlFile, "w")

    println(modelFile, "# Model name: " * modelName)

    println(modelFile, "# Number of parameters: " * string(modelDict["numOfParameters"]))
    println(modelFile, "# Number of species: " * string(modelDict["numOfSpecies"]))
    println(modelFile, "function getODEModel_" * modelName * "()")

    println(modelFile, "")
    println(modelFile, "    ### Define independent and dependent variables")
    defineVariables = "    ModelingToolkit.@variables t"
    for key in keys(modelDict["states"])
        defineVariables = defineVariables * " " * key * "(t)"
    end
    println(modelFile, defineVariables)

    println(modelFile, "")
    println(modelFile, "    ### Store dependent variables in array for ODESystem command")
    defineVariables = "    stateArray = ["
    for (index, key) in enumerate(keys(modelDict["states"]))
        if index < length(modelDict["states"])
            defineVariables = defineVariables * key * ", "
        else
            defineVariables = defineVariables * key * "]"
        end
    end
    println(modelFile, defineVariables)

    println(modelFile, "")
    println(modelFile, "    ### Define variable parameters")
    if length(modelDict["nonConstantParameters"]) > 0
        defineVariableParameters = "    ModelingToolkit.@variables"
        for key in keys(modelDict["nonConstantParameters"])
            defineVariableParameters = defineVariableParameters * " " * key * "(t)"
        end
        println(modelFile, defineVariableParameters)
    end

    println(modelFile, "")
    println(modelFile, "    ### Define potential algebraic variables")
    if length(modelDict["inputFunctions"]) > 0
        defineVariableParameters = "    ModelingToolkit.@variables"
        for key in keys(modelDict["inputFunctions"])
            defineVariableParameters = defineVariableParameters * " " * key * "(t)"
        end
        println(modelFile, defineVariableParameters)
    end

    println(modelFile, "")
    println(modelFile, "    ### Define parameters")
    defineParameters = "    ModelingToolkit.@parameters"
    for key in keys(modelDict["parameters"])
        defineParameters = defineParameters * " " * key
    end
    println(modelFile, defineParameters)

    println(modelFile, "")
    println(modelFile, "    ### Store parameters in array for ODESystem command")
    defineParameters = "    parameterArray = ["
    for (index, key) in enumerate(keys(modelDict["parameters"]))
        if index < length(modelDict["parameters"])
           defineParameters = defineParameters * key * ", "
        else
           defineParameters = defineParameters * key * "]"
        end
    end
    println(modelFile, defineParameters)

    println(modelFile, "")
    println(modelFile, "    ### Define an operator for the differentiation w.r.t. time")
    println(modelFile, "    D = Differential(t)")

    stringOfEvents = modelDict["stringOfEvents"]
    println(modelFile, "")
    println(modelFile, "    ### Continious events ###")
    if length(stringOfEvents) > 0
        println(modelFile, "    continuous_events = [")
        println(modelFile, "    " * stringOfEvents)
        println(modelFile, "    ]")
    end

    discreteEventString = modelDict["discreteEventString"]
    println(modelFile, "")
    println(modelFile, "    ### Discrete events ###")
    if length(discreteEventString) > 0
        println(modelFile, "    discrete_events = [")
        println(modelFile, "    " * discreteEventString)
        println(modelFile, "    ]")
    end

    println(modelFile, "")
    println(modelFile, "    ### Derivatives ###")
    println(modelFile, "    eqs = [")
    for (sIndex, key) in enumerate(keys(modelDict["states"]))
        # If the state is not part of any reaction we set its value to zero.
        if occursin(Regex("~\\s*\$"),modelDict["derivatives"][key])
            modelDict["derivatives"][key] *= "0.0"
        end
        if sIndex == 1
            print(modelFile, "    " * modelDict["derivatives"][key])
        else
            print(modelFile, ",\n    " * modelDict["derivatives"][key])
        end
    end
    for key in keys(modelDict["nonConstantParameters"])
        print(modelFile, ",\n    D(" * key * ") ~ 0")
    end
    for key in keys(modelDict["inputFunctions"])
        print(modelFile, ",\n    " * modelDict["inputFunctions"][key])
    end
    println(modelFile, "")
    println(modelFile, "    ]")

    println(modelFile, "")
    if length(stringOfEvents) > 0 && length(discreteEventString) > 0
        println(modelFile, "    @named sys = ODESystem(eqs, t, stateArray, parameterArray, continuous_events = continuous_events, discrete_events = discrete_events)")
    elseif length(stringOfEvents) > 0 && length(discreteEventString) == 0
        println(modelFile, "    @named sys = ODESystem(eqs, t, stateArray, parameterArray, continuous_events = continuous_events)")
    elseif length(stringOfEvents) == 0 && length(discreteEventString) > 0
        println(modelFile, "    @named sys = ODESystem(eqs, t, stateArray, parameterArray, discrete_events = discrete_events)")
    else
        println(modelFile, "    @named sys = ODESystem(eqs, t, stateArray, parameterArray)")
    end

    println(modelFile, "")
    println(modelFile, "    ### Initial species concentrations ###")
    println(modelFile, "    initialSpeciesValues = [")
    for (index, (key, value)) in enumerate(modelDict["states"])
        if tryparse(Float64,value) !== nothing
            value = string(parse(Float64,value))
        end
        if index == 1
            assignString = "    " * key * " => " * value
        else
            assignString = ",\n    " * key * " => " * value
        end
        print(modelFile, assignString)
    end
    for (key, value) in modelDict["nonConstantParameters"]
        assignString = ",\n    " * key * " => " * value
        print(modelFile, assignString)
    end
    println(modelFile, "")
    println(modelFile, "    ]")

    println(modelFile, "")
    println(modelFile, "    ### SBML file parameter values ###")
    println(modelFile, "    trueParameterValues = [")
    for (index, (key, value)) in enumerate(modelDict["parameters"])
        if tryparse(Float64,value) !== nothing
            value = string(parse(Float64,value))
        end
        if index == 1
            assignString = "    " * key * " => " * value
        else
            assignString = ",\n    " * key * " => " * value
        end
        print(modelFile, assignString)
    end
    println(modelFile, "")
    println(modelFile, "    ]")
    println(modelFile, "")

    println(modelFile, "    return sys, initialSpeciesValues, trueParameterValues")
    println(modelFile, "")
    println(modelFile, "end")

    close(modelFile)
end


function mathToString(math)
    mathStr, _ = _mathToString(math)
    return mathStr
end


function _mathToString(math::SBML.MathApply)

    if math.fn ∈ ["*", "/", "+", "-", "power"] && length(math.args) == 2
        fn = math.fn == "power" ? "^" : math.fn
        _part1, addParenthesis1 = _mathToString(math.args[1])
        _part2, addParenthesis2 = _mathToString(math.args[2])
        # In case we hit the bottom in the recursion we do not need to add paranthesis 
        # around the math-expression making the equations easier to read
        part1 = addParenthesis1 ?  '(' * _part1 * ')' : _part1
        part2 = addParenthesis2 ?  '(' * _part2 * ')' : _part2
        return part1 * fn * part2, true
    end

    if math.fn ∈ ["+", "-"] && length(math.args) == 1
        _formula, addParenthesis = _mathToString(math.args[1])
        formula = addParenthesis ? '(' * _formula * ')' : _formula
        return math.fn * formula, true
    end

    # Piecewise can have arbibrary number of arguments 
    if math.fn == "piecewise"
        formula = "piecewise("
        for arg in math.args
            _formula, _ = _mathToString(arg) 
            formula *= _formula * ", "
        end
        return formula[1:end-2] * ')', false
    end

    if math.fn ∈ ["lt", "gt", "leq", "geq"]
        @assert length(math.args) == 2
        part1, _ = _mathToString(math.args[1]) 
        part2, _ = _mathToString(math.args[2])
        return math.fn * "(" * part1 * ", " * part2 * ')', false
    end

    if math.fn ∈ ["exp", "log", "log2", "log10", "sin", "cos", "tan"]
        @assert length(math.args) == 1
        formula, _ = _mathToString(math.args[1])
        return math.fn * '(' * formula * ')', false
    end

    # At this point the only feasible option left is a SBMLFunction
    formula = math.fn * '('
    for arg in math.args
        _formula, _ = _mathToString(arg) 
        formula *= _formula * ", "
    end
    return formula[1:end-2] * ')', false
end
function _mathToString(math::SBML.MathVal)
    return string(math.val), false
end
function _mathToString(math::SBML.MathIdent)
    return string(math.id), false
end
function _mathToString(math::SBML.MathTime)
    return string(math.id), false
end
