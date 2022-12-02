# TODO: Refactor code and document functions. Check SBMLToolkit if can be used. 


using PyCall, DataFrames, CSV


include(pwd() * "/src/SBML/Common.jl")
include(pwd() * "/src/SBML/Process_functions.jl")
include(pwd() * "/src/SBML/Process_rules.jl")


"""
    XmlToModellingToolkit(pathXml::String, modelName::String, dirModel::String)

    Convert a SBML file in pathXml to a Julia ModelingToolkit file and store 
    the resulting file in dirModel with name modelName.jl. 

    The SBML importer goes via libsbml in Python and currently likelly only 
    works with SBML level 3. 
"""
function XmlToModellingToolkit(pathXml::String, modelName::String, dirModel::String; writeToFile=true::Bool)

    libsbml = pyimport("libsbml")
    reader = libsbml.SBMLReader()

    document = reader[:readSBML](pathXml)
    model = document[:getModel]() # Get the model
    modelDict = buildODEModelDictionary(libsbml, model)

    if writeToFile
        writeODEModelToFile(modelDict, modelName, dirModel)
    end

    return modelDict

end


# Rewrites triggers in events to propper form for ModelingToolkit
function asTrigger(triggerFormula)
    if "geq" == triggerFormula[1:3]
        strippedFormula = triggerFormula[5:end-1]
    elseif "gt" == triggerFormula[1:2]
        strippedFormula = triggerFormula[4:end-1]
    elseif "leq" == triggerFormula[1:3]
        strippedFormula = triggerFormula[5:end-1]
    elseif "lt" == triggerFormula[1:2]
        strippedFormula = triggerFormula[4:end-1]
    end
    parts = splitBetween(strippedFormula, ',')
    if occursin("time", parts[1])
        parts[1] = replaceWholeWord(parts[1], "time", "t")
    end
    expression = "[" * parts[1] * " ~ " * parts[2] * "]"
    return expression
end


# Rewrites derivatives for a model by replacing functions, any lagging piecewise, and power functions.
function rewriteDerivatives(derivativeAsString, modelDict, baseFunctions)
    newDerivativeAsString = derivativeAsString
    newDerivativeAsString = replaceFunctionWithFormula(newDerivativeAsString, modelDict["modelFunctions"])
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


function processInitialAssignment(libsbml, model, modelDict::Dict, baseFunctions::Array{String, 1})

    initallyAssignedVariable = Dict{String, String}()
    initallyAssignedParameter = Dict{String, String}()
    for initAssign in model[:getListOfInitialAssignments]()
        
        assignName = initAssign[:getId]()
        assignMath = initAssign[:getMath]()
        assignFormula = libsbml[:formulaToString](assignMath)
        assignFormula = rewriteDerivatives(assignFormula, modelDict, baseFunctions)
        
        # Figure out wheter parameters or state is affected by the initial assignment  
        if assignName in keys(modelDict["states"])
            modelDict["states"][assignName] = assignFormula
            initallyAssignedVariable[assignName] = "states"

        elseif assignName in keys(modelDict["nonConstantParameters"])
            modelDict["nonConstantParameters"][assignName] = assignFormula
            initallyAssignedVariable[assignName] = "nonConstantParameters"

        elseif assignName in keys(modelDict["parameters"])
            modelDict["parameters"][assignName] = assignFormula
            initallyAssignedParameter[assignName] = "parameters"
        else
            println("Error: could not find assigned variable/parameter")
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


function buildODEModelDictionary(libsbml, model)

    # Nested dictionaries to store relevant model data:
    # i) Model parameters (constant during for a simulation)
    # ii) Model parameters that are nonConstant (e.g due to events) during a simulation
    # iii) Model states 
    # iv) Model function (functions in the SBML file we rewrite to Julia syntax)
    # v) Model rules (rules defined in the SBML model we rewrite to Julia syntax)
    # vi) Model derivatives (derivatives defined by the SBML model)
    modelDict = Dict()    
    modelDict["states"] = Dict()
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
    # Mathemathical base functions (can be expanded if needed)
    baseFunctions = ["exp", "log", "log2", "log10", "sin", "cos", "tan", "pi"]
    stringOfEvents = ""
    discreteEventString = ""

    # Extract model states, their initial values and set up derivative expression for each state
    for spec in model[:getListOfSpecies]()
        stateId = spec[:getId]()
        # If initial amount is zero (default) check if intial value is given in getInitialConcentration
        if spec[:getInitialAmount]() == 0
            if spec[:getInitialConcentration]() === NaN # Default SBML value
                modelDict["states"][stateId] = string(spec[:getInitialAmount]())
            else
                modelDict["states"][stateId] = string(spec[:getInitialConcentration]())
            end

        # Else use the value provided in getInitialAmount
        else
            modelDict["states"][stateId] = string(spec[:getInitialAmount]())
        end

        modelDict["derivatives"][stateId] = "D(" * stateId * ") ~ " # ModellingToolkitSyntax
    end

    # Extract model parameters and their default values
    for parameter in model[:getListOfParameters]()
        modelDict["parameters"][parameter[:getId]()] = string(parameter[:getValue]())
    end

    # Extract model compartments and store their volumes along with the model parameters 
    for compartment in model[:getListOfCompartments]()
        modelDict["parameters"][compartment[:getId]()] = string(compartment[:getSize]())
    end

    # Rewrite SBML functions into Julia syntax functions and store in dictionary to allow them to 
    # be inserted into equation formulas downstream.
    for functionDefinition in model[:getListOfFunctionDefinitions]()
        math = functionDefinition[:getMath]()
        functionName = functionDefinition[:getId]()    
        args = getSBMLFuncArg(math)
        functionFormula = getSBMLFuncFormula(math, libsbml)
        modelDict["modelFunctions"][functionName] = [args[2:end-1], functionFormula] # (args, formula)
    end

    ### Define events
    stringOfEvents = ""
    for (eIndex, event) in enumerate(model[:getListOfEvents]())

        println("Model has an event :o :o")

        eventName = event[:getName]()
        trigger = event[:getTrigger]()
        triggerMath = trigger[:getMath]()
        triggerFormula = asTrigger(libsbml[:formulaToString](triggerMath))
        eventAsString = ""
        for (eaIndex, eventAssignment) in enumerate(event[:getListOfEventAssignments]())
            variableName = eventAssignment[:getVariable]()
            # if the variable in the event is not set as a variable, make it so and remove it as a parameter or constant
            if variableName in keys(parameterDict)
                modelDict["nonConstantParameters"][variableName] = parameterDict[variableName]
                delete!(parameterDict, variableName)
            end

            eventMath = eventAssignment[:getMath]()
            eventMathAsString = libsbml[:formulaToString](eventMath)

            # Add the event 
            if eaIndex == 1
                eventAsString = "[" * variableName * " ~ " * eventMathAsString
            else
                eventAsString = eventAsString * ", " * variableName * " ~ " * eventMathAsString
            end
        end
        eventAsString = eventAsString * "]"
        fullEvent = triggerFormula * " => " * eventAsString
        if eIndex == 1
            stringOfEvents = fullEvent
        else
            stringOfEvents = stringOfEvents * ", " * fullEvent
        end
    end

    # Extract model rules. Each rule-type is processed differently.
    for rule in model[:getListOfRules]()
        println("RuleType = ", rule[:getElementName]())
        ruleType = rule[:getElementName]() 
        
        if ruleType == "assignmentRule"
            ruleVariable = rule[:getVariable]() 
            ruleFormula = getRuleFormula(rule)
            processAssignmentRule!(modelDict, ruleFormula, ruleVariable, baseFunctions)

        elseif ruleType == "algebraicRule"
            # TODO
            println("Currently we do not support algebraic rules :(")

        elseif ruleType == "rateRule"  
            ruleVariable = rule[:getVariable]() # variable
            ruleFormula = getRuleFormula(rule)
            processRateRule!(modelDict, ruleFormula, ruleVariable, baseFunctions)
        end
    end

    # Positioned after rules since some assignments may include functions
    processInitialAssignment(libsbml, model, modelDict, baseFunctions)

    # Process reactions into Julia functions, and keep correct Stoichiometry
    reactions = [(r, r[:getKineticLaw]()[:getFormula]()) for r in model[:getListOfReactions]()]
    for (reac, formula) in reactions
        products = [(p[:species], p[:getStoichiometry]()) for p in reac[:getListOfProducts]()]
        reactants = [(r[:species], r[:getStoichiometry]()) for r in reac[:getListOfReactants]()]
        formula = rewriteDerivatives(formula, modelDict, baseFunctions)
        for (rName, rStoich) in reactants
            rComp = model[:getSpecies](rName)[:getCompartment]()
            modelDict["derivatives"][rName] = modelDict["derivatives"][rName] * "-" * string(rStoich) * " * ( 1 /" * rComp * " ) * (" * formula * ")"
        end
        for (pName, pStoich) in products
            pComp = model[:getSpecies](pName)[:getCompartment]()
            modelDict["derivatives"][pName] = modelDict["derivatives"][pName] * "+" * string(pStoich) * " * ( 1 /" * pComp * " ) * (" * formula * ")"
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

    modelDict["stringOfEvents"] = stringOfEvents
    modelDict["discreteEventString"] = discreteEventString
    modelDict["numOfParameters"] =   string(length(model[:getListOfParameters]()))
    modelDict["numOfSpecies"] =   string(length(model[:getListOfSpecies]()))
    
    return modelDict
end

"""
    writeODEModelToFile(modelDict, modelName, dirModel)

    Takes a modelDict as defined by buildODEModelDictionary
    and creates a Julia ModelingToolkit file and stores 
    the resulting file in dirModel with name modelName.jl. 

"""
function writeODEModelToFile(modelDict, modelName, dirModel)
    ### Writing to file 
    modelFile = open(dirModel * "/" * modelName * ".jl", "w")

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
