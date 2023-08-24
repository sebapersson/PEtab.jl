# TODO: Refactor code and document functions. Check SBMLToolkit if can be used.



"""
    XmlToModellingToolkit(pathXml::String, modelName::String, dirModel::String)

Convert a SBML file in pathXml to a Julia ModelingToolkit file and store
the resulting file in dirModel with name modelName.jl.
"""
function XmlToModellingToolkit(pathXml::String, pathJlFile::AbstractString, modelName::AbstractString; 
                               onlyGetSBMLDict::Bool=false, ifElseToEvent::Bool=true, writeToFile::Bool=true)

    modelSBML = readSBML(pathXml)
    modelDict = buildODEModelDictionary(modelSBML, ifElseToEvent)

    if onlyGetSBMLDict == false
        modelStr = createODEModelFunction(modelDict, pathJlFile, modelName, false, writeToFile)
        return modelDict, modelStr
    end

    return modelDict, ""
end


"""
JLToModellingToolkit(pathJlFile::String, dirJulia::String, modelName::String; ifElseToEvent::Bool=true)
Loads the Julia ModelingToolkit file located in pathJlFile.
If the file contains ifelse statements and ifElseToEvent=true
a fixed file will be stored in the Julia_model_files folder
with the suffix _fix in its filename.
"""
function JLToModellingToolkit(pathJlFile::String, dirJulia::String, modelName::String; 
                              ifElseToEvent::Bool=true, writeToFile::Bool=true)
    
    # Some parts of the modelDict are needed to create the other julia files for the model.
    modelDict = Dict()
    modelDict["boolVariables"] = Dict()
    modelDict["modelRuleFunctions"] = Dict()
    modelDict["inputFunctions"] = Dict()
    modelDict["parameters"] = Dict()
    modelDict["states"] = Dict()
    modelDict["numOfParameters"] = Dict()
    modelDict["numOfSpecies"] = Dict()
    modelDict["equationList"] = Dict()
    modelDict["stateMap"] = Dict()
    modelDict["paramMap"] = Dict()

    # Read modelFile to work with it
    odefun = include(pathJlFile)
    expr = Expr(:call, Symbol(odefun))
    odeSys, stateMap, paramMap = eval(expr)

    modelDict["stateMap"] = stateMap
    modelDict["paramMap"] = paramMap

    # Extract some "metadata"
    modelDict["numOfParameters"] = string(length(paramMap))
    modelDict["numOfSpecies"] = string(length(stateMap))

    equationList = string.(equations(odeSys))
    equationList = replace.(equationList, "Differential" => "D")
    equationList = replace.(equationList, "(t)" => "")
    modelDict["equationList"] = equationList

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
    modelFileJl = pathJlFile
    if ifElseToEvent == true
        # Rewrite any time-dependent ifelse to boolean statements such that we can express these as events.
        # This is recomended, as it often increases the stabillity when solving the ODE, and decreases run-time
        timeDependentIfElseToBool!(modelDict)
        if length(modelDict["boolVariables"]) > 0
            # changes final .jl in path to _fix.jl
            # and changes output model file path to the fixed one.
            fileName = splitpath(pathJlFile)[end]
            fileNameFix = replace(fileName, Regex(".jl\$") => "_fix.jl")
            modelFileJl = joinpath(dirJulia, fileNameFix)
            # Create a new "fixed" julia file
            modelStr = createODEModelFunction(modelDict, pathJlFile, modelName, true, writeToFile)
        else
            modelStr = getFunctionsAsString(modelFileJl, 1)[1]
        end
    else
        modelStr = getFunctionsAsString(modelFileJl, 1)[1]
    end
    
    return modelDict, modelFileJl, modelStr
end



# Rewrites triggers in events to propper form for ModelingToolkit
function asTrigger(triggerFormula, modelDict, modelSBML)

    if triggerFormula[1] == '(' && triggerFormula[end] == ')'
        triggerFormula = triggerFormula[2:end-1]
    end

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

    # States in ODE-system are typically in substance units, but formulas in 
    # concentratio. Thus, each state is divided with its corresponding 
    # compartment 
    for (speciesId, specie) in modelSBML.species
        if speciesId ∈ keys(modelSBML.species) && modelDict["stateGivenInAmounts"][speciesId][1] == false
            continue
        end
        parts[1] = replaceWholeWord(parts[1], speciesId, speciesId * '/' * specie.compartment)
        parts[2] = replaceWholeWord(parts[2], speciesId, speciesId * '/' * specie.compartment)
    end

    return parts[1] * " " * separatorUse * " " * parts[2] 
end


# Rewrites derivatives for a model by replacing functions, any lagging piecewise, and power functions.
function rewriteDerivatives(derivativeAsString, modelDict, baseFunctions, modelSBML; checkScaling=false)
    
    newDerivativeAsString = replaceFunctionWithFormula(derivativeAsString, modelDict["modelFunctions"])
    newDerivativeAsString = replaceFunctionWithFormula(newDerivativeAsString, modelDict["modelRuleFunctions"])
    
    if occursin("pow(", newDerivativeAsString)
        newDerivativeAsString = removePowFunctions(newDerivativeAsString)
    end
    if occursin("piecewise(", newDerivativeAsString)
        newDerivativeAsString = rewritePiecewiseToIfElse(newDerivativeAsString, "foo", modelDict, baseFunctions, modelSBML, retFormula=true)
    end

    newDerivativeAsString = replaceWholeWordDict(newDerivativeAsString, modelDict["modelFunctions"])
    newDerivativeAsString = replaceWholeWordDict(newDerivativeAsString, modelDict["modelRuleFunctions"])

    if checkScaling == false
        return newDerivativeAsString
    end

    # Handle case when specie is given in amount, but the equations are given in concentration 
    for (stateId, state) in modelSBML.species
        if modelDict["stateGivenInAmounts"][stateId][1] == true && modelDict["hasOnlySubstanceUnits"][stateId] == false
            compartment = state.compartment
            newDerivativeAsString = replaceWholeWord(newDerivativeAsString, stateId, "(" * stateId * "/" * compartment * ")")
        end
    end

    return newDerivativeAsString
end


function processInitialAssignment(modelSBML, modelDict::Dict, baseFunctions::Array{String, 1})

    initallyAssignedVariable = Dict{String, String}()
    initallyAssignedParameter = Dict{String, String}()
    for (assignId, initialAssignment) in modelSBML.initial_assignments
        
        _formula = mathToString(initialAssignment)
        formula = rewriteDerivatives(_formula, modelDict, baseFunctions, modelSBML)

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

    # Lastly, if initial assignment refers to a state we need to scale with compartment 
    for id in keys(initallyAssignedVariable)
        if id ∉ keys(modelSBML.species)
            continue
        end
        if isnothing(modelSBML.species[id].substance_units)
            continue
        end
        # We end up here 
        if !(any([val[1] for val in  values(modelDict["stateGivenInAmounts"])]) == true)
            continue
        end
        if modelSBML.species[id].substance_units == "substance"
            modelDict["stateGivenInAmounts"][id] = (true, modelSBML.species[id].compartment)
        end
        modelDict["states"][id] = '(' * modelDict["states"][id] * ") * " * modelSBML.species[id].compartment
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
    modelDict["stateGivenInAmounts"] = Dict()
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
    modelDict["algebraicRules"] = Dict()
    modelDict["assignmentRulesStates"] = Dict()
    # Mathemathical base functions (can be expanded if needed)
    baseFunctions = ["exp", "log", "log2", "log10", "sin", "cos", "tan", "pi"]
    stringOfEvents = ""
    discreteEventString = ""

    for (stateId, state) in modelSBML.species
        # If initial amount is zero or nothing (default) should use initial-concentration if non-empty 
        if isnothing(state.initial_amount) && isnothing(state.initial_concentration)
            modelDict["states"][stateId] = "0.0"
            modelDict["stateGivenInAmounts"][stateId] = (false, state.compartment)
        elseif !isnothing(state.initial_concentration)
            modelDict["states"][stateId] = string(state.initial_concentration)
            modelDict["stateGivenInAmounts"][stateId] = (false, state.compartment)
        else 
            modelDict["states"][stateId] = string(state.initial_amount)
            modelDict["stateGivenInAmounts"][stateId] = (true, state.compartment)
        end

        # Setup for downstream processing 
        modelDict["hasOnlySubstanceUnits"][stateId] = isnothing(state.only_substance_units) ? false : state.only_substance_units
        modelDict["isBoundaryCondition"][stateId] = state.boundary_condition 

        # In case equation is given in conc., but state is given in amounts 
        modelDict["derivatives"][stateId] = "D(" * stateId * ") ~ "

        # In case being a boundary condition the state can only be changed by the user 
        if modelDict["isBoundaryCondition"][stateId] == true
           modelDict["derivatives"][stateId] *= "0.0"
        end
    end

    # Extract model parameters and their default values. In case a parameter is non-constant 
    # it is treated as a state. Compartments are treated simular to states (allowing them to 
    # be dynamic)
    nonConstantParameterNames = []
    for (parameterId, parameter) in modelSBML.parameters
        if parameter.constant == true
            modelDict["parameters"][parameterId] = string(parameter.value)
            continue
        end

        modelDict["hasOnlySubstanceUnits"][parameterId] = false
        modelDict["stateGivenInAmounts"][parameterId] = (false, "")
        modelDict["isBoundaryCondition"][parameterId] = false
        modelDict["states"][parameterId] = isnothing(parameter.value) ? "0.0" : string(parameter.value)
        modelDict["derivatives"][parameterId] = parameterId * " ~ "
        nonConstantParameterNames = push!(nonConstantParameterNames, parameterId)
    end
    for (compartmentId, compartment) in modelSBML.compartments
        # Allowed in SBML ≥ 2.0 with nothing, should then be interpreted as 
        # having no compartment (equal to a value of 1.0 for compartment)
        if compartment.constant == true
            size = isnothing(compartment.size) ? 1.0 : compartment.size
            modelDict["parameters"][compartmentId] = string(size)
            continue
        end
        
        modelDict["hasOnlySubstanceUnits"][compartmentId] = false
        modelDict["stateGivenInAmounts"][compartmentId] = (false, "")
        modelDict["isBoundaryCondition"][compartmentId] = false
        modelDict["states"][compartmentId] = isnothing(compartment.size) ? 1.0 : compartment.size
        modelDict["derivatives"][compartmentId] = compartmentId * " ~ "
        nonConstantParameterNames = push!(nonConstantParameterNames, compartmentId)
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
        _triggerFormula = replaceFunctionWithFormula(mathToString(event.trigger.math), modelDict["modelFunctions"])
        triggerFormula = asTrigger(_triggerFormula, modelDict, modelSBML)
        eventFormulas = Vector{String}(undef, length(event.event_assignments))
        eventAssignTo = similar(eventFormulas)
        for (i, eventAssignment) in pairs(event.event_assignments)
            eventAssignTo[i] = eventAssignment.variable
            eventFormulas[i] = replaceFunctionWithFormula(mathToString(eventAssignment.math), modelDict["modelFunctions"])
            # Species typically given in substance units, but formulas in conc. Thus we must account for assignment 
            # formula being in conc., but we are changing something by amount 
            if eventAssignTo[i] ∈ keys(modelSBML.species)
                if eventAssignTo[i] ∈ keys(modelSBML.species) && modelDict["stateGivenInAmounts"][eventAssignTo[i]][1] == false
                    continue
                end
                eventFormulas[i] = modelSBML.species[eventAssignTo[i]].compartment *  " * (" * eventFormulas[i] * ')'
            end
        end
        eventName = isempty(eventName) ? "event" * string(eIndex) : eventName
        modelDict["events"][eventName] = [triggerFormula, eventAssignTo .* " = " .* eventFormulas]
        eIndex += 1
    end

    assignmentRulesNames = []
    rateRulesNames = []
    for rule in modelSBML.rules
        if rule isa SBML.AssignmentRule
            ruleFormula = extractRuleFormula(rule)
            assignmentRulesNames = push!(assignmentRulesNames, rule.variable)
            processAssignmentRule!(modelDict, ruleFormula, rule.variable, baseFunctions, modelSBML)
        end

        if rule isa SBML.RateRule
            ruleFormula = extractRuleFormula(rule)
            rateRulesNames = push!(rateRulesNames, rule.variable)
            processRateRule!(modelDict, ruleFormula, rule.variable, modelSBML, baseFunctions)
        end

        if rule isa SBML.AlgebraicRule
            _ruleFormula = extractRuleFormula(rule)
            ruleFormula = replaceFunctionWithFormula(_ruleFormula, modelDict["modelFunctions"])
            ruleName = isempty(modelDict["algebraicRules"]) ? "1" : maximum(keys(modelDict["algebraicRules"])) * "1" # Need placeholder key 
            modelDict["algebraicRules"][ruleName] = "0 ~ " * ruleFormula
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

        formula = rewriteDerivatives(_formula, modelDict, baseFunctions, modelSBML, checkScaling=true)
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
    # For states given in amount but model equations are in conc., multiply with compartment 
    for (stateId, derivative) in modelDict["derivatives"]
        if modelDict["stateGivenInAmounts"][stateId][1] == false
            continue
        end
        # Algebraic rule (see below)
        if replace(derivative, " " => "")[end] == '~' || replace(derivative, " " => "")[end] == '0'
            continue
        end
        derivative = replace(derivative, "~" => "~ (") 
        modelDict["derivatives"][stateId] = derivative * ") * " * modelSBML.species[stateId].compartment
    end

    # For states given by assignment rules 
    for (state, formula) in modelDict["assignmentRulesStates"]
        modelDict["derivatives"][state] = state * " ~ " * formula
        if state ∈ nonConstantParameterNames
            delete!(modelDict["states"], state)
            delete!(modelDict["parameters"], state)
            nonConstantParameterNames = filter(x -> x != state, nonConstantParameterNames)
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

    # In case the model has algebraic rules some of the derivatives (up to this point) are zero. To figure out 
    # which variable for which the derivative should be eliminated as the state conc. is given by the algebraic
    # rule cycle through rules to see which state has not been given as assignment by another rule. Moreover, return 
    # flag that model is a DAE so it can be properly processed when creating PEtabODEProblem. 
    if !isempty(modelDict["algebraicRules"])
        for (species, reaction) in modelDict["derivatives"]
            shouldContinue = true
            # In case we have zero derivative for a state (e.g S ~ 0 or S ~)
            if species ∈ rateRulesNames || species ∈ assignmentRulesNames
                continue
            end
            if replace(reaction, " " => "")[end] != '~' && replace(reaction, " " => "")[end] != '0'
                continue
            end
            if species ∈ keys(modelSBML.species) && modelSBML.species[species].constant == true
                continue
            end
            if modelDict["isBoundaryCondition"][species] == true && modelSBML.species[species].constant == true
                continue
            end

            # Check if state occurs in any of the algebraic rules 
            for (ruleId, rule) in modelDict["algebraicRules"]
                if replaceWholeWord(rule, species, "") != rule 
                    shouldContinue = false
                end
            end
            shouldContinue == true && continue

            # If we reach this point the state eqution is zero without any form 
            # of assignment -> state must be solved for via the algebraic rule 
            delete!(modelDict["derivatives"], species)
        end
    end
    for nonConstantParameter in nonConstantParameterNames
        if nonConstantParameter ∉ keys(modelDict["derivatives"])
            continue
        end
        if replace(modelDict["derivatives"][nonConstantParameter], " " => "")[end] == '~'
            modelDict["derivatives"][nonConstantParameter] *= string(modelDict["states"][nonConstantParameter])
        end
    end

    modelDict["stringOfEvents"] = stringOfEvents
    modelDict["discreteEventString"] = discreteEventString
    modelDict["numOfParameters"] = string(length(keys(modelDict["parameters"])))
    modelDict["numOfSpecies"] = string(length(keys(modelDict["states"])))
    modelDict["nonConstantParameterNames"] = nonConstantParameterNames
    modelDict["rateRulesNames"] = rateRulesNames

    return modelDict
end


"""
    createODEModelFunction(modelDict, pathJlFile, modelName, juliaFile, writeToFile::Bool)

Takes a modelDict as defined by buildODEModelDictionary
and creates a Julia ModelingToolkit file and stores
the resulting file in dirModel with name modelName.jl.
"""
function createODEModelFunction(modelDict, pathJlFile, modelName, juliaFile, writeToFile::Bool)

        stringDict = Dict()
        stringDict["variables"] = Dict()
        stringDict["stateArray"] = Dict()
        stringDict["variableParameters"] = Dict()
        stringDict["algebraicVariables"] = Dict()
        stringDict["parameters"] = Dict()
        stringDict["parameterArray"] = Dict()
        stringDict["continuousEvents"] = Dict()
        stringDict["discreteEvents"] = Dict()
        stringDict["derivatives"] = Dict()
        stringDict["ODESystem"] = Dict()
        stringDict["initialSpeciesValues"] = Dict()
        stringDict["trueParameterValues"] = Dict()

        stringDict["variables"] = "    ModelingToolkit.@variables t "
        stringDict["stateArray"] = "    stateArray = ["
        stringDict["variableParameters"] = ""
        stringDict["algebraicVariables"] = ""
        stringDict["parameters"] = "    ModelingToolkit.@parameters "
        stringDict["parameterArray"] = "    parameterArray = ["
        stringDict["continuousEvents"] = ""
        stringDict["discreteEvents"] = ""
        stringDict["derivatives"] = "    eqs = [\n"
        stringDict["ODESystem"] = "    @named sys = ODESystem(eqs, t, stateArray, parameterArray)"
        stringDict["initialSpeciesValues"] = "    initialSpeciesValues = [\n"
        stringDict["trueParameterValues"] = "    trueParameterValues = [\n"

    if juliaFile == true
        
        for key in keys(modelDict["states"])
            stringDict["variables"] *= key * " "
            stringDict["stateArray"] *= replace(key,"(t)"=>"") * ", "
        end
        stringDict["stateArray"] = stringDict["stateArray"][1:end-2] * "]"
        
        if length(modelDict["inputFunctions"]) > 0
            for key in keys(modelDict["inputFunctions"])
                stringDict["variables"] *= key * "(t) "
            end
        end
        
        for key in keys(modelDict["parameters"])
            stringDict["parameters"] *= key * " "
            stringDict["parameterArray"] *= replace(key,"(t)"=>"") * ", "
        end
        stringDict["parameterArray"] = stringDict["parameterArray"][1:end-2] * "]"
        
        for eq in modelDict["equationList"]
            # Fixes equations containing ifelse
            if occursin("ifelse", eq)
                # The equation starts either with "D(key) ~" or just "key ~"
                tildePos = findfirst("~",eq)[1]
                key = eq[1:tildePos-1]
                key = replace(key,"D(" => "")
                key = replace(key,")" => "")
                key = replace(key," " => "")
                stringDict["derivatives"] *= "    " * modelDict["inputFunctions"][key] * ",\n"
            else
                stringDict["derivatives"] *= "    " * eq * ",\n"
            end
        end

        stringDict["derivatives"] = stringDict["derivatives"][1:end-2] * "\n    ]"
        
        for stat in modelDict["stateMap"]
            statN = replace(string(stat.first),"(t)"=>"")
            statV = string(stat.second)
            stringDict["initialSpeciesValues"] *= "        " * statN * " => " * statV * ", \n"
        end
        stringDict["initialSpeciesValues"] = stringDict["initialSpeciesValues"][1:end-3] * "\n    ]"
        

        parameterList = string.(modelDict["paramMap"])
        for par in parameterList
            stringDict["trueParameterValues"] *= "        " * par * ", \n"
        end
        # Adds boolVariables to trueParameterValues
        for par in keys(modelDict["boolVariables"])
            stringDict["trueParameterValues"] *= "        " * par * " => 0.0, \n"
        end
        stringDict["trueParameterValues"] = stringDict["trueParameterValues"][1:end-3] * "\n    ]"

    else

        # Add dummy to create system if empty 
        if isempty(modelDict["states"])
            modelDict["states"]["fooo"] = "0.0"
            modelDict["derivatives"]["fooo"] = "D(fooo) ~ 0.0"
        end            

        for key in keys(modelDict["states"])
            stringDict["variables"] *= key * "(t) "
        end
        for (key, value) in modelDict["assignmentRulesStates"]
            stringDict["variables"] *= key * "(t) "
        end

        for (key, value) in modelDict["states"]
            stringDict["stateArray"] *= key * ", "
        end
        for (key, value) in modelDict["assignmentRulesStates"]
            stringDict["stateArray"] *= key * ", "
        end
        stringDict["stateArray"] = stringDict["stateArray"][1:end-2] * "]"

        if length(modelDict["nonConstantParameters"]) > 0
            stringDict["variableParameters"] = "    ModelingToolkit.@variables"
            for key in keys(modelDict["nonConstantParameters"])
                stringDict["variableParameters"] *= " " * key * "(t)"
            end
        end
        
        if length(modelDict["inputFunctions"]) > 0
            stringDict["algebraicVariables"] = "    ModelingToolkit.@variables"
            for key in keys(modelDict["inputFunctions"])
                stringDict["algebraicVariables"] *= " " * key * "(t)"
            end
        end
        
        for key in keys(modelDict["parameters"])
            stringDict["parameters"] *= key * " "
        end

        for (index, key) in enumerate(keys(modelDict["parameters"]))
            if index < length(modelDict["parameters"])
                stringDict["parameterArray"] *= key * ", "
            else
                stringDict["parameterArray"] *= key * "]"
            end
        end
        if isempty(modelDict["parameters"])
            stringDict["parameters"] = ""
            stringDict["parameterArray"] *= "]"
        end

        stringDict["continuousEvents"] = modelDict["stringOfEvents"]
        if length(modelDict["stringOfEvents"]) > 0
            stringDict["continuousEvents"] = "    continuous_events = ["
            stringDict["continuousEvents"] *= "    " * modelDict["stringOfEvents"]
            stringDict["continuousEvents"] *= "    ]"
        end
        
        stringDict["discreteEvents"] = modelDict["discreteEventString"]
        if length(modelDict["discreteEventString"]) > 0
            stringDict["discreteEvents"] = "    discrete_events = ["
            stringDict["discreteEvents"] *= "    " * modelDict["discreteEventString"]
            stringDict["discreteEvents"] *=  "    ]"
        end

        sIndex = 1
        for key in keys(modelDict["states"])
            # If the state is not part of any reaction we set its value to zero, 
            # unless is has been removed from derivative dict as it is given by 
            # an algebraic rule 
            if key ∉ keys(modelDict["derivatives"]) # Algebraic rule given 
                continue
            end
            if occursin(Regex("~\\s*\$"),modelDict["derivatives"][key])
                modelDict["derivatives"][key] *= "0.0"
            end
            if sIndex == 1
                stringDict["derivatives"] *= "    " * modelDict["derivatives"][key]
            else
                stringDict["derivatives"] *= ",\n    " * modelDict["derivatives"][key]
            end
            sIndex += 1
        end
        for key in keys(modelDict["nonConstantParameters"])
            stringDict["derivatives"] *= ",\n    D(" * key * ") ~ 0"
        end
        for key in keys(modelDict["inputFunctions"])
            stringDict["derivatives"] *= ",\n    " * modelDict["inputFunctions"][key]
        end
        for key in keys(modelDict["algebraicRules"])
            stringDict["derivatives"] *= ",\n    " * modelDict["algebraicRules"][key]
        end
        for key in keys(modelDict["assignmentRulesStates"])
            stringDict["derivatives"] *= ",\n    " * key * " ~ " * modelDict["assignmentRulesStates"][key]
        end
        stringDict["derivatives"] *= "\n"
        stringDict["derivatives"] *= "    ]"

        if length(modelDict["stringOfEvents"]) > 0 && length(modelDict["discreteEventString"]) > 0
            stringDict["ODESystem"] = "    @named sys = ODESystem(eqs, t, stateArray, parameterArray, continuous_events = continuous_events, discrete_events = discrete_events)"
        elseif length(modelDict["stringOfEvents"]) > 0 && length(modelDict["discreteEventString"]) == 0
            stringDict["ODESystem"] = "    @named sys = ODESystem(eqs, t, stateArray, parameterArray, continuous_events = continuous_events)"
        elseif length(modelDict["stringOfEvents"]) == 0 && length(modelDict["discreteEventString"]) > 0
            stringDict["ODESystem"] = "    @named sys = ODESystem(eqs, t, stateArray, parameterArray, discrete_events = discrete_events)"
        end

        index = 1
        for (key, value) in modelDict["states"]

            # These should not be mapped into the u0Map as they are just dynamic 
            # parameters expression which are going to be simplifed away (and are 
            # not in a sense states since they are not give by a rate-rule)
            if key ∈ modelDict["nonConstantParameterNames"] && key ∉ modelDict["rateRulesNames"]
                continue
            end
            if typeof(value) <: Real
                value = string(value)
            elseif tryparse(Float64, value) !== nothing
                value = string(parse(Float64, value))
            end
            if index == 1
                assignString = "    " * key * " => " * value
            else
                assignString = ",\n    " * key * " => " * value
            end
            stringDict["initialSpeciesValues"] *= assignString
            index += 1
        end
        for (key, value) in modelDict["nonConstantParameters"]
            assignString = ",\n    " * key * " => " * value
            stringDict["initialSpeciesValues"] *= assignString
        end
        for (key, value) in modelDict["assignmentRulesStates"]
            assignString = ",\n    " * key * " => " * value
            stringDict["initialSpeciesValues"] *= assignString
        end
        stringDict["initialSpeciesValues"] *= "\n"
        stringDict["initialSpeciesValues"] *= "    ]"
        
        for (index, (key, value)) in enumerate(modelDict["parameters"])
            if tryparse(Float64,value) !== nothing
                value = string(parse(Float64,value))
            end
            if index == 1
                assignString = "    " * key * " => " * value
            else
                assignString = ",\n    " * key * " => " * value
            end
            stringDict["trueParameterValues"] *= assignString
        end
        stringDict["trueParameterValues"] *= "\n"
        stringDict["trueParameterValues"] *= "    ]"
        
    end

    ### Writing to file
    modelName = replace(modelName, "-" => "_")
    io = IOBuffer()
    println(io, "function getODEModel_" * modelName * "(foo)")
    println(io, "\t# Model name: " * modelName)
    println(io, "\t# Number of parameters: " * modelDict["numOfParameters"])
    println(io, "\t# Number of species: " * modelDict["numOfSpecies"])
    println(io, "")

    println(io, "    ### Define independent and dependent variables")
    println(io, stringDict["variables"])
    println(io, "")
    println(io, "    ### Store dependent variables in array for ODESystem command")
    println(io, stringDict["stateArray"])
    println(io, "")
    println(io, "    ### Define variable parameters")
    println(io, stringDict["variableParameters"])
    println(io, "    ### Define potential algebraic variables")
    println(io, stringDict["algebraicVariables"])
    println(io, "    ### Define parameters")
    println(io, stringDict["parameters"])
    println(io, "")
    println(io, "    ### Store parameters in array for ODESystem command")
    println(io, stringDict["parameterArray"])
    println(io, "")
    println(io, "    ### Define an operator for the differentiation w.r.t. time")
    println(io, "    D = Differential(t)")
    println(io, "")
    println(io, "    ### Continious events ###")
    println(io, stringDict["continuousEvents"])
    println(io, "    ### Discrete events ###")
    println(io, stringDict["discreteEvents"])
    println(io, "    ### Derivatives ###")
    println(io, stringDict["derivatives"])
    println(io, "")
    println(io, stringDict["ODESystem"])
    println(io, "")
    println(io, "    ### Initial species concentrations ###")
    println(io, stringDict["initialSpeciesValues"])
    println(io, "")
    println(io, "    ### SBML file parameter values ###")
    println(io, stringDict["trueParameterValues"])
    println(io, "")
    println(io, "    return sys, initialSpeciesValues, trueParameterValues")
    println(io, "")
    println(io, "end")
    strModel = String(take!(io))
    close(io)
    
    # In case user request file to be written 
    if writeToFile == true
        open(pathJlFile, "w") do f
            write(f, strModel)
        end
    end
    return strModel
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

    # Special function which must be rewritten to Julia syntax 
    if math.fn == "ceiling"
        formula, _ = _mathToString(math.args[1])
        return "ceil" * '(' * formula * ')', false
    end

    # Factorials are, naturally, very challenging for ODE solvers. In case against the odds they 
    # are provided we compute the factorial via the gamma-function (to handle Num type). 
    if math.fn == "factorial"
        @warn "Factorial in the ODE model. PEtab.jl can handle factorials, but, solving the ODEs with factorial is 
            numerically challenging, and thus if possible should be avioded"
        formula, _ = _mathToString(math.args[1])
        return "SpecialFunctions.gamma" * '(' * formula * " + 1.0)", false
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
