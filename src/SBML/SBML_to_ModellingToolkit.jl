# TODO: Refactor code and document functions. Check SBMLToolkit if can be used.



"""
    XmlToModellingToolkit(pathXml::String, model_name::String, dir_model::String)

Convert a SBML file in pathXml to a Julia ModelingToolkit file and store
the resulting file in dir_model with name model_name.jl.
"""
function XmlToModellingToolkit(pathXml::String, pathJlFile::AbstractString, model_name::AbstractString; 
                               onlyGetSBMLDict::Bool=false, ifelse_to_event::Bool=true, write_to_file::Bool=true)

    modelSBML = readSBML(pathXml)
    modelDict = buildODEModelDictionary(modelSBML, ifelse_to_event)

    if onlyGetSBMLDict == false
        modelStr = createODEModelFunction(modelDict, pathJlFile, model_name, false, write_to_file)
        return modelDict, modelStr
    end

    return modelDict, ""
end


"""
JLToModellingToolkit(pathJlFile::String, dir_julia::String, model_name::String; ifelse_to_event::Bool=true)
Loads the Julia ModelingToolkit file located in pathJlFile.
If the file contains ifelse statements and ifelse_to_event=true
a fixed file will be stored in the Julia_model_files folder
with the suffix _fix in its filename.
"""
function JLToModellingToolkit(pathJlFile::String, dir_julia::String, model_name::String; 
                              ifelse_to_event::Bool=true, write_to_file::Bool=true)
    
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
    modelDict["state_map"] = Dict()
    modelDict["paramMap"] = Dict()

    # Read modelFile to work with it
    odefun = include(pathJlFile)
    expr = Expr(:call, Symbol(odefun))
    odeSys, state_map, paramMap = eval(expr)

    modelDict["state_map"] = state_map
    modelDict["paramMap"] = paramMap

    # Extract some "metadata"
    modelDict["numOfParameters"] = string(length(paramMap))
    modelDict["numOfSpecies"] = string(length(state_map))

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
    
    for stat in state_map
        modelDict["states"][string(stat.first)] = string(stat.second)
    end
    
    #Initialize output model file path to input path
    modelFileJl = pathJlFile
    if ifelse_to_event == true
        # Rewrite any time-dependent ifelse to boolean statements such that we can express these as events.
        # This is recomended, as it often increases the stabillity when solving the ODE, and decreases run-time
        timeDependentIfElseToBool!(modelDict)
        if length(modelDict["boolVariables"]) > 0
            # changes final .jl in path to _fix.jl
            # and changes output model file path to the fixed one.
            fileName = splitpath(pathJlFile)[end]
            fileNameFix = replace(fileName, Regex(".jl\$") => "_fix.jl")
            modelFileJl = joinpath(dir_julia, fileNameFix)
            # Create a new "fixed" julia file
            modelStr = createODEModelFunction(modelDict, pathJlFile, model_name, true, write_to_file)
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

    # Handle that in SBML models sometimes t is decoded as time
    newDerivativeAsString = replaceWholeWord(newDerivativeAsString, "time", "t")

    return newDerivativeAsString
end


function processInitialAssignment(modelSBML, modelDict::Dict, baseFunctions::Array{String, 1})

    initallyAssignedVariable = Dict{String, String}()
    initallyAssignedParameter = Dict{String, String}()
    for (assignId, initialAssignment) in modelSBML.initial_assignments
        
        _formula = mathToString(initialAssignment)
        formula = rewriteDerivatives(_formula, modelDict, baseFunctions, modelSBML)
        # Initial time i zero 
        formula = replaceWholeWord(formula, "t", "0.0")

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


function buildODEModelDictionary(modelSBML, ifelse_to_event::Bool)

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
    modelDict["compartmentFormula"] = Dict()
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
            eventFormulas[i] = replaceWholeWord(eventFormulas[i], "t", "integrator.t")
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

    # In case we have that the compartment is given by an assignment rule, then we need to account for this 
    for (compartmentId, compartmenFormula) in modelDict["compartmentFormula"]
        for (eventId, event) in modelDict["events"]
            triggerFormula = event[1]
            eventAssignments = event[2]
            triggerFormula = replaceWholeWord(triggerFormula, compartmentId, compartmenFormula)
            for i in eachindex(eventAssignments)
                eventAssignments[i] = replaceWholeWord(eventAssignments[i], compartmentId, compartmenFormula)
            end
            modelDict["events"][eventId] = [triggerFormula, eventAssignments]
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
    if ifelse_to_event == true
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

    # Up to this point technically some states can have a zero derivative, but their value can change because 
    # their compartment changes. To sidestep this, turn the state into an equation 
    for (specie, reaction) in modelDict["derivatives"]
        if specie ∉ keys(modelSBML.species)
            continue
        end
        if replace(reaction, " " => "")[end] != '~' && replace(reaction, " " => "")[end] != '0'
            continue
        end
        divideWithCompartment = modelDict["stateGivenInAmounts"][specie][1] == false
        c = modelSBML.species[specie].compartment
        if divideWithCompartment == false
            continue
        end
        modelDict["derivatives"][specie] = specie * " ~ (" * modelDict["states"][specie] * ") / " * c
    end

    # Sometimes parameter can be non-constant, but still have a constant rhs and they primarly change value 
    # because of event assignments. This must be captured, so the SBML importer will look at the RHS of non-constant 
    # parameters, and if it is constant the parameter will be moved to the parameter regime again in order to avoid 
    # simplifaying the parameter away.
    for id in nonConstantParameterNames
        # Algebraic rule 
        if id ∉ keys(modelDict["derivatives"])
            continue
        end
        lhs, rhs = replace.(split(modelDict["derivatives"][id], '~'), " " => "")
        if lhs[1] == 'D'
            continue
        end
        if !isNumber(rhs)
            continue
        end
        modelDict["derivatives"][id] = "D(" * id * ") ~ 0" 
        modelDict["states"][id] = rhs
        nonConstantParameterNames = filter(x -> x != id, nonConstantParameterNames)
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
    createODEModelFunction(modelDict, pathJlFile, model_name, juliaFile, write_to_file::Bool)

Takes a modelDict as defined by buildODEModelDictionary
and creates a Julia ModelingToolkit file and stores
the resulting file in dir_model with name model_name.jl.
"""
function createODEModelFunction(modelDict, pathJlFile, model_name, juliaFile, write_to_file::Bool)

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
        
        for stat in modelDict["state_map"]
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
            if sIndex != 1
                stringDict["derivatives"] *= ",\n    D(" * key * ") ~ 0"
            else
                stringDict["derivatives"] *= ",    D(" * key * ") ~ 0"
                sIndex += 1
            end
        end
        for key in keys(modelDict["inputFunctions"])
            if sIndex != 1
                stringDict["derivatives"] *= ",\n    " * modelDict["inputFunctions"][key]
            else
                stringDict["derivatives"] *= "    " * modelDict["inputFunctions"][key]
                sIndex += 1
            end
        end
        for key in keys(modelDict["algebraicRules"])
            if sIndex != 1
                stringDict["derivatives"] *= ",\n    " * modelDict["algebraicRules"][key]
            else
                stringDict["derivatives"] *= "    " * modelDict["algebraicRules"][key]
                sIndex += 1
            end
        end
        for key in keys(modelDict["assignmentRulesStates"])
            if sIndex != 1
                stringDict["derivatives"] *= ",\n    " * key * " ~ " * modelDict["assignmentRulesStates"][key]
            else
                stringDict["derivatives"] *= "    " * key * " ~ " * modelDict["assignmentRulesStates"][key]
                sIndex += 1
            end
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
            if index != 1
                assignString = ",\n    " * key * " => " * value
            else
                assignString = "    " * key * " => " * value
                index += 1
            end
            stringDict["initialSpeciesValues"] *= assignString
        end
        for (key, value) in modelDict["assignmentRulesStates"]
            if index != 1
                assignString = ",\n    " * key * " => " * value
            else
                assignString = "    " * key * " => " * value
                index += 1
            end
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
    model_name = replace(model_name, "-" => "_")
    io = IOBuffer()
    println(io, "function getODEModel_" * model_name * "(foo)")
    println(io, "\t# Model name: " * model_name)
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
    if write_to_file == true
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

    if math.fn == "log" && length(math.args) == 2
        base, addParenthesis1 = _mathToString(math.args[1])
        arg, addParenthesis2 = _mathToString(math.args[2])
        part1 = addParenthesis1 ?  '(' * base * ')' : base
        part2 = addParenthesis2 ?  '(' * arg * ')' : arg
        return "log(" * part1 * ", " * part2 * ")", true
    end


    if math.fn == "root" && length(math.args) == 2
        base, addParenthesis1 = _mathToString(math.args[1])
        arg, addParenthesis2 = _mathToString(math.args[2])
        part1 = addParenthesis1 ?  '(' * base * ')' : base
        part2 = addParenthesis2 ?  '(' * arg * ')' : arg
        return  part2 * "^(1 / " * part1 * ")", true
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

    if math.fn ∈ ["lt", "gt", "leq", "geq", "eq"]
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

    if math.fn ∈ ["arctan", "arcsin", "arccos", "arcsec", "arctanh", "arcsinh", "arccosh", 
                  "arccsc", "arcsech", "arccoth", "arccot", "arccot", "arccsch"]
        @assert length(math.args) == 1
        formula, _ = _mathToString(math.args[1])
        return "a" * math.fn[4:end] * '(' * formula * ')', false
    end

    if math.fn ∈ ["exp", "log", "log2", "log10", "sin", "cos", "tan", "csc", "ln"]
        fn = math.fn == "ln" ? "log" : math.fn
        @assert length(math.args) == 1
        formula, _ = _mathToString(math.args[1])
        return fn * '(' * formula * ')', false
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
    # Time unit is consistently in models refered to as time 
    return "t", false
end
function _mathToString(math::SBML.MathAvogadro)
    # Time unit is consistently in models refered to as time 
    return "6.02214179e23", false
end
function _mathToString(math::SBML.MathConst)
    if math.id == "exponentiale"
        return "2.718281828459045", false
    elseif math.id == "pi"
        return "3.1415926535897", false
    else
        return math.id, false
    end
end
