#=
    Functions for creating the ParameterIndices struct. This index struct contains maps for how to
    split the θ_est into the dynamic, non-dynamic, sd and observable parameters. There are also
    indices for mapping parameters between odeProblem.p and θ_est for parameters which are constant
    constant accross experimental conditions, and parameters specific to experimental conditions.
=#

function computeIndicesθ(parameterInfo::ParametersInfo,
                         measurementsInfo::MeasurementsInfo,
                         petabModel::PEtabModel)::ParameterIndices

    experimentalConditionsFile = petabModel.pathConditions
    return computeIndicesθ(parameterInfo, measurementsInfo, petabModel.system, petabModel.parameterMap, petabModel.stateMap, experimentalConditionsFile)

end
function computeIndicesθ(parameterInfo::ParametersInfo,
                         measurementsInfo::MeasurementsInfo,
                         odeSystem,
                         paramterMap,
                         stateMap,
                         experimentalConditionsFile::CSV.File)::ParameterIndices

    θ_observableNames, θ_sdNames, θ_nonDynamicNames, θ_dynamicNames = computeθNames(parameterInfo, measurementsInfo,
        odeSystem, experimentalConditionsFile)
    # When computing the gradient tracking parameters not part of ODE system is helpful
    iθ_notOdeSystemNames::Vector{Symbol} = Symbol.(unique(vcat(θ_sdNames, θ_observableNames, θ_nonDynamicNames)))
    # Names in the big θ_est vector
    θ_estNames::Vector{Symbol} = Symbol.(vcat(θ_dynamicNames, iθ_notOdeSystemNames))

    # Indices for each parameter in the big θ_est vector
    iθ_dynamic::Vector{Int64} = [findfirst(x -> x == θ_dynamicNames[i], θ_estNames) for i in eachindex(θ_dynamicNames)]
    iθ_sd::Vector{Int64} = [findfirst(x -> x == θ_sdNames[i], θ_estNames) for i in eachindex(θ_sdNames)]
    iθ_observable::Vector{Int64} = [findfirst(x -> x == θ_observableNames[i], θ_estNames) for i in eachindex(θ_observableNames)]
    iθ_nonDynamic::Vector{Int64} = [findfirst(x -> x == θ_nonDynamicNames[i], θ_estNames) for i in eachindex(θ_nonDynamicNames)]
    iθ_notOdeSystem::Vector{Int64} = [findfirst(x -> x == iθ_notOdeSystemNames[i], θ_estNames) for i in eachindex(iθ_notOdeSystemNames)]

    # When extracting observable or sd parameter for computing the cost we use a pre-computed map to efficently
    # extract correct parameters
    mapθ_observable = buildθSdOrObservableMap(θ_observableNames, measurementsInfo, parameterInfo, buildθ_observable=true)
    mapθ_sd = buildθSdOrObservableMap(θ_sdNames, measurementsInfo, parameterInfo, buildθ_observable=false)

    # Compute a map to map parameters between θ_dynamic and odeProblem.p
    nameParametersODESystem = Symbol.(string.(parameters(odeSystem)))
    iθDynamic::Vector{Int64} = findall(x -> x ∈ nameParametersODESystem, θ_dynamicNames)
    iODEProblemθDynamic::Vector{Int64} = [findfirst(x -> x == θ_dynamicNames[i], nameParametersODESystem) for i in iθDynamic]
    mapODEProblem::MapODEProblem = MapODEProblem(iθDynamic, iODEProblemθDynamic)

    # Set up a map for changing between experimental conditions
    mapsConditionId::Dict{Symbol, MapConditionId} = getMapsConditionId(odeSystem, paramterMap, stateMap, parameterInfo, experimentalConditionsFile, θ_dynamicNames)

    # Set up a named tuple tracking the transformation of each parameter
    _θ_scale = [parameterInfo.parameterScale[findfirst(x -> x == θ_name, parameterInfo.parameterId)] for θ_name in θ_estNames]
    θ_scale::Dict{Symbol, Symbol} = Dict([(θ_estNames[i], _θ_scale[i]) for i in eachindex(θ_estNames)])

    θ_indices = ParameterIndices(iθ_dynamic,
                                 iθ_observable,
                                 iθ_sd,
                                 iθ_nonDynamic,
                                 iθ_notOdeSystem,
                                 θ_dynamicNames,
                                 θ_observableNames,
                                 θ_sdNames,
                                 θ_nonDynamicNames,
                                 iθ_notOdeSystemNames,
                                 θ_estNames,
                                 θ_scale,
                                 mapθ_observable,
                                 mapθ_sd,
                                 mapODEProblem,
                                 mapsConditionId)

    return θ_indices
end


function computeθNames(parameterInfo::ParametersInfo,
                       measurementsInfo::MeasurementsInfo,
                       odeSystem,
                       experimentalConditionsFile::CSV.File)::Tuple{Vector{Symbol},Vector{Symbol},Vector{Symbol},Vector{Symbol}}

    # Extract the name of all parameter types
    θ_observableNames::Vector{Symbol} = getNamesObservableOrSdParameters(measurementsInfo.observableParameters, parameterInfo)
    isθ_observable::Vector{Bool} = [parameterInfo.parameterId[i] in θ_observableNames for i in eachindex(parameterInfo.parameterId)]

    θ_sdNames::Vector{Symbol} = getNamesObservableOrSdParameters(measurementsInfo.noiseParameters, parameterInfo)
    isθ_sd::Vector{Bool} = [parameterInfo.parameterId[i] in θ_sdNames for i in eachindex(parameterInfo.parameterId)]

    # Non-dynamic parameters. This are parameters not entering the ODE system (or initial values), are not
    # noise-parameter or observable parameters, but appear in SD and/or OBS functions. We need to track these
    # as non-dynamic parameters since  want to compute gradients for these given a fixed ODE solution.
    # Non-dynamic parameters not allowed to be observable or sd parameters
    _isθ_nonDynamic = (parameterInfo.estimate .&& .!isθ_observable .&& .!isθ_sd)
    _θ_nonDynamicNames = parameterInfo.parameterId[_isθ_nonDynamic]
    # Non-dynamic parameters not allowed to be part of the ODE-system
    _θ_nonDynamicNames = _θ_nonDynamicNames[findall(x -> x ∉ Symbol.(string.(parameters(odeSystem))), _θ_nonDynamicNames)]
    # Non-dynamic parameters not allowed to be experimental condition specific parameters
    conditionsSpecificθDynamic = identifyCondSpecificDynanmicθ(odeSystem, parameterInfo, experimentalConditionsFile)
    θ_nonDynamicNames::Vector{Symbol} = _θ_nonDynamicNames[findall(x -> x ∉ conditionsSpecificθDynamic, _θ_nonDynamicNames)]
    isθ_nonDynamic = [parameterInfo.parameterId[i] in θ_nonDynamicNames for i in eachindex(parameterInfo.parameterId)]

    isθ_dynamic::Vector{Bool} = (parameterInfo.estimate .&& .!isθ_nonDynamic .&& .!isθ_sd .&& .!isθ_observable)
    θ_dynamicNames::Vector{Symbol} = parameterInfo.parameterId[isθ_dynamic]

    return θ_observableNames, θ_sdNames, θ_nonDynamicNames, θ_dynamicNames
end


# Helper function for extracting ID:s for observable and noise parameters from the noise- and observable column
# in the PEtab file.
function getNamesObservableOrSdParameters(noiseOrObservableCol::T1,
                                          parameterInfo::ParametersInfo) where {T1<:Vector{<:Union{<:String,<:AbstractFloat}}}

    θ_estNames = Symbol[]
    for i in eachindex(noiseOrObservableCol)
        if isempty(noiseOrObservableCol[i]) || isNumber(string(noiseOrObservableCol[i]))
            continue
        end

        parametersRowi = split(noiseOrObservableCol[i], ';')
        for _parameter in parametersRowi

            parameter = Symbol(_parameter)
            # Disregard Id if parameters should not be estimated, or
            iParameter = findfirst(x -> x == parameter, parameterInfo.parameterId)
            if isNumber(_parameter) || parameter in θ_estNames || parameterInfo.estimate[iParameter] == false
                continue
            elseif isnothing(iParameter)
                @error "Parameter $parameter could not be found in parameter file"
            end

            θ_estNames = vcat(θ_estNames, parameter)
        end
    end

    return θ_estNames
end


# Identifaying dynamic parameters to estimate, where the dynamic parameters are only used for some specific
# experimental conditions.
function identifyCondSpecificDynanmicθ(odeSystem,
                                       parameterInfo::ParametersInfo,
                                       experimentalConditionsFile::CSV.File)::Vector{Symbol}

    allODESystemParameters = string.(parameters(odeSystem))
    modelStateNames = string.(states(odeSystem))
    modelStateNames = replace.(modelStateNames, "(t)" => "")
    parametersToEstimate = parameterInfo.parameterId[parameterInfo.estimate]

    # List of parameters which have specific values for specific experimental conditions, these can be extracted
    # from the rows of the experimentalConditionsFile (where the column is the name of the parameter in the ODE-system,
    # and the rows are the corresponding names of the parameter value to estimate)
    conditionsSpecificθDynamic = Vector{Symbol}(undef, 0)
    colNames = string.(experimentalConditionsFile.names)
    length(colNames) == 1 && return conditionsSpecificθDynamic
    iStart = colNames[2] == "conditionName" ? 3 : 2 # Sometimes PEtab file does not include column conditionName
    for i in iStart:length(experimentalConditionsFile.names)

        if colNames[i] ∉ allODESystemParameters && colNames[i] ∉ modelStateNames
            @error "Problem : Parameter ", colNames[i], " should be in the ODE model as it dicates an experimental condition"
        end

        for j in eachindex(experimentalConditionsFile)
            if (_parameter = Symbol(string(experimentalConditionsFile[j][i]))) ∈ parametersToEstimate
                conditionsSpecificθDynamic = vcat(conditionsSpecificθDynamic, _parameter)
            end
        end
    end

    return unique(conditionsSpecificθDynamic)
end


# For each observation build a map that correctly from either θ_observable or θ_sd map extract the correct value
# for the time-point specific observable and noise parameters when compuing σ or h (observable) value.
function buildθSdOrObservableMap(θ_names::Vector{Symbol},
                                 measurementsInfo::MeasurementsInfo,
                                 parameterInfo::ParametersInfo;
                                 buildθ_observable=true)

    parameterMap::Vector{θObsOrSdParameterMap} = Vector{θObsOrSdParameterMap}(undef, length(measurementsInfo.time))
    if buildθ_observable == true
        timePointSpecificValues = measurementsInfo.observableParameters
    else
        timePointSpecificValues = measurementsInfo.noiseParameters
    end

    # For each time-point build an associated map which stores if i) noise/obserable parameters are constants, ii) should
    # be estimated, iii) and corresponding index in parameter vector
    for i in eachindex(parameterMap)
        # In case we do not have any noise/obserable parameter
        if isempty(timePointSpecificValues[i])
            parameterMap[i] = θObsOrSdParameterMap(Vector{Bool}(undef, 0), Vector{Int64}(undef, 0), Vector{Float64}(undef, 0), Int64(0), false)
        end

        # In case of a constant noise/obserable parameter encoded as a Float in the PEtab file.
        if typeof(timePointSpecificValues[i]) <: Real
            parameterMap[i] = θObsOrSdParameterMap(Vector{Bool}(undef, 0), Vector{Int64}(undef, 0), Float64[timePointSpecificValues[i]], Int64(0), true)
        end

        # In case observable or noise parameter maps to a parameter
        if !isempty(timePointSpecificValues[i]) && !(typeof(timePointSpecificValues[i]) <: Real)

            # Parameter are delimited by ; in the PEtab files and they can be constant, or they can
            # be in the vector to estimate θ
            parametersInExpression = split(timePointSpecificValues[i], ';')
            nParameters::Int = length(parametersInExpression)
            shouldEstimate::Vector{Bool} = Vector{Bool}(undef, nParameters)
            indexInθ::Vector{Int64} = Vector{Int64}(undef, nParameters)
            constantValues::Array{Float64,1} = Vector{Float64}(undef, nParameters)

            for j in eachindex(parametersInExpression)
                # In case observable parameter in paramsRet[j] should be estimated save which index
                # it has in the θ vector
                if Symbol(parametersInExpression[j]) ∈ θ_names
                    shouldEstimate[j] = true
                    indexInθ[j] = Int64(findfirst(x -> x == Symbol(parametersInExpression[j]), θ_names))
                    continue
                end

                # In case observable parameter in paramsRet[j] is constant save its constant value.
                # The constant value can be found either directly in the measurementsInfoFile, or in
                # in the parametersFile.
                shouldEstimate[j] = false
                # Hard coded in Measurement data file
                if isNumber(parametersInExpression[j])
                    constantValues[j] = parse(Float64, parametersInExpression[j])
                    continue
                end
                # Hard coded in Parameters file
                if Symbol(parametersInExpression[j]) in parameterInfo.parameterId
                    constantValues[j] = parameterInfo.nominalValue[findfirst(x -> x == Symbol(parametersInExpression[j]), parameterInfo.parameterId)]
                    continue
                end

                @error "Cannot find matching for parameter ", parametersInExpression[j], " when building map."
            end

            parameterMap[i] = θObsOrSdParameterMap(shouldEstimate, indexInθ[shouldEstimate], constantValues[.!shouldEstimate],
                Int64(length(parametersInExpression)), false)
        end
    end

    return parameterMap
end


# A map to accurately map parameters for a specific experimental conditionId to the ODE-problem
function getMapsConditionId(odeSystem,
                            parameterMap,
                            stateMap,
                            parameterInfo::ParametersInfo,
                            experimentalConditionsFile::CSV.File,
                            _θ_dynamicNames::Vector{Symbol})::Dict{Symbol, MapConditionId}

    θ_dynamicNames = string.(_θ_dynamicNames)
    nConditions = length(experimentalConditionsFile)
    modelStateNames = string.(states(odeSystem))
    modelStateNames = replace.(modelStateNames, "(t)" => "")
    allODESystemParameters = string.(parameters(odeSystem))

    iStart = :conditionName in experimentalConditionsFile.names ? 3 : 2 # conditionName is optional in PEtab file
    conditionSpecificVariables = string.(experimentalConditionsFile.names[iStart:end])

    mapsConditionId::Dict{Symbol, MapConditionId} = Dict()

    for i in 1:nConditions

        constantParameters::Vector{Float64} = Vector{Float64}(undef, 0)
        iODEProblemConstantParameters::Vector{Int64} = Vector{Int64}(undef, 0)
        constantsStates::Vector{Float64} = Vector{Float64}(undef, 0)
        iODEProblemConstantStates::Vector{Int64} = Vector{Int64}(undef, 0)
        iθDynamic::Vector{Int64} = Vector{Int64}(undef, 0)
        iODEProblemθDynamic::Vector{Int64} = Vector{Int64}(undef, 0)

        conditionIdName = Symbol(string(experimentalConditionsFile[i][1]))

        rowI = string.(collect(experimentalConditionsFile[i])[iStart:end])
        for j in eachindex(rowI)

            # In case a condition specific ode-system parameter is mapped to constant number
            if isNumber(rowI[j]) && conditionSpecificVariables[j] ∈ allODESystemParameters
                constantParameters = vcat(constantParameters, parse(Float64, rowI[j]))
                iODEProblemConstantParameters = vcat(iODEProblemConstantParameters, findfirst(x -> x == conditionSpecificVariables[j], allODESystemParameters))
                continue
            end
            if isNumber(rowI[j]) && conditionSpecificVariables[j] ∈ modelStateNames
                constantParameters = vcat(constantParameters, parse(Float64, rowI[j]))
                iODEProblemConstantParameters = vcat(iODEProblemConstantParameters, findfirst(x -> x == "__init__" * conditionSpecificVariables[j] * "__", allODESystemParameters))
                continue
            end
            isNumber(rowI[j]) && @error "Error : Cannot build map for experimental condition variable", conditionSpecificVariables[j]


            # In case we are trying to change one the θ_dynamic parameters we are estimating
            if rowI[j] ∈ θ_dynamicNames && conditionSpecificVariables[j] ∈ allODESystemParameters
                iθDynamic = vcat(iθDynamic, findfirst(x -> x == rowI[j], θ_dynamicNames))
                iODEProblemθDynamic = vcat(iODEProblemθDynamic, findfirst(x -> x == conditionSpecificVariables[j], allODESystemParameters))
                continue
            end
            if rowI[j] ∈ θ_dynamicNames && conditionSpecificVariables[j] ∈ modelStateNames
                iθDynamic = vcat(iθDynamic, findfirst(x -> x == rowI[j], θ_dynamicNames))
                iODEProblemθDynamic = vcat(iODEProblemθDynamic, findfirst(x -> x == "__init__" * conditionSpecificVariables[j] * "__", allODESystemParameters))
                continue
            end
            rowI[j] ∈ θ_dynamicNames && @error "Could not map " * string(conditionSpecificVariables[j]) * " when building condition map"

            # In case rowI is a parameter but we do not estimate said parameter
            if rowI[j] ∈ string.(parameterInfo.parameterId)
                iVal = findfirst(x -> x == rowI[j], string.(parameterInfo.parameterId))
                constantParameters = vcat(constantParameters, parameterInfo.nominalValue[iVal])
                iODEProblemConstantParameters = vcat(iODEProblemConstantParameters, findfirst(x -> x == conditionSpecificVariables[j], allODESystemParameters))
                continue
            end

            # In case rowI is missing (specifically NaN) the default SBML-file value should be used. To this end we need to
            # have access to the parameter and state map to handle both states and parameters. Then must fix such that
            # __init__ parameters  take on the correct value.
            if rowI[j] == "missing" && conditionSpecificVariables[j] ∈ allODESystemParameters
                valueDefault = getDefaultValueFromMaps(string(conditionSpecificVariables[j]), parameterMap, stateMap)
                constantParameters = vcat(constantParameters, valueDefault)
                iODEProblemConstantParameters = vcat(iODEProblemConstantParameters, findfirst(x -> x == conditionSpecificVariables[j], allODESystemParameters))
                continue
            end
            if rowI[j] == "missing" && conditionSpecificVariables[j] ∈ modelStateNames
                valueDefault = getDefaultValueFromMaps(string(conditionSpecificVariables[j]), parameterMap, stateMap)
                constantParameters = vcat(constantParameters, valueDefault)
                iODEProblemConstantParameters = vcat(iODEProblemConstantParameters, findfirst(x -> x == "__init__" * conditionSpecificVariables[j] * "__", allODESystemParameters))
                continue
            end

            # NaN can only applie for states
            if rowI[j] == "NaN" && conditionSpecificVariables[j] ∈ modelStateNames
                constantParameters = vcat(constantParameters, NaN)
                iODEProblemConstantParameters = vcat(iODEProblemConstantParameters, findfirst(x -> x == "__init__" * conditionSpecificVariables[j] * "__", allODESystemParameters))
                continue
            else
                strWrite = "If a row in conditions file is NaN then the column header must be a state"
                throw(PEtabFileError(strWrite))
            end

            # If we reach this far something is off and an error must be thrown
            strWrite = "Could not map parameters for condition " * string(conditionIdName) * " for parameter " * string(rowI[j])
            throw(PEtabFileError(strWrite))
        end

        mapsConditionId[conditionIdName] = MapConditionId(constantParameters,
                                                          iODEProblemConstantParameters,
                                                          constantsStates,
                                                          iODEProblemConstantStates,
                                                          iθDynamic,
                                                          iODEProblemθDynamic)
    end

    return mapsConditionId
end


# Extract default parameter value from state, or parameter map
function getDefaultValueFromMaps(whichParameterOrState, parameterMap, stateMap)

    parameterMapNames = string.([parameterMap[i].first for i in eachindex(parameterMap)])
    stateMapNames = replace.(string.([stateMap[i].first for i in eachindex(stateMap)]), "(t)" => "")

    # Parameters are only allowed to map to concrete values
    if whichParameterOrState ∈ parameterMapNames
        whichIndex = findfirst(x -> x == whichParameterOrState, parameterMapNames)
        return parse(Float64, string(parameterMap[whichIndex].second))
    end

    # States can by default map to a parameter by one level of recursion
    @assert whichParameterOrState ∈ stateMapNames
    whichIndex = findfirst(x -> x == whichParameterOrState, stateMapNames)
    valueMapTo = string(stateMap[whichIndex].second)
    if valueMapTo ∈ parameterMapNames
        whichIndexParameter = findfirst(x -> x == valueMapTo, parameterMapNames)
        _valueMapTo = string(parameterMap[whichIndexParameter].second)
        if _valueMapTo ∈ parameterMapNames
            _whichIndexParameter = findfirst(x -> x == _valueMapTo, parameterMapNames)
            return parse(Float64, string.(parameterMap[_whichIndexParameter].second))
        else
            return parse(Float64, _valueMapTo)
        end
    end

    return parse(Float64, string(stateMap[whichIndex].second))
end
