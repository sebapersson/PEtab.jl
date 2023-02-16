#=
    Functions for creating the ParameterIndices struct. This index struct contains maps for how to
    split the θ_est into the dynamic, non-dynamic, sd and observable parameters. There are also
    indices for mapping parameters between odeProblem.p and θ_est for parameters which are constant
    constant accross experimental conditions, and parameters specific to experimental conditions.
=#


function computeIndicesθ(parameterInfo::ParametersInfo,
                         measurementsInfo::MeasurementsInfo,
                         odeSystem::ODESystem,
                         experimentalConditionsFile::DataFrame)::ParameterIndices

    θ_observableNames, θ_sdNames, θ_nonDynamicNames, θ_dynamicNames = computeθNames(parameterInfo, measurementsInfo,
                                                                        odeSystem, experimentalConditionsFile)
    # When computing the gradient tracking parameters not part of ODE system is helpful
    iθ_notOdeSystemNames::Vector{Symbol} = Symbol.(unique(vcat(θ_sdNames, θ_observableNames, θ_nonDynamicNames)))
    # Names in the big θ_est vector
    θ_estNames::Vector{Symbol} = Symbol.(vcat(θ_dynamicNames, iθ_notOdeSystemNames))

    # Indices for each parameter in the big θ_est vector
    iθ_dynamic::Vector{Int64} = [findfirst(x -> x == θ_dynamicNames[i],  θ_estNames) for i in eachindex(θ_dynamicNames)]
    iθ_sd::Vector{Int64} = [findfirst(x -> x == θ_sdNames[i],  θ_estNames) for i in eachindex(θ_sdNames)]
    iθ_observable::Vector{Int64} = [findfirst(x -> x == θ_observableNames[i],  θ_estNames) for i in eachindex(θ_observableNames)]
    iθ_nonDynamic::Vector{Int64} = [findfirst(x -> x == θ_nonDynamicNames[i],  θ_estNames) for i in eachindex(θ_nonDynamicNames)]
    iθ_notOdeSystem::Vector{Int64} = [findfirst(x -> x == iθ_notOdeSystemNames[i],  θ_estNames) for i in eachindex(iθ_notOdeSystemNames)]
    
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
    mapsConditionId = getMapsConditionId(odeSystem, parameterInfo, experimentalConditionsFile, θ_dynamicNames)

    # Set up a named tuple tracking the transformation of each parameter
    _θ_scale = [parameterInfo.parameterScale[findfirst(x -> x == θ_name, parameterInfo.parameterId)] for θ_name in θ_estNames]
    θ_scale::NamedTuple = NamedTuple{Tuple(name for name in θ_estNames)}(Tuple(scale for scale in _θ_scale))

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
                       odeSystem::ODESystem,
                       experimentalConditionsFile::DataFrame)::Tuple{Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}}

    # Extract the name of all parameter types
    θ_observableNames::Vector{Symbol} = getNamesObservableOrSdParameters(measurementsInfo.observableParameters, parameterInfo)
    isθ_observable::Vector{Bool}  = [parameterInfo.parameterId[i] in θ_observableNames for i in eachindex(parameterInfo.parameterId)]

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
                                          parameterInfo::ParametersInfo) where T1<:Vector{<:Union{<:String, <:AbstractFloat}}

    θ_estNames = Symbol[]
    for i in eachindex(noiseOrObservableCol)
        if isempty(noiseOrObservableCol[i])
            continue
        # Sometimes the observable or sd value are hardcoded
        elseif isNumber(noiseOrObservableCol[i])
            continue
        else
            parametersRowi = split(noiseOrObservableCol[i], ';')
            for _parameter in parametersRowi

                parameter = Symbol(_parameter)
                # Disregard Id if parameters should not be estimated, or
                iParameter = findfirst(x -> x == parameter, parameterInfo.parameterId)
                if isNumber(_parameter)
                    continue
                elseif isnothing(iParameter)
                    println("Warning : param $parameter could not be found in parameter file")
                elseif parameter in θ_estNames
                    continue
                elseif parameterInfo.estimate[iParameter] == false
                    continue
                else
                    θ_estNames = vcat(θ_estNames, parameter)
                end
            end
        end
    end

    return θ_estNames
end


# Identifaying dynamic parameters to estimate, where the dynamic parameters are only used for some specific
# experimental conditions.
function identifyCondSpecificDynanmicθ(odeSystem::ODESystem,
                                       parameterInfo::ParametersInfo,
                                       experimentalConditionsFile::DataFrame)::Vector{Symbol}

    allODESystemParameters = Symbol.(string.(parameters(odeSystem)))
    parametersToEstimate = parameterInfo.parameterId[parameterInfo.estimate]

    # List of parameters which have specific values for specific experimental conditions, these can be extracted
    # from the rows of the experimentalConditionsFile (where the column is the name of the parameter in the ODE-system,
    # and the rows are the corresponding names of the parameter value to estimate)
    conditionsSpecificθDynamic = Vector{Symbol}(undef, 0)
    colNames = names(experimentalConditionsFile)
    iStart = colNames[2] == "conditionName" ? 3 : 2 # Sometimes PEtab file does not include column conditionName
    for i in iStart:ncol(experimentalConditionsFile)

        if Symbol(colNames[i]) ∉ allODESystemParameters
            println("Problem : Parameter ", colNames[i], " should be in the ODE model as it dicates an experimental condition")
        end

        for j in 1:nrow(experimentalConditionsFile)
            if (_parameter = Symbol(string(experimentalConditionsFile[j, i]))) ∈ parametersToEstimate
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

        # In case of a constant noise/obserable parameter encoded as a Float in the PEtab file.
        elseif typeof(timePointSpecificValues[i]) <: AbstractFloat
            parameterMap[i] = θObsOrSdParameterMap(Vector{Bool}(undef, 0), Vector{Int64}(undef, 0), Float64[timePointSpecificValues[i]], Int64(0), true)

        elseif !isempty(timePointSpecificValues[i])

            # Parameter are delimited by ; in the PEtab files and they can be constant, or they can
            # be in the vector to estimate θ
            parametersInExpression = split(timePointSpecificValues[i], ';')
            nParameters::Int = length(parametersInExpression)
            shouldEstimate::Vector{Bool} = Vector{Bool}(undef, nParameters)
            indexInθ::Vector{Int64} = Vector{Int64}(undef, nParameters)
            constantValues::Array{Float64, 1} = Vector{Float64}(undef, nParameters)

            for j in eachindex(parametersInExpression)
                # In case observable parameter in paramsRet[j] should be estimated save which index
                # it has in the θ vector
                if Symbol(parametersInExpression[j]) ∈ θ_names
                    shouldEstimate[j] = true
                    indexInθ[j] = Int64(findfirst(x -> x == Symbol(parametersInExpression[j]), θ_names))

                # In case observable parameter in paramsRet[j] is constant save its constant value.
                # The constant value can be found either directly in the measurementsInfoFile, or in
                # in the parametersFile.
                else
                    shouldEstimate[j] = false
                    # Hard coded in Measurement data file
                    if isNumber(parametersInExpression[j])
                        constantValues[j] = parse(Float64, parametersInExpression[j])

                    # Hard coded in Parameters file
                    elseif Symbol(parametersInExpression[j]) in parameterInfo.parameterId
                        constantValues[j] = parameterInfo.nominalValue[findfirst(x -> x == Symbol(parametersInExpression[j]), parameterInfo.parameterId)]

                    else
                        println("Warning : cannot find matching for parameter ", parametersInExpression[j], " when building map.")
                    end
                end
            end

            parameterMap[i] = θObsOrSdParameterMap(shouldEstimate, indexInθ[shouldEstimate], constantValues[.!shouldEstimate],
                                                   Int64(length(parametersInExpression)), false)
        end
    end

    return parameterMap
end


# A map to accurately map parameters for a specific experimental conditionId to the ODE-problem
function getMapsConditionId(odeSystem::ODESystem,
                            parameterInfo::ParametersInfo,
                            experimentalConditionsFile::DataFrame,
                            _θ_dynamicNames::Vector{Symbol})::NamedTuple

    θ_dynamicNames = string.(_θ_dynamicNames)
    nConditions = nrow(experimentalConditionsFile)
    modelStateNames = replace.(string.(states(odeSystem), "(t)" => ""))
    allODESystemParameters = string.(parameters(odeSystem))

    iStart = "conditionName" in names(experimentalConditionsFile) ? 3 : 2 # conditionName is optional in PEtab file
    conditionSpecificVariables = names(experimentalConditionsFile)[iStart:end]

    _mapsConditionId::Vector{MapConditionId} = Vector{MapConditionId}(undef, nConditions)
    conditionIdNames = Vector{Symbol}(undef, nConditions)

    for i in 1:nConditions

        constantParameters::Vector{Float64} = Vector{Float64}(undef, 0)
        iODEProblemConstantParameters::Vector{Int64} = Vector{Int64}(undef, 0)
        constantsStates::Vector{Float64} = Vector{Float64}(undef, 0)
        iODEProblemConstantStates::Vector{Int64} = Vector{Int64}(undef, 0)
        iθDynamic::Vector{Int64} = Vector{Int64}(undef, 0)
        iODEProblemθDynamic::Vector{Int64} = Vector{Int64}(undef, 0)

        conditionIdNames[i] = Symbol(string(experimentalConditionsFile[i, 1]))

        rowI = string.(collect(experimentalConditionsFile[i, iStart:end]))
        for j in eachindex(rowI)

            # When the experimental condition parameters is a number (Float) it can either set the value
            # for an ODEProblem parameter or state
            if isNumber(rowI[j])
                if conditionSpecificVariables[j] ∈ allODESystemParameters
                    constantParameters = vcat(constantParameters, parse(Float64, rowI[j]))
                    iODEProblemConstantParameters = vcat(iODEProblemConstantParameters, findfirst(x -> x == conditionSpecificVariables[j], allODESystemParameters))
                elseif conditionSpecificVariables[j] ∈ modelStateNames
                    constantsStates = vcat(constantsStates, parse(Float64, rowI[j]))
                    iODEProblemConstantStates = vcat(iODEProblemConstantStates, findfirst(x -> x == conditionSpecificVariables[j], modelStateNames))
                else
                    println("Error : Cannot build map for experimental condition ", conditionSpecificVariables[j])
                end
                continue
            end

            # In case we are trying to change one the θ_dynamic parameters we are estimating
            if rowI[j] ∈ θ_dynamicNames
                iθDynamic = vcat(iθDynamic, findfirst(x -> x == rowI[j], θ_dynamicNames))
                iODEProblemθDynamic = vcat(iODEProblemθDynamic, findfirst(x -> x == conditionSpecificVariables[j], allODESystemParameters))
                continue
            end

            # In case rowI is a parameter but we do not estimate said parameter
            if rowI[j] ∈ string.(parameterInfo.parameterId)
                iVal = findfirst(x -> x == rowI[j], string.(parameterInfo.parameterId))
                constantParameters = vcat(constantParameters, parameterInfo.nominalValue[iVal])
                iODEProblemConstantParameters = vcat(iODEProblemConstantParameters, findfirst(x -> x == conditionSpecificVariables[j], allODESystemParameters))
                continue
            end

            # If we reach this far something is off
            strWrite = "Could not map parameters for condition ", conditionIdNames[i], " for parameter ", rowI[j]
            throw(PEtabFileError(strWrite))
        end

        _mapsConditionId[i] = MapConditionId(constantParameters,
                                             iODEProblemConstantParameters,
                                             constantsStates,
                                             iODEProblemConstantStates,
                                             iθDynamic,
                                             iODEProblemθDynamic)
    end

    mapsConditionId = Tuple(element for element in _mapsConditionId)
    return NamedTuple{Tuple(conditionId for conditionId in conditionIdNames)}(mapsConditionId)
end
