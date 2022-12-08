# Functions for efficently mapping parameters. Specifcially, mapping the optimizsing
# parameter vector to dynamic, observable and sd parameter vectors, and for mapping
# these three vectors efficently when computing the observables, u0 and sd-values.


"""
    getIndicesParam(paramData::ParamData, measurementData::MeasurementData)::ParameterIndices

    For a PeTab-model creates index vectors for extracting the dynamic-, observable- and sd-parameters from
    the  optimizsing parameter vector, and indices and maps to effectively for an observation, when computing
    the likelhood, to extract the correct SD- and obs-parameters.

    When correctly extracting sd- or obs-parameters the `ParamMap` struct is used. This struct has precomputed
    values for which parameters belong to a specific observation, for example, which of the to be estimated
    observable parameters should be used when computing a part of the likelihood.

    See also: [`ParameterIndices`, `ParamMap`]
"""
function getIndicesParam(paramData::ParamData,
                         measurementData::MeasurementData,
                         odeSystem::ODESystem,
                         experimentalConditionsFile::DataFrame)::ParameterIndices

    namesObsParam = getIdEst(measurementData.obsParam, paramData)
    isObsParam = [paramData.parameterID[i] in namesObsParam for i in eachindex(paramData.parameterID)]

    namesSdParam = getIdEst(measurementData.sdParams, paramData)
    isSdParam = [paramData.parameterID[i] in namesSdParam for i in eachindex(paramData.parameterID)]

    # Parameters not entering the ODE system (or initial values), are not noise-parameter or observable
    # parameters, but appear in SD and/or OBS functions. Must be tagged specifically as we do
    # want to compute gradients for these given a fixed ODE solution.
    condSpecificDynParam = identityCondSpecificDynParam(odeSystem, paramData, experimentalConditionsFile)
    isNonDynamicParam = (paramData.shouldEst .&& .!isSdParam .&& .!isObsParam)
    namesNonDynamicParam = paramData.parameterID[isNonDynamicParam]
    namesNonDynamicParam = namesNonDynamicParam[findall(x -> x ∉ (string.(parameters(odeSystem))), namesNonDynamicParam)]
    namesNonDynamicParam = namesNonDynamicParam[findall(x -> x ∉ condSpecificDynParam, namesNonDynamicParam)]
    isNonDynamicParam = [paramData.parameterID[i] in namesNonDynamicParam for i in eachindex(paramData.parameterID)]

    isDynamicParam = (paramData.shouldEst .&& .!isNonDynamicParam .&& .!isSdParam .&& .!isObsParam)
    namesParamDyn = paramData.parameterID[isDynamicParam]

    # A paramter can be both a SD and observable parameter
    namesSdObsNonDynParam::Array{String, 1} = string.(unique(vcat(namesSdParam, namesObsParam, namesNonDynamicParam)))
    namesParamEst::Array{String, 1} = string.(vcat(string.(namesParamDyn), string.(namesSdObsNonDynParam)))

    # Index vector for the dynamic and sd parameters as UInt32 vectors
    iDynPar::Array{UInt32, 1} = [findfirst(x -> x == namesParamDyn[i],  namesParamEst) for i in eachindex(namesParamDyn)]
    iSdPar::Array{UInt32, 1} = [findfirst(x -> x == namesSdParam[i],  namesParamEst) for i in eachindex(namesSdParam)]
    iObsPar::Array{UInt32, 1} = [findfirst(x -> x == namesObsParam[i],  namesParamEst) for i in eachindex(namesObsParam)]
    iNonDynPar::Array{UInt32, 1} = [findfirst(x -> x == namesNonDynamicParam[i],  namesParamEst) for i in eachindex(namesNonDynamicParam)]
    iSdObsNonDynPar::Array{UInt32, 1} = [findfirst(x -> x == namesSdObsNonDynParam[i],  namesParamEst) for i in eachindex(namesSdObsNonDynParam)]

    # When extracting observable or sd parameter for computing the likelhood a pre-calculcated map-array
    # is used to efficently extract correct parameters. The arrays below define which map to extract to
    # use in the map array for each specific observation
    # Observable parameters
    keysObsMap = unique(measurementData.obsParam)
    indexObsParamMap = [UInt32(findfirst(x -> x == measurementData.obsParam[i], keysObsMap)) for i in eachindex(measurementData.obsParam)]
    # Noise parameters
    keysSdMap = unique(measurementData.sdParams)
    indexSdParamMap = [UInt32(findfirst(x -> x == measurementData.sdParams[i], keysSdMap)) for i in eachindex(measurementData.sdParams)]

    # Build maps to efficently map Obs- and SD-parameter when computing the likelhood
    mapArrayObsParam = buildMapParameters(keysObsMap, measurementData, paramData, true)
    mapArraySdParam = buildMapParameters(keysSdMap, measurementData, paramData, false)

    namesAllString = string.(parameters(odeSystem))

    # Set up a map for changing ODEProblem model parameters when doing parameter estimation
    iDynParamInSys = [findfirst(x -> x == namesParamDyn[i], namesAllString) for i in eachindex(namesParamDyn)]
    iDynParamInSys = convert(Array{Integer, 1}, iDynParamInSys[.!isnothing.(iDynParamInSys)])
    iDynParamInVecEst = convert(Array{Integer, 1}, [findfirst(x -> x == namesAllString[iSys], namesParamEst) for iSys in iDynParamInSys])
    mapDynParEst = MapDynParEst(iDynParamInSys, iDynParamInVecEst)

    # Set up a map for changing between experimental conditions
    mapExpCond, constParamPerCond = getMapExpCond(odeSystem, namesAllString, measurementData, paramData, experimentalConditionsFile, namesParamDyn, namesParamEst)

    paramIndicies = ParameterIndices(iDynPar,
                                     iObsPar,
                                     iSdPar,
                                     iSdObsNonDynPar,
                                     iNonDynPar,
                                     string.(namesParamDyn),
                                     string.(namesObsParam),
                                     string.(namesSdParam),
                                     namesSdObsNonDynParam,
                                     string.(namesNonDynamicParam),
                                     namesParamEst,
                                     indexObsParamMap,
                                     indexSdParamMap,
                                     mapArrayObsParam,
                                     mapArraySdParam,
                                     mapDynParEst,
                                     mapExpCond,
                                     constParamPerCond)

    return paramIndicies
end


function buildMapParameters(keysMap::T1,
                            measurementData::MeasurementData,
                            parameterData::ParamData,
                            buildObsParam::Bool)::Array{ParamMap, 1} where T1<:Array{<:Union{<:String, <:AbstractFloat}, 1}


    mapEntries::Array{ParamMap, 1} = Array{ParamMap, 1}(undef, length(keysMap))

    if buildObsParam == true
        namesParamEst = getIdEst(measurementData.obsParam, parameterData)
    else
        namesParamEst = getIdEst(measurementData.sdParams, parameterData)
    end

    # For each Key build associated dict-entry which i) Holds the constant obsParam value or ii) holds the
    # index in obsParamVecEst-vector depending on the observable parameters should be estimated or not.
    for i in eachindex(keysMap)
        if isempty(keysMap[i])
            mapEntries[i] = ParamMap(Array{Bool, 1}(undef, 0), Array{UInt32, 1}(undef, 0), Array{Float64, 1}(undef, 0), UInt32(0))

        elseif typeof(keysMap[i]) <: AbstractFloat
            mapEntries[i] = ParamMap(Array{Bool, 1}(undef, 0), Array{UInt32, 1}(undef, 0), Array{Float64, 1}(undef, 0), UInt32(0))

        elseif !isempty(keysMap[i])

            paramsRet = split(keysMap[i], ';')
            nParamsRet::Int = length(paramsRet)
            shouldEst::Array{Bool, 1} = Array{Bool, 1}(undef, nParamsRet)
            indexUse::Array{UInt32, 1} = Array{UInt32, 1}(undef, nParamsRet)
            valuesConst::Array{Float64, 1} = Array{Float64, 1}(undef, nParamsRet)

            for j in eachindex(paramsRet)
                # In case observable parameter in paramsRet[j] should be estimated save which index
                # it has in the obsParam parameter vector.
                if paramsRet[j] in namesParamEst
                    shouldEst[j] = true
                    indexUse[j] = UInt32(findfirst(x -> x == paramsRet[j], namesParamEst))

                # In case observable parameter in paramsRet[j] is constant save its constant value.
                # The constant value can be found either directly in the measurementDataFile, or in
                # in the parametersFile.
                else
                    shouldEst[j] = false
                    # Hard coded in Measurement data file
                    if isNumber(paramsRet[j])
                        valuesConst[j] = parse(Float64, paramsRet[j])

                    # Hard coded in Parameters file
                    elseif paramsRet[j] in parameterData.parameterID
                        valuesConst[j] = parameterData.paramVal[findfirst(x -> x == paramsRet[j], parameterData.parameterID)]

                    else
                        println("Warning : cannot find matching for parameter ", paramsRet[j], " when building map.")
                    end
                end
            end

            mapEntries[i] = ParamMap(shouldEst, indexUse[shouldEst], valuesConst[.!shouldEst], UInt32(length(paramsRet)))
        end
    end

    return mapEntries
end


function getObsOrSdParam(paramVec, paramMap::ParamMap)

    # Helper function to in a non-mutating way map SD-parameters in non-mutating way
    function mapParamEst(iWal)
        whichI = sum(paramMap.shouldEst[1:iWal])
        return paramMap.indexUse[whichI]
    end
    function mapParamConst(iWal)
        whichI = sum(.!paramMap.shouldEst[1:iWal])
        return whichI
    end

    # In case of no SD/observable parameter exit function
    if paramMap.nParam == 0
        return
    end

    # In case of single-value return do not have to return an array and think about type
    if paramMap.nParam == 1
        if paramMap.shouldEst[1] == true
            return paramVec[paramMap.indexUse][1]
        else
            return paramMap.valuesConst[1]
        end
    end

    # In case some of the parameters are estimated the return array can be either dual
    # or tracked array when computing derivatives.
    nParamEst = sum(paramMap.shouldEst)
    if nParamEst == paramMap.nParam
        return paramVec[paramMap.indexUse]

    # Computaionally most demanding case. Here a subset (or all) of the parameters
    # are to be estimated. Thus important that valsRetRet is correct type and this
    # the code is non-mutating (to support Zygote)
    elseif nParamEst > 0
        valsRet = [paramMap.shouldEst[i] == true ? paramVec[mapParamEst(i)] : 0.0 for i in 1:paramMap.nParam]
        valsRetRet = [paramMap.shouldEst[i] == false ? paramMap.valuesConst[mapParamConst(i)] : valsRet[i] for i in 1:paramMap.nParam]
        return valsRetRet

    # In no parameter are estimated can return constant values as floats
    else
        return paramMap.valuesConst
    end
end


# Helper function for extracting Observable parameter when computing the cost. Will be heavily altered as slow.
function getIdEst(idsInStr::T1, paramData::ParamData) where T1<:Array{<:Union{<:String, <:AbstractFloat}, 1}
    idVec = String[]

    for i in eachindex(idsInStr)
        if isempty(idsInStr[i])
            continue
        elseif typeof(idsInStr[i]) <: AbstractFloat
            continue
        else
            idsInStrSplit = split(idsInStr[i], ';')
            for idInStr in idsInStrSplit

                # Disregard Id if parameters should not be estimated, or
                iParam = findfirst(x -> x == idInStr, paramData.parameterID)
                if isNumber(idInStr)
                    continue
                elseif isnothing(iParam)
                    println("Warning : param $idInStr could not be found in parameter file")
                elseif idInStr in idVec
                    continue
                elseif paramData.shouldEst[iParam] == false
                    continue
                else
                    idVec = vcat(idVec, idInStr)
                end
            end
        end
    end

    return idVec
end


# Identifaying dynamic parameters to estimate, where the dynamic parameters are only used for some specific
# experimental conditions.
function identityCondSpecificDynParam(odeSystem::ODESystem,
                                      parameterData::ParamData,
                                      experimentalConditionsFile::DataFrame)::Array{String, 1}

    allOdeParamNames = string.(parameters(odeSystem))
    paramShouldEst = parameterData.parameterID[parameterData.shouldEst]

    # List of parameters who are dynamic parameters but we use specific parameters for specific experimental
    # conditions.
    notNonDynParam = Array{String, 1}(undef, 0)

    # Extract which columns contain actual parameter values
    colNames = names(experimentalConditionsFile)
    iStart = colNames[2] == "conditionName" ? 3 : 2
    i = iStart

    # Identify parameters who are a part of the ODE system (but only used for specific experimental conditions)
    for i in iStart:ncol(experimentalConditionsFile)

        if colNames[i] ∉ allOdeParamNames
            println("Problem : Parameter ", colNames[i], " should be in the ODE model as it dicates an experimental condition")
        end

        for j in 1:nrow(experimentalConditionsFile)
            if (expCondVar = string(experimentalConditionsFile[j, i])) ∈ paramShouldEst
                notNonDynParam = vcat(notNonDynParam, expCondVar)
            end
        end
    end

    return unique(notNonDynParam)
end


# A mapping experimental map for experimental conditions
function getMapExpCond(odeSystem::ODESystem,
                       namesAllString::Array{String, 1},
                       measurementData::MeasurementData,
                       paramData::ParamData,
                       experimentalConditionsFile::DataFrame,
                       namesParamDyn::Array{String, 1},
                       namesParamEst::Array{String, 1})


    lenExpCond = nrow(experimentalConditionsFile)
    stateNamesStr = replace.(string.(states(odeSystem), "(t)" => ""))

    i_start = "conditionName" in names(experimentalConditionsFile) ? 3 : 2
    paramStateChange = names(experimentalConditionsFile)[i_start:end]

    mapExpCondArr = Array{MapExpCond, 1}(undef, lenExpCond)
    condIdName = Array{String, 1}(undef, lenExpCond)

    for i in 1:lenExpCond

        expCondParamConstVal::Array{Float64, 1} = Array{Float64, 1}(undef, 0)
        iOdeProbParamConstVal::Array{Int, 1} = Array{Int, 1}(undef, 0)
        expCondStateConstVal::Array{Float64, 1} = Array{Float64, 1}(undef, 0)
        iOdeProbStateConstVal::Array{Int, 1} = Array{Int, 1}(undef, 0)
        iDynEstVec::Array{Int, 1} = Array{Int, 1}(undef, 0)
        iOdeProbDynParam::Array{Int, 1} = Array{Int, 1}(undef, 0)

        rowI = string.(collect(experimentalConditionsFile[i, i_start:end]))
        condID = experimentalConditionsFile[i, 1]
        condIdName[i] = string(condID)

        for j in eachindex(rowI)

            # When the experimental condition parameters is a number it can either be a number to set a parameter
            # or state to.
            if isNumber(rowI[j])
                if paramStateChange[j] ∈ namesAllString
                    expCondParamConstVal = vcat(expCondParamConstVal, parse(Float64, rowI[j]))
                    iOdeProbParamConstVal = vcat(iOdeProbParamConstVal, findfirst(x -> x == paramStateChange[j], namesAllString))
                elseif paramStateChange[j] ∈ stateNamesStr
                    expCondStateConstVal = vcat(expCondStateConstVal, parse(Float64, rowI[j]))
                    iOdeProbStateConstVal = vcat(iOdeProbStateConstVal, findfirst(x -> x == paramStateChange[j], stateNamesStr))
                else
                    println("Error : Cannot build map for experimental condition ", paramStateChange[j])
                end
                continue
            end

            # In case we are changing to one of the parameters we are estimating
            if rowI[j] ∈ namesParamDyn
                iDynEstVec = vcat(iDynEstVec, findfirst(x -> x == rowI[j], namesParamEst))
                iOdeProbDynParam = vcat(iOdeProbDynParam, findfirst(x -> x == paramStateChange[j], namesAllString))
                continue
            end

            # In case rowI is a parameter (but that is constant)
        if rowI[j] ∈ paramData.parameterID
                iVal = findfirst(x -> x == rowI[j], paramData.parameterID)
                expCondParamConstVal = vcat(expCondParamConstVal, paramData.paramVal[iVal])
                iOdeProbParamConstVal = vcat(iOdeProbParamConstVal, findfirst(x -> x == paramStateChange[j], namesAllString))
                continue
            end

            # If we reach this far something is off
            println("Could not map parameters for condition $condID  parameters", rowI[j])
        end

        mapExpCondArr[i] = MapExpCond(string(condID),
                                      expCondParamConstVal,
                                      iOdeProbParamConstVal,
                                      expCondStateConstVal,
                                      iOdeProbStateConstVal,
                                      iDynEstVec,
                                      iOdeProbDynParam)
    end

    # In order to prevent Zygote from trying to find rules for the struct
    # mapExpCond the constant parameter values accross experimental conditions
    # are stored in a Vector{Vector} from which they can be retrevied.
    # TODO: Check if needed
    constParamPerCond = [mapExpCondArr[i].expCondParamConstVal[:] for i in eachindex(mapExpCondArr)] # Copy to avoid any potential issues

    return mapExpCondArr, constParamPerCond
end
