"""
    setUpPeTabModel(modelName::String, dirModel::String)::PeTabModel

    Given a model directory (dirModel) containing the PeTab files and a
    xml-file on format modelName.xml will return a PeTabModel struct holding
    paths to PeTab files, ode-system in ModellingToolkit format, functions for
    evaluating yMod, u0 and standard deviations, and a parameter and state maps
    for how parameters and states are mapped in the ModellingToolkit ODE system
    along with state and parameter names.

    dirModel must contain a SBML file named modelName.xml, and files starting with
    measurementData, experimentalCondition, parameter, and observables (tsv-files).
    The latter files must be unique (e.g only one file starting with measurementData)

    TODO : Example
"""
function setUpPeTabModel(modelName::String, dirModel::String; forceBuildJlFile::Bool=false, verbose::Bool=true)::PeTabModel

    # Sanity check user input
    modelFileXml = dirModel * modelName * ".xml"
    modelFileJl = dirModel * modelName * ".jl"
    if !isdir(dirModel)
        if verbose
            @printf("Model directory %s does not exist\n", dirModel)
        end
    end
    if !isfile(modelFileXml)
        if verbose
            @printf("Model directory does not contain xml-file with name %s\n", modelName * "xml")
        end
    end
    # If Julia model file does exists build it
    if !isfile(modelFileJl) && forceBuildJlFile == false
        if verbose
            @printf("Julia model file does not exist - will build it\n")
        end
        modelDict = XmlToModellingToolkit(modelFileXml, modelName, dirModel)
    elseif isfile(modelFileJl) && forceBuildJlFile == false
        if verbose
            @printf("Julia model file exists at %s - will not rebuild it\n", modelFileJl)
        end
    elseif forceBuildJlFile == true
        if verbose
            @printf("By user option rebuilds Julia model file\n")
        end
        if isfile(modelFileJl)
            rm(modelFileJl)
        end
        modelDict = XmlToModellingToolkit(modelFileXml, modelName, dirModel)
    end

    # Extract ODE-system and mapping of maps of how to map parameters to states and model parmaeters
    include(modelFileJl)
    expr = Expr(:call, Symbol("getODEModel_" * modelName))
    odeSys, stateMap, paramMap = eval(expr)
    #odeSysUse = ode_order_lowering(odeSys)
    odeSysUse = structural_simplify(odeSys)
    # Parameter and state names for ODE-system
    parameterNames = parameters(odeSysUse)
    stateNames = states(odeSysUse)

    # Sanity check for presence of all PeTab-files
    pathMeasurementData = checkForPeTabFile("measurementData", dirModel)
    pathExperimentalCond = checkForPeTabFile("experimentalCondition", dirModel)
    pathParameters = checkForPeTabFile("parameters", dirModel)
    pathObservables = checkForPeTabFile("observables", dirModel)

    # Build functions for observables, sd and u0 if does not exist and include
    pathObsSdU0 = dirModel * modelName * "ObsSdU0.jl"
    if !isfile(pathObsSdU0) || forceBuildJlFile == true
        if verbose && forceBuildJlFile == false
            @printf("File for yMod, U0 and Sd does not exist - building it\n")
        end
        if verbose && forceBuildJlFile == true
            @printf("By user option will rebuild Ymod, Sd and u0\n")
        end
        if !@isdefined(modelDict)
            modelDict = XmlToModellingToolkit(modelFileXml, modelName, dirModel, writeToFile=false)
        end
        createFileYmodSdU0(modelName, dirModel, odeSysUse, stateMap, modelDict)
    else
        if verbose
            @printf("File for yMod, U0 and Sd does exist - will not rebuild it\n")
        end
    end
    include(pathObsSdU0)

    peTabModel = PeTabModel(modelName,
                            evalYmod,
                            evalU0!,
                            evalU0,
                            evalSd!,
                            odeSysUse,
                            paramMap,
                            stateMap,
                            parameterNames,
                            stateNames,
                            dirModel,
                            pathMeasurementData,
                            pathExperimentalCond,
                            pathObservables,
                            pathParameters)

    return peTabModel
end


"""
    readDataFiles(dirModel::String; readObs::Bool=false)

    Given a directory for a model, e.g ./Beer_MolBioSystems2014, read the associated PeTab files
    for the measurements, parameters, experimental conditions and (if true) the observables.
"""
function readDataFiles(dirModel::String; readObs::Bool=false)

    # Check if PeTab files exist and get their path
    pathMeasurementData = checkForPeTabFile("measurementData", dirModel)
    pathExperimentalCond = checkForPeTabFile("experimentalCondition", dirModel)
    pathParameters = checkForPeTabFile("parameters", dirModel)
    pathObservables = checkForPeTabFile("observables", dirModel)

    experimentalConditions = CSV.read(pathExperimentalCond, DataFrame)
    measurementData = CSV.read(pathMeasurementData, DataFrame)
    parameterData = CSV.read(pathParameters, DataFrame)
    if readObs == true
        observableData = CSV.read(pathObservables, DataFrame)
        return experimentalConditions, measurementData, parameterData, observableData
    else
        return experimentalConditions, measurementData, parameterData
    end
end


"""
    processParameterData(parameterData::DataFrame)::ParamData

    Process the PeTab parameterData file into a type-stable Julia struct.
"""
function processParameterData(parameterData::DataFrame)::ParamData

    nParam = length(parameterData[!, "estimate"])

    # Pre-allocate arrays to hold data
    lb::Array{Float64, 1} = zeros(Float64, nParam)
    ub::Array{Float64, 1} = zeros(Float64, nParam)
    paramVal::Array{Float64, 1} = zeros(Float64, nParam) # Vector with Nominal value in PeTab-file
    logScale::Array{Bool, 1} = Array{Bool, 1}(undef, nParam)
    paramId::Array{String, 1} = Array{String, 1}(undef, nParam)
    shouldEst::Array{Bool, 1} = Array{Bool, 1}(undef, nParam)

    for i in eachindex(shouldEst)

        # If upper or lower bounds are missing assume +Inf and -Inf respectively.
        if ismissing(parameterData[i, "lowerBound"])
            lb[i] = -Inf
        else
            lb[i] = parameterData[i, "lowerBound"]
        end
        if ismissing(parameterData[i, "upperBound"])
            ub[i] = Inf
        else
            ub[i] = parameterData[i, "upperBound"]
        end

        paramVal[i] = parameterData[i, "nominalValue"]
        paramId[i] = parameterData[i, "parameterId"]
        # Currently only supports parameters on log10-scale -> TODO: Change this
        logScale[i] = parameterData[i, "parameterScale"] == "log10" ? true : false
        shouldEst[i] = parameterData[i, "estimate"] == 1 ? true : false
    end
    nParamEst::Int = Int(sum(shouldEst))

    return ParamData(paramVal, lb, ub, paramId, logScale, shouldEst, nParamEst)
end


"""
    processMeasurementData(measurementData::DataFrame, observableData::DataFrame)::MeasurementData

    Process the PeTab measurementData file into a type-stable Julia struct.
"""
function processMeasurementData(measurementData::DataFrame, observableData::DataFrame)::MeasurementData

    # Arrays for storing observed time-points and measurment values (yObs)
    yObs::Array{Float64, 1} = convert(Array{Float64, 1}, measurementData[!, "measurement"])
    tObs::Array{Float64, 1} = convert(Array{Float64, 1}, measurementData[!, "time"])
    nObs = length(yObs)

    # Get the experimental condition ID describing the experimental conditions for each observed time-point.
    # In case of preequilibration simulation the condition ID is stored in a single-string as the
    # concatenation of the pre and post equlibration ID:s.
    conditionId::Array{String, 1} = Array{String, 1}(undef, nObs)
    if !("preequilibrationConditionId" in names(measurementData))
        preEq = [missing for i in 1:nObs]
    else
        preEq = measurementData[!, "preequilibrationConditionId"]
    end
    simCond = String.(measurementData[!, "simulationConditionId"])
    for i in eachindex(conditionId)
        if ismissing(preEq[i])
            conditionId[i] = String(simCond[i])
        else
            conditionId[i] = String(preEq[i]) * String(simCond[i])
        end
    end
    if any(x -> ismissing(x), preEq)
        preEq = Array{String, 1}(undef, 0)
    else
        preEq = String.(preEq)
    end

    # PeTab observable ID for each measurment
    obsID::Array{String, 1} = string.(measurementData[!, "observableId"])

    # Noise parameters in the PeTab file either have a parameter ID, or they have
    # a value (fixed). Here regardless the values are mapped to the sdParams vector
    # as string. If sdObs[i] is numeric is the parsed before computing the cost.
    if !("noiseParameters" in names(measurementData))
        sdObs = [missing for i in 1:nObs]
    else
        sdObs = measurementData[!, "noiseParameters"]
    end
    sdParams::Array{Union{String, Float64}, 1} = Array{Union{String, Float64}, 1}(undef, nObs)
    for i in eachindex(sdObs)
        if ismissing(sdObs[i])
            sdParams[i] = ""
        elseif typeof(sdObs[i]) <:AbstractString && isNumber(sdObs[i])
            sdParams[i] = parse(Float64, sdObs[i])
        elseif typeof(sdObs[i]) <:Real
            sdParams[i] = convert(Float64, sdObs[i])
        else
            sdParams[i] = string(sdObs[i])
        end
    end

    # obsParamFile[i] can store more than one parameter. This is parsed
    # when computing the likelihood.
    if !("observableParameters" in names(measurementData))
        obsParamFile = [missing for i in 1:nObs]
    else
        obsParamFile = measurementData[!, "observableParameters"]
    end
    obsParam = Array{String, 1}(undef, nObs)
    for i in 1:nObs
        if ismissing(obsParamFile[i])
            obsParam[i] = ""
        else
            obsParam[i] = string(obsParamFile[i])
        end
    end

    # Currently supports log10 transformation of measurementData
    transformArr = Array{Symbol, 1}(undef, nObs)
    if !("observableTransformation" in names(observableData))
        transformArr .= [:lin for i in 1:nObs]
    else
        for i in 1:nObs
            iRow = findfirst(x -> x == obsID[i], observableData[!, "observableId"])
            transformArr[i] = Symbol(observableData[iRow, "observableTransformation"])
        end
    end

    # Save for each observation its pre-equlibrium and simulation condition id.


    # To avoid repeating calculations yObs is stored in a transformed and untransformed format
    yObsTransformed::Array{Float64, 1} = deepcopy(yObs)
    transformYobsOrYmodArr!(yObsTransformed, transformArr)

    # For each experimental condition we want to know the vector of time-points to save the ODE solution at
    # for each experimental condition. For each t-obs we also want to know which index in t-save vector
    # it corresponds to.
    iTimePoint = Array{Int64, 1}(undef, nObs)
    iPerConditionId = Dict() # Index in measurment data corresponding to specific condition id
    uniqueConditionID = unique(conditionId)
    tVecSave = Dict()
    for i in eachindex(uniqueConditionID)
        iConditionId = findall(x -> x == uniqueConditionID[i], conditionId)
        # Sorting is needed so that when extracting time-points when computing the cost
        # we extract the correct index.
        tVecSave[uniqueConditionID[i]] = sort(unique(tObs[iConditionId]))
        iPerConditionId[uniqueConditionID[i]] = iConditionId
        for j in iConditionId
            iTimePoint[j] = findfirst(x -> x == tObs[j], tVecSave[uniqueConditionID[i]])
        end
    end

    return MeasurementData(yObs, yObsTransformed, tObs, obsID, conditionId, sdParams, transformArr, obsParam, tVecSave, iTimePoint, iPerConditionId, preEq, simCond)
end


"""
    getTimeMax(measurementData::DataFrame, expId::String)::Float64

    Small helper function to get the time-max value for a specific simulationConditionId when simulating
    the PeTab ODE-model
"""
function getTimeMax(measurementData::DataFrame, expId::String)::Float64
    return Float64(maximum(measurementData[findall(x -> x == expId, measurementData[!, "simulationConditionId"]), "time"]))
end


"""
    getSimulationInfo(measurementData::DataFrame)::SimulationInfo

    Using the PeTab measurementData-file extract information on the foward ODE simulations.

    Specifcially extract the experimental ID:s from the experimentalCondition - PeTab file;
    firstExpIds (preequilibration ID:s), the shiftExpIds (postequilibration), and
    simulateSS (whether or not to simulate ODE-model to steady state). Further
    stores a solArray with the ODE solution where conditionIdSol of the ID for
    each forward solution

    TODO: Compute t-vec save at from measurementDataFile (instead of providing another struct)
"""
function getSimulationInfo(measurementDataFile::DataFrame,
                           measurementData::MeasurementData;
                           absTolSS::Float64=1e-8,
                           relTolSS::Float64=1e-6)::SimulationInfo

    # If preequilibrationConditionId column is not empty the model should
    # first be simulated to a stady state
    colNames = names(measurementDataFile)
    if !("preequilibrationConditionId" in colNames)
        preEqIDs = Array{String, 1}(undef, 0)
    else
        preEqIDs = convert(Array{String, 1}, unique(filter(x -> !ismissing(x), measurementDataFile[!, "preequilibrationConditionId"])))
    end
    simulateSS = length(preEqIDs) > 0

    # In case the the model is simulated to steday state get pre and post equlibration experimental conditions
    if simulateSS == true
        firstExpIds = preEqIDs
        shiftExpIds = Any[]
        for firstExpId in firstExpIds
            iRows = findall(x -> x == firstExpId, measurementDataFile[!, "preequilibrationConditionId"])
            shiftExpId = unique(measurementDataFile[iRows, "simulationConditionId"])
            push!(shiftExpIds, shiftExpId)
        end
        shiftExpIds = convert(Vector{Vector{String}}, shiftExpIds)
    end

    # In case the the model is mpt simulated to steday state store experimental condition in firstExpIds
    if simulateSS == false
        firstExpIds = convert(Array{String, 1}, unique(measurementDataFile[!, "simulationConditionId"]))
        shiftExpIds = Array{Array{String, 1}, 1}(undef, 0)
    end

    # Compute number of foward simulations to cover all experimental conditions and allocate array for them
    if simulateSS == true
        nForwardSol = Int64(sum([length(shiftExpIds[i]) for i in eachindex(shiftExpIds)]))
    else
        nForwardSol = Int64(length(firstExpIds))
    end
    # When computing the gradient and hessian the ODE-system needs to be resolved to compute the gradient
    # of the dynamic parameters, while for the observable/sd parameters the system should not be resolved.
    # Rather, an ODE solution without dual numbers is required and this solution can be the same which is
    # used when computing the cost.
    solArray = Vector{ODESolution}(undef, nForwardSol)
    solArrayGrad = Vector{ODESolution}(undef, nForwardSol)

    # Array with conition-ID for each foward simulations. As we always solve the ODE in the same order this can
    # be pre-computed.
    conditionIdSol = Array{String, 1}(undef, nForwardSol)
    tMaxForwardSim = Array{Float64, 1}(undef, nForwardSol)
    if simulateSS == true
        k = 1
        for i in eachindex(firstExpIds)
            for j in eachindex(shiftExpIds[i])
                firstExpId = firstExpIds[i]
                shiftExpId = shiftExpIds[i][j]
                conditionIdSol[k] = firstExpId * shiftExpId
                tMaxForwardSim[k] = getTimeMax(measurementDataFile, shiftExpId)
                k +=1
            end
        end
    else
        for i in eachindex(firstExpIds)
            firstExpId = firstExpIds[i]
            conditionIdSol[i] = firstExpId
            tMaxForwardSim[i] = getTimeMax(measurementDataFile, firstExpId)
        end

    end

    simulationInfo = SimulationInfo(firstExpIds,
                                    shiftExpIds,
                                    conditionIdSol,
                                    tMaxForwardSim,
                                    simulateSS,
                                    solArray,
                                    solArrayGrad,
                                    absTolSS,
                                    relTolSS,
                                    deepcopy(measurementData.tVecSave))
    return simulationInfo
end


"""
    checkForPeTabFile(fileSearchFor::String, dirModel::String)::String

    Helper function to check in dirModel if a file starting with fileSearchFor exists.
    If true return file path.
"""
function checkForPeTabFile(fileSearchFor::String, dirModel::String)::String

    filesDirModel = readdir(dirModel)
    iUse = findall(x -> occursin(fileSearchFor, x), filesDirModel)
    if length(iUse) > 1
        @printf("Error : More than 1 file starting with %s in %s\n", fileSearchFor, filesDirModel)
    end
    if length(iUse) == 0
        @printf("Error : No file starting with %s in %s\n", fileSearchFor, filesDirModel)
    end

    return dirModel * filesDirModel[iUse[1]]
end


function getPriorInfo(paramEstIndices::ParameterIndices, parameterDataFile::DataFrame)::PriorInfo

    if "objectivePriorType" ∉ names(parameterDataFile)
        return PriorInfo(Array{Function, 1}(undef, 0), Bool[], false)
    end

    namesParamEst = paramEstIndices.namesParamEst
    priorLogPdf = Array{Function, 1}(undef, length(namesParamEst))
    priorOnParamScale = Array{Bool, 1}(undef, length(namesParamEst))
    paramID = string.(parameterDataFile[!, "parameterId"])

    contPrior = 0.0
    for i in eachindex(namesParamEst)

        iUse = findfirst(x -> x == namesParamEst[i], paramID)

        priorF = parameterDataFile[iUse, "objectivePriorType"]
        if ismissing(priorF)
            priorLogPdf[i] = noPrior
            priorOnParamScale[i] = false
            continue
        end

        priorVal = parse.(Float64, split(parameterDataFile[iUse, "objectivePriorParameters"], ";"))

        if priorF == "parameterScaleNormal"
            contPrior += logpdf(Normal(priorVal[1], priorVal[2]), log10(parameterDataFile[iUse, "nominalValue"]))
            priorLogPdf[i] = (p) -> logpdf(Normal(priorVal[1], priorVal[2]), p)
            priorOnParamScale[i] = true
        elseif priorF == "parameterScaleLaplace"
            priorLogPdf[i] = (p) -> logpdf(Laplace(priorVal[1], priorVal[2]), p)
            priorOnParamScale[i] = true
        elseif priorF == "normal"
            priorLogPdf[i] = (p) -> logpdf(Normal(priorVal[1], priorVal[2]), p)
            priorOnParamScale[i] = false
        elseif priorF == "laplace"
            priorLogPdf[i] = (p) -> logpdf(Laplace(priorVal[1], priorVal[2]), p)
            priorOnParamScale[i] = false
        elseif priorF == "logNormal"
            priorLogPdf[i] = (p) -> logpdf(LogNormal(priorVal[1], priorVal[2]), p)
            priorOnParamScale[i] = false
        elseif priorF == "logLaplace"
            println("Error : Julia does not yet have support for log-laplace")
        else
            println("Error : PeTab standard does not support a prior of type ", priorF)
        end

    end

    return PriorInfo(priorLogPdf, priorOnParamScale, true)
end
# Helper function in case there is not any parameter priors
function noPrior(p::Real)::Real
    return 0.0
end
