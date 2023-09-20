function readPEtabYamlFile(path_yaml::AbstractString; jlFile::Bool=false)

    if !isfile(path_yaml)
        throw(PEtabFileError("Model YAML file does not exist in the model directory"))
    end
    fileYAML = YAML.load_file(path_yaml)

    dir_model = dirname(path_yaml)
    path_SBML = ""
    if jlFile == false
        path_SBML = joinpath(dir_model, fileYAML["problems"][1]["sbml_files"][1])
        if !isfile(path_SBML)
            throw(PEtabFileError("SBML file does not exist in the model directory"))
        end
    end

    path_measurements = joinpath(dir_model, fileYAML["problems"][1]["measurement_files"][1])
    if !isfile(path_measurements)
        throw(PEtabFileError("Measurements file does not exist in the model directory"))
    end

    path_observables = joinpath(dir_model, fileYAML["problems"][1]["observable_files"][1])
    if !isfile(path_observables)
        throw(PEtabFileError("Observables file does not exist in the models directory"))
    end

    path_conditions = joinpath(dir_model, fileYAML["problems"][1]["condition_files"][1])
    if !isfile(path_conditions)
        throw(PEtabFileError("Conditions file does not exist in the models directory"))
    end

    path_parameters = joinpath(dir_model, fileYAML["parameter_file"])
    if !isfile(path_parameters)
        throw(PEtabFileError("Parameter file does not exist in the models directory"))
    end

    # Extract YAML directory and use directory name as model name and build directory for Julia files
    dir_julia = joinpath(dir_model, "Julia_model_files")
    model_name = splitdir(dir_model)[end]
    if !isdir(dir_julia)
        mkdir(dir_julia)
    end

    return path_SBML, path_parameters, path_conditions, path_observables, path_measurements, dir_julia, dir_model, model_name
end


function readPEtabFiles(path_yaml::String; jlFile::Bool=false)

    path_SBML, path_parameters, path_conditions, path_observables, path_measurements, dir_julia, dir_model, model_name = readPEtabYamlFile(path_yaml, jlFile=jlFile)

    experimentalConditions = CSV.File(path_conditions, stringtype=String)
    measurementsData = CSV.File(path_measurements, stringtype=String)
    parametersData = CSV.File(path_parameters, stringtype=String)
    observablesData = CSV.File(path_observables, stringtype=String)
    checkFilesForCorrectDataType(experimentalConditions, measurementsData, parametersData, observablesData)

    return experimentalConditions, measurementsData, parametersData, observablesData
end
function readPEtabFiles(petab_model::PEtabModel)

    experimentalConditions = petab_model.path_conditions
    measurementsData = petab_model.path_measurements
    parametersData = petab_model.path_parameters
    observablesData = petab_model.path_observables
    checkFilesForCorrectDataType(experimentalConditions, measurementsData, parametersData, observablesData)

    return experimentalConditions, measurementsData, parametersData, observablesData
end


function checkFilesForCorrectDataType(experimentalConditions, measurementsData, parametersData, observablesData)

    # Allowed DataTypes for the different columns in the files
    stringTypes = [String]
    numberTypes = [Float64, Int64]
    stringOrNumberTypes = [String, Float64, Int64]
    transformationTypes = ["lin", "log", "log10"]
    distributionTypes = ["laplace", "normal"]
    estimateTypes = [0,1]
    priorParameterTypes = ["uniform", "normal", "laplace", "logNormal", "logLaplace", "parameterScaleUniform", "parameterScaleNormal", "parameterScaleLaplace"]

    # Check experimentalConditions
    colsToCheck = ["conditionId", "conditionName"]
    allowedTypesVec = [stringTypes, stringTypes]
    requiredCols = ["conditionId"]
    checkDataFrameColumns(experimentalConditions, "experimentalConditions", colsToCheck, allowedTypesVec, requiredCols)

    colsToCheck = experimentalConditions.names
    # Check parameter columns in experimentalConditions
    if :conditionName in colsToCheck
        colsToCheck = colsToCheck[colsToCheck .!= :conditionId .&& colsToCheck .!= :conditionName]
    else
        colsToCheck = colsToCheck[colsToCheck .!= :conditionId]
    end
    colsToCheck = string.(colsToCheck)
    if !isempty(colsToCheck)
        allowedTypesVec = repeat([stringOrNumberTypes], length(colsToCheck))
        checkDataFrameColumns(experimentalConditions, "experimentalConditions", colsToCheck, allowedTypesVec, [])
    end

    # Check measurementsData
    colsToCheck =     ["observableId", "simulationConditionId", "measurement", "time", "preequilibrationConditionId", "observableParameters", "noiseParameters", "datasetId", "replicateId"]
    allowedTypesVec = [stringTypes, stringTypes, numberTypes, numberTypes, stringTypes, stringOrNumberTypes, stringOrNumberTypes, stringOrNumberTypes, stringOrNumberTypes]
    requiredCols = ["observableId", "simulationConditionId", "measurement", "time"]
    checkDataFrameColumns(measurementsData, "measurementsData", colsToCheck, allowedTypesVec, requiredCols)

    # Check parametersData
    colsToCheck = ["parameterId", "parameterScale", "lowerBound", "upperBound", "nominalValue", "estimate", "parameterName", "initializationPriorType", "initializationPriorParameters", "objectivePriorType", "objectivePriorParameters"]
    # Some models have missing parameters in their bounds even though it's mandatory, so we add Missing as an allowed DataType for these columns.
    allowedTypesVec = [stringTypes, transformationTypes, numberTypes, numberTypes, numberTypes, estimateTypes, stringTypes, priorParameterTypes, stringTypes, priorParameterTypes, stringTypes]
    requiredCols = ["parameterId", "parameterScale", "lowerBound", "upperBound", "nominalValue", "estimate"]
    checkDataFrameColumns(parametersData, "parametersData", colsToCheck, allowedTypesVec, requiredCols)

    # Check observablesData
    colsToCheck = ["observableId", "observableFormula", "noiseFormula", "observableName", "observableTransformation", "noiseDistribution"]
    allowedTypesVec = [stringTypes, stringTypes, stringOrNumberTypes, stringTypes, transformationTypes, distributionTypes]
    requiredCols = ["observableId", "observableFormula", "noiseFormula"]
    checkDataFrameColumns(observablesData, "observablesData", colsToCheck, allowedTypesVec, requiredCols)

end


"""
checkDataFrameColumns(dataFrame, dataFrameName, colsToCheck, allowedTypesVec, requiredCols)

    Goes through each column from colsToCheck in dataFrame and checks
    if each column is of any of the DataTypes specified in allowedTypesVec[colIndex].
    Returns true if all columns are ok and false otherwise.
    requiredCols is an array of mandatory columns. If a mandatory column is missing
    an error is thrown, and if a mandatory column contains missing rows a warning is thrown.
"""
function checkDataFrameColumns(dataFrame, dataFrameName, colsToCheck, allowedTypesVec, requiredCols)

    check = true
    colsToCheck = Symbol.(colsToCheck)
    requiredCols = Symbol.(requiredCols)

    for colIndex in eachindex(colsToCheck)

        colName = colsToCheck[colIndex]
        allowedTypes = allowedTypesVec[colIndex]

        # If column is required and not present an error is thrown.
        if (colName in requiredCols) && !(colName in dataFrame.names)
            throw(PEtabFileError("Required column '" * colName * "' missing in file '" * dataFrameName * "'"))
        # If column is required and there are missing values in that column a warning is thrown.
        elseif (colName in requiredCols) && (Missing <: eltype(dataFrame[colName]))
            if colName == "upperBound" || colName == "lowerBound"
                for row in eachindex(dataFrame[colName])
                    if dataFrame[row][colName] === missing && dataFrame[row][:estimate] == 1
                        println("Warning : Required column " * colName * " contains rows with missing values on row " * string(row) * ".")
                    end
                end
            else
                println("Warning : Required column " * colName * " contains rows with missing values.")
            end
        # If column is not required and present the check is skipped.
        elseif !(colName in requiredCols) && !(colName in string.(dataFrame.names))
            continue
        end

        # Extract column excluding missing values
        colToCheck = skipmissing(dataFrame[colName])
        dType = eltype(colToCheck)

        if (allowedTypes isa Array{DataType, 1})
            check &= dType <: Union{allowedTypes...}
        elseif (allowedTypes isa Array{String, 1}) || (allowedTypes isa Array{Int64, 1})
            check &= all(x->x in allowedTypes, colToCheck)
        end

        if !check
            throw(PEtabFileError("Wrong DataType or value in file '" * string(dataFrameName) * "' column '" * string(colName) * "'. Must be: " * "[" * join(allowedTypes, ", ") * "]" * "."))
        end
    end
end
