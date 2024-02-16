function read_petab_yaml(path_yaml::AbstractString)
    if !isfile(path_yaml)
        throw(PEtabFileError("Model YAML file does not exist in the model directory"))
    end
    file_yaml = YAML.load_file(path_yaml)

    dir_model = dirname(path_yaml)
    path_SBML = joinpath(dir_model, file_yaml["problems"][1]["sbml_files"][1])
    if !isfile(path_SBML)
        throw(PEtabFileError("SBML file does not exist in the model directory"))
    end

    path_measurements = joinpath(dir_model,
                                 file_yaml["problems"][1]["measurement_files"][1])
    if !isfile(path_measurements)
        throw(PEtabFileError("Measurements file does not exist in the model directory"))
    end

    path_observables = joinpath(dir_model, file_yaml["problems"][1]["observable_files"][1])
    if !isfile(path_observables)
        throw(PEtabFileError("Observables file does not exist in the models directory"))
    end

    path_conditions = joinpath(dir_model, file_yaml["problems"][1]["condition_files"][1])
    if !isfile(path_conditions)
        throw(PEtabFileError("Conditions file does not exist in the models directory"))
    end

    path_parameters = joinpath(dir_model, file_yaml["parameter_file"])
    if !isfile(path_parameters)
        throw(PEtabFileError("Parameter file does not exist in the models directory"))
    end

    # Extract YAML directory and use directory name as model name and build directory for Julia files
    dir_julia = joinpath(dir_model, "Julia_model_files")
    model_name = splitdir(dir_model)[end]
    if !isdir(dir_julia)
        mkdir(dir_julia)
    end

    return path_SBML, path_parameters, path_conditions, path_observables, path_measurements,
           dir_julia, dir_model, model_name
end

function read_petab_files(path_yaml::String)
    path_SBML, path_parameters, path_conditions, path_observables, path_measurements, dir_julia, dir_model, model_name = read_petab_yaml(path_yaml)

    experimental_conditions = CSV.File(path_conditions, stringtype = String)
    measurements_data = CSV.File(path_measurements, stringtype = String)
    parameters_data = CSV.File(path_parameters, stringtype = String)
    observables_data = CSV.File(path_observables, stringtype = String)
    check_petab_files(experimental_conditions, measurements_data, parameters_data,
                      observables_data)

    return experimental_conditions, measurements_data, parameters_data, observables_data
end
function read_petab_files(petab_model::PEtabModel)
    experimental_conditions = petab_model.path_conditions
    measurements_data = petab_model.path_measurements
    parameters_data = petab_model.path_parameters
    observables_data = petab_model.path_observables
    check_petab_files(experimental_conditions, measurements_data, parameters_data,
                      observables_data)

    return experimental_conditions, measurements_data, parameters_data, observables_data
end

function check_petab_files(experimental_conditions::CSV.File, measurements_data,
                           parameters_data, observables_data)

    # Allowed DataTypes for the different columns in the files
    string_types = [String]
    number_types = [Float64, Int64]
    string_number_types = [String, Float64, Int64]
    transformation_types = ["lin", "log", "log10"]
    distribution_types = ["laplace", "normal"]
    estimateTypes = [0, 1]
    priorParameterTypes = [
        "uniform",
        "normal",
        "laplace",
        "logNormal",
        "logLaplace",
        "parameterScaleUniform",
        "parameterScaleNormal",
        "parameterScaleLaplace"
    ]

    # Check experimental_conditions
    columns_check = ["conditionId", "conditionName"]
    allowed_types = [string_types, string_types]
    required_columns = ["conditionId"]
    check_df_columns(experimental_conditions, "experimental_conditions", columns_check,
                     allowed_types, required_columns)

    columns_check = experimental_conditions.names
    # Check parameter columns in experimental_conditions
    if :conditionName in columns_check
        columns_check = columns_check[columns_check .!= :conditionId .&& columns_check .!= :conditionName]
    else
        columns_check = columns_check[columns_check .!= :conditionId]
    end
    columns_check = string.(columns_check)
    if !isempty(columns_check)
        allowed_types = repeat([string_number_types], length(columns_check))
        check_df_columns(experimental_conditions, "experimental_conditions", columns_check,
                         allowed_types, [])
    end

    # Check measurements_data

    columns_check = [
        "observableId",
        "simulationConditionId",
        "measurement",
        "time",
        "preequilibrationConditionId",
        "observableParameters",
        "noiseParameters",
        "datasetId",
        "replicateId"
    ]
    allowed_types = [
        string_types,
        string_types,
        number_types,
        number_types,
        string_types,
        string_number_types,
        string_number_types,
        string_number_types,
        string_number_types
    ]
    required_columns = ["observableId", "simulationConditionId", "measurement", "time"]
    check_df_columns(measurements_data, "measurements_data", columns_check, allowed_types,
                     required_columns)

    # Check parameters_data
    columns_check = [
        "parameterId",
        "parameterScale",
        "lowerBound",
        "upperBound",
        "nominalValue",
        "estimate",
        "parameterName",
        "initializationPriorType",
        "initializationPriorParameters",
        "objectivePriorType",
        "objectivePriorParameters"
    ]
    # Some models have missing parameters in their bounds even though it's mandatory, so we add Missing as an allowed DataType for these columns.
    allowed_types = [
        string_types,
        transformation_types,
        number_types,
        number_types,
        number_types,
        estimateTypes,
        string_types,
        priorParameterTypes,
        string_types,
        priorParameterTypes,
        string_types
    ]
    required_columns = [
        "parameterId",
        "parameterScale",
        "lowerBound",
        "upperBound",
        "nominalValue",
        "estimate"
    ]
    check_df_columns(parameters_data, "parameters_data", columns_check, allowed_types,
                     required_columns)

    # Check observables_data
    columns_check = [
        "observableId",
        "observableFormula",
        "noiseFormula",
        "observableName",
        "observableTransformation",
        "noiseDistribution"
    ]
    allowed_types = [
        string_types,
        string_types,
        string_number_types,
        string_types,
        transformation_types,
        distribution_types
    ]
    required_columns = ["observableId", "observableFormula", "noiseFormula"]
    check_df_columns(observables_data, "observables_data", columns_check, allowed_types,
                     required_columns)
end

"""
    check_df_columns(df, df_name, columns_check, allowed_types, required_columns)

Goes through each column from columns_check in df and checks
if each column is of any of the DataTypes specified in allowed_types[col_index].
Returns true if all columns are ok and false otherwise.
required_columns is an array of mandatory columns. If a mandatory column is missing
an error is thrown, and if a mandatory column contains missing rows a warning is thrown.
"""
function check_df_columns(df, df_name, columns_check, allowed_types, required_columns)
    check = true
    columns_check = Symbol.(columns_check)
    required_columns = Symbol.(required_columns)

    for col_index in eachindex(columns_check)
        column_name = columns_check[col_index]
        allowed_types_column = allowed_types[col_index]

        # If column is required and not present an error is thrown.
        if (column_name in required_columns) && !(column_name in df.names)
            throw(PEtabFileError("Required column '" * column_name * "' missing in file '" *
                                 df_name * "'"))
            # If column is required and there are missing values in that column a warning is thrown.
        elseif (column_name in required_columns) && (Missing <: eltype(df[column_name]))
            if column_name == "upperBound" || column_name == "lowerBound"
                for row in eachindex(df[column_name])
                    if df[row][column_name] === missing && df[row][:estimate] == 1
                        println("Warning : Required column " * column_name *
                                " contains rows with missing values on row " * string(row) *
                                ".")
                    end
                end
            else
                println("Warning : Required column " * string(column_name) *
                        " contains rows with missing values.")
            end
            # If column is not required and present the check is skipped.
        elseif !(column_name in required_columns) && !(column_name in string.(df.names))
            continue
        end

        # Extract column excluding missing values
        column_to_check = skipmissing(df[column_name])
        dType = eltype(column_to_check)

        if (allowed_types_column isa Array{DataType, 1})
            check &= dType <: Union{allowed_types_column...}
        elseif (allowed_types_column isa Array{String, 1}) ||
               (allowed_types_column isa Array{Int64, 1})
            check &= all(x -> x in allowed_types_column, column_to_check)
        end

        if !check
            throw(PEtabFileError("Wrong DataType or value in file '" * string(df_name) *
                                 "' column '" * string(column_name) * "'. Must be: " * "[" *
                                 join(allowed_types, ", ") * "]" * "."))
        end
    end
end
