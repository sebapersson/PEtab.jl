function read_tables(path_yaml::String)::Dict{Symbol, DataFrame}
    paths = _get_petab_paths(path_yaml)
    parameters_df = _read_table(paths[:parameters], :parameters)
    conditions_df = _read_table(paths[:conditions], :conditions)
    observables_df = _read_table(paths[:observables], :observables)
    measurements_df = _read_table(paths[:measurements], :measurements)
    tables = Dict(:parameters => parameters_df, :conditions => conditions_df,
                  :observables => observables_df, :measurements => measurements_df)
    # Part of PEtab extensions, and not required and/or usually encountered.
    if haskey(paths, :mapping_table)
        tables[:mapping_table] = _read_table(paths[:mapping_table], :mapping)
    else
        tables[:mapping_table] = DataFrame()
    end
    return tables
end

function _get_petab_paths(path_yaml::AbstractString)::Dict{Symbol, String}
    if !isfile(path_yaml)
        throw(PEtabFileError("yaml file $(path_yaml) does not exist"))
    end
    yaml_file = YAML.load_file(path_yaml)
    version = yaml_file["format_version"]
    @assert version in [1, 2] "Incorrect PEtab version in yaml file"
    dirmodel = dirname(path_yaml)
    if version == 1
        return _parse_yaml_v1(yaml_file, dirmodel)
    else
        return _parse_yaml_v2(yaml_file, dirmodel)
    end
end

function _parse_yaml_v1(yaml_file, dirmodel::String)::Dict{Symbol, String}
    dirjulia = joinpath(dirmodel, "Julia_model_files")
    path_SBML = _get_path(yaml_file, dirmodel, "sbml_files")
    path_measurements = _get_path(yaml_file, dirmodel, "measurement_files")
    path_observables = _get_path(yaml_file, dirmodel, "observable_files")
    path_conditions = _get_path(yaml_file, dirmodel, "condition_files")
    path_parameters = _get_path(yaml_file, dirmodel, "parameter_file")
    return Dict(:SBML => path_SBML, :parameters => path_parameters,
                :conditions => path_conditions, :observables => path_observables,
                :measurements => path_measurements, :dirmodel => dirmodel,
                :dirjulia => dirjulia)
end

function _parse_yaml_v2(yaml_file, dirmodel::String)::Dict{Symbol, String}
    dirjulia = joinpath(dirmodel, "Julia_model_files")
    path_SBML = _get_path(yaml_file, dirmodel, "sbml_files")
    path_measurements = _get_path(yaml_file, dirmodel, "measurement_files")
    path_observables = _get_path(yaml_file, dirmodel, "observable_files")
    path_conditions = _get_path(yaml_file, dirmodel, "condition_files")
    path_parameters = _get_path(yaml_file, dirmodel, "parameter_file")
    path_mapping = _get_path(yaml_file, dirmodel, "mapping_files")
    return Dict(:SBML => path_SBML, :parameters => path_parameters,
                :conditions => path_conditions, :observables => path_observables,
                :measurements => path_measurements, :dirmodel => dirmodel,
                :dirjulia => dirjulia, :mapping_table => path_mapping)
end

function _read_table(path::String, file::Symbol)::DataFrame
    df = CSV.read(path, DataFrame; stringtype = String)
    _check_table(df, file)
    return df
end

function _get_path(yaml_file, dirmodel::String, file::String)::String
    # For version 2.0 different model languges are supported
    if file == "sbml_files" && haskey(yaml_file["problems"][1], "model_files")
        model_info = yaml_file["problems"][1]["model_files"]
        @assert model_info["language"] == "sbml" "Only SBML models are supported"
        path = joinpath(dirmodel, model_info["location"])
    elseif file != "parameter_file"
        path = joinpath(dirmodel, yaml_file["problems"][1][file][1])
    else
        path = joinpath(dirmodel, yaml_file[file])
    end
    if !isempty(path) && !isfile(path)
        throw(PEtabFileError("$path is not a valid path for the $file table"))
    end
    return path
end

"""
    _check_table(df, table::Symbol)::Nothing

Check that a PEtab table has the required columns, and each column has correct types.
"""
function _check_table(df, table::Symbol)::Nothing
    if table == :measurements
        colsinfo = MEASUREMENT_COLS
    elseif table == :conditions
        colsinfo = CONDITIONS_COLS
    elseif table == :parameters
        colsinfo = PARAMETERS_COLS
    elseif table == :observables
        colsinfo = OBSERVABLES_COLS
    elseif table == :mapping
        colsinfo = MAPPING_COLS
    end

    for (name, colinfo) in colsinfo
        if colinfo[:required] == true
            _check_has_column(df, name, table)
        end
        if name in names(df)
            _check_column_types(df, name, colinfo[:types], table)
        end
    end

    # For the condition table the columns (except conditionId/Name) correspond to parameter,
    # compartment or specie ids. As these column names depends on the problem, they cannot
    # be hard-coded, and must thus be handled as a special case
    if table == :conditions
        icheck = findall(x -> x ∉ ["conditionId", "conditionName"], names(df))
        for name in names(df)[icheck]
            _check_column_types(df, name, Union{AbstractString, Real, Missing}, :conditions)
        end
    end
    return nothing
end

function _check_has_column(df::DataFrame, column_name::String, table::Symbol)::Nothing
    if !(column_name in names(df))
        throw(PEtabFileError("Required column $column_name is missing from $table table"))
    end
    return nothing
end

function _check_column_types(df::DataFrame, column_name::String, valid_types,
                             table::Symbol)::Nothing
    for val in df[!, column_name]
        typeof(val) <: valid_types && continue
        throw(PEtabFileError("Column $column_name in $table table has invalid type " *
                             "invalid type $(typeof(val)) for entry $val. Valid " *
                             "types are $valid_types"))
    end
    return nothing
end
