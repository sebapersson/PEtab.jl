function read_tables(path_yaml::String)::Tuple{DataFrame, DataFrame, DataFrame, DataFrame}
    paths = _get_petab_paths(path_yaml)
    conditions_df = CSV.read(paths[:conditions], DataFrame; stringtype=String)
    _check_table(conditions_df, :conditions)
    measurements_df = CSV.read(paths[:measurements], DataFrame; stringtype=String)
    _check_table(measurements_df, :measurements)
    parameters_df = CSV.read(paths[:parameters], DataFrame; stringtype=String)
    _check_table(parameters_df, :parameters)
    observables_df = CSV.read(paths[:observables], DataFrame; stringtype=String)
    _check_table(observables_df, :observables)
    return conditions_df, measurements_df, parameters_df, observables_df
end

function _get_petab_paths(path_yaml::AbstractString)::NamedTuple
    if !isfile(path_yaml)
        throw(PEtabFileError("yaml file $(path_yaml) does not exist"))
    end
    yaml_file = YAML.load_file(path_yaml)
    dirmodel = dirname(path_yaml)
    path_SBML = _read_file(yaml_file, dirmodel, "sbml_files")
    path_measurements = _read_file(yaml_file, dirmodel, "measurement_files")
    path_observables = _read_file(yaml_file, dirmodel, "observable_files")
    path_conditions = _read_file(yaml_file, dirmodel, "condition_files")
    path_parameters = _read_file(yaml_file, dirmodel, "parameter_file")
    return (SBML=path_SBML, parameters=path_parameters, conditions=path_conditions, observables=path_observables, measurements=path_measurements)
end

function _read_file(yaml_file, dirmodel::String, file::String)::String
    if file != "parameter_file"
        path = joinpath(dirmodel, yaml_file["problems"][1][file][1])
    else
        path = joinpath(dirmodel, yaml_file[file])
    end
    if !isfile(path)
        throw(PEtabFileError("$path_conditions is not a valid path for the $file table"))
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

function _check_column_types(df::DataFrame, column_name::String, valid_types, table::Symbol)::Nothing
    for val in df[!, column_name]
        if !(typeof(val) <: valid_types)
            throw(PEtabFileError("Column $column_name in $table table has invalid type " *
                                 "invalid type $(typeof(val)) for entry $val. Allowed " *
                                 "types are $valid_types"))
        end
    end
    return nothing
end
