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
    end
    return tables
end

function _get_petab_paths(path_yaml::AbstractString)::Dict{Symbol, String}
    if !isfile(path_yaml)
        throw(PEtabFileError("yaml file $(path_yaml) does not exist"))
    end
    yaml_file = YAML.load_file(path_yaml)
    dirmodel = dirname(path_yaml)
    dirjulia = joinpath(dirmodel, "Julia_model_files")
    path_SBML = _get_path(yaml_file, dirmodel, "sbml_files")
    path_measurements = _get_path(yaml_file, dirmodel, "measurement_files")
    path_observables = _get_path(yaml_file, dirmodel, "observable_files")
    path_conditions = _get_path(yaml_file, dirmodel, "condition_files")
    path_parameters = _get_path(yaml_file, dirmodel, "parameter_file")
    paths = Dict(:SBML => path_SBML, :parameters => path_parameters,
                 :conditions => path_conditions, :observables => path_observables,
                 :measurements => path_measurements, :dirmodel => dirmodel,
                 :dirjulia => dirjulia)
    path_mapping = _get_path(yaml_file, dirmodel, "mapping_tables")
    if !isempty(path_mapping)
        paths[:mapping_table] = path_mapping
    end
    return paths
end

function _read_table(path::String, file::Symbol)::DataFrame
    df = CSV.read(path, DataFrame; stringtype = String)
    _check_table(df, file)
    return df
end

function _get_path(yaml_file, dirmodel::String, file::String)::String
    if !(file in ["parameter_file", "mapping_tables"])
        path = joinpath(dirmodel, yaml_file["problems"][file][1])
    elseif file == "mapping_tables"
        if !haskey(yaml_file["problems"], "mapping_tables")
            path = ""
        else
            path = joinpath(dirmodel, yaml_file["problems"][file])
        end
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
