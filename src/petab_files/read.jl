function read_tables_v1(path_yaml::String)::Dict{Symbol, DataFrame}
    paths = _get_petab_paths(path_yaml)
    parameters_df = _read_table(paths[:parameters], :parameters_v1)
    conditions_df = _read_table(paths[:conditions], :conditions_v1)
    observables_df = _read_table(paths[:observables], :observables_v1)
    measurements_df = _read_table(paths[:measurements], :measurements_v1)
    return Dict(:parameters => parameters_df, :conditions => conditions_df,
                :observables => observables_df, :measurements => measurements_df)
end

function read_tables_v2(path_yaml::String)::Dict{Symbol, DataFrame}
    petab_paths = _get_petab_paths(path_yaml)
    experiments_df = _read_table(petab_paths[:experiment], :experiments_v2)
    conditions_df = _read_table(petab_paths[:conditions], :conditions_v2)
    measurements_df = _read_table(petab_paths[:measurements], :measurements_v2)
    observables_df = _read_table(petab_paths[:observables], :observables_v2)
    parameters_df = _read_table(petab_paths[:parameters], :parameters_v2)
    return Dict(:parameters => parameters_df, :conditions => conditions_df,
                :observables => observables_df, :measurements => measurements_df,
                :experiments => experiments_df)
end


function _get_petab_paths(path_yaml::AbstractString)::Dict{Symbol, String}
    if !isfile(path_yaml)
        throw(PEtabFileError("PEtab problem YAML file does not exist at path $(path_yaml)"))
    end
    yaml_file = YAML.load_file(path_yaml)
    dirmodel = dirname(path_yaml)
    dirjulia = joinpath(dirmodel, "Julia_model_files")

    path_measurements = _get_path(yaml_file, dirmodel, "measurement_files")
    path_observables = _get_path(yaml_file, dirmodel, "observable_files")
    path_parameters = _get_path(yaml_file, dirmodel, "parameter_files")
    path_conditions = _get_path(yaml_file, dirmodel, "condition_files")
    petab_paths = Dict(:parameters => path_parameters, :conditions => path_conditions,
        :observables => path_observables, :measurements => path_measurements,
        :dirmodel => dirmodel, :dirjulia => dirjulia, :yaml => path_yaml)

    petab_version = _get_version(yaml_file)
    if petab_version == "1.0.0"
        path_SBML = _get_path(yaml_file, dirmodel, "sbml_files")
    else
        path_SBML = _get_model_path_v2(yaml_file, dirmodel)
        path_experiments = _get_path(yaml_file, dirmodel, "experiment_files")
        petab_paths[:experiment] = path_experiments
    end
    petab_paths[:SBML] = path_SBML
    return petab_paths
end

function _read_table(path::String, file::Symbol)::DataFrame
    # Optional tables in PEtab v2
    if isempty(path) && file == :experiments_v2
        return DataFrame(experimentId = String[])
    end
    if isempty(path) && file == :conditions_v2
        return DataFrame()
    end

    df = CSV.read(path, DataFrame; stringtype = String)
    _check_table(df, file)
    return df
end

function _get_path(yaml_file::Dict, dirmodel::String, file::String)::String
    petab_version = _get_version(yaml_file)

    # Condition and Experiment files are optional in PEtab v2
    if petab_version == "2.0.0"
        if isempty(yaml_file[file]) && file in ["condition_files", "experiment_files"]
            return ""
        end
        path = joinpath(dirmodel, yaml_file[file][1])
    elseif !(petab_version == "1.0.0" && file == "parameter_files")
        path = joinpath(dirmodel, yaml_file["problems"][1][file][1])
    else
        path = joinpath(dirmodel, yaml_file["parameter_file"])
    end
    if !isfile(path)
        throw(PEtabFileError("$(path) is not a valid path for the $file tables"))
    end
    return path
end

function _get_model_path_v2(yaml_file::Dict, dirmodel::String)::String
    model_files = yaml_file["model_files"]
    if length(keys(model_files)) != 1
        throw(PEtabFileError("PEtab.jl currently only supports 1 model file. If your \
            application has a need for multiple models, please open an issue on GitHub"))
    end

    model_id = collect(keys(model_files))[1]
    model_language = model_files[model_id]["language"]
    if model_language != "sbml"
        throw(PEtabFileError("PEtab.jl does not support $(model_language) models. \
            Currently only SBML models are supported. In case the provided model cannot \
            be re-written to an SBML model, please open an issue on GitHub"))
    end

    path_model = joinpath(dirmodel, model_files[model_id]["location"])
    if !isfile(path_model)
        throw(PEtabFileError("$(path_model) is not a valid path for the model file"))
    end
    return path_model
end

_get_version(path_yaml::String)::String = _get_version(YAML.load_file(path_yaml))
function _get_version(yaml_file::Dict)::String
    petab_version = string(yaml_file["format_version"])
    if petab_version in ["1", "1.0.0"]
        return "1.0.0"
    elseif petab_version in ["2", "2.0.0"]
        return "2.0.0"
    end
    throw(PEtabFileError("Invalid PEtab version in problem YAML file. Valid versions \
            are 1, 1.0.0, 2, or 2.0.0 not the provided $(petab_version)"))
end

"""
    _check_table(df, table::Symbol)::Nothing

Check that a PEtab table has the required columns, and each column has correct types.
"""
function _check_table(df, table::Symbol)::Nothing
    colsinfo = COLUMN_INFO[table]

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
    if table == :conditions_v1
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
