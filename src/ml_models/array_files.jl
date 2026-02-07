function _get_ps_path(ml_id::Symbol, paths::Dict{Symbol, String})::String
    yaml_file = YAML.load_file(paths[:yaml])
    array_files = yaml_file["extensions"]["sciml"]["array_files"]

    path_ps_files = String[]
    for array_file in array_files
        path_ps = joinpath(paths[:dirmodel], array_file)
        if !isfile(path_ps)
            throw(PEtab.PEtabInputError("The provided SciML extension array file \
                $(array_file) does not exist at $(path_ps)"))
        end
        hdf5_file = HDF5.h5open(path_ps, "r")
        if haskey(hdf5_file["parameters"], "$(ml_id)")
            push!(path_ps_files, path_ps)
            close(hdf5_file)
            break
        end
        close(hdf5_file)
    end

    if isempty(path_ps_files)
        throw(PEtab.PEtabInputError("Parameters for neural network $(ml_id) has not \
            been provided in an array file"))
    end
    return path_ps_files[1]
end

function _get_input_path(
        input_variable::Union{String, Symbol}, yaml_file::Dict, paths::Dict{Symbol, String}
    )::Union{String, Nothing}
    !haskey(paths, :dirmodel) && return nothing

    yaml_file_extensions = yaml_file["extensions"]["sciml"]
    !haskey(yaml_file_extensions, "array_files") && return nothing

    for array_file_path in joinpath.(paths[:dirmodel], yaml_file_extensions["array_files"])
        array_file = HDF5.h5open(array_file_path, "r")
        !haskey(array_file, "inputs") && continue
        if haskey(array_file["inputs"], "$(input_variable)")
            close(array_file)
            return array_file_path
        end
        close(array_file)
    end
    return nothing
end

function _get_input_array(
        input_id::String, file_path::Symbol, condition_id::Symbol
    )::Array{Float64}

    input_file = HDF5.h5open(string(file_path), "r")
    # Find the correct dataset associated with the provided simulation condition
    for condition_ids in keys(input_file["inputs"][input_id])
        group_input_id = input_file["inputs"][input_id]

        # "0" encoding means the data applies across all simulation conditions
        if condition_ids == "0" && length(keys(input_file["inputs"][input_id])) == 1
            input_values = HDF5.read_dataset(group_input_id, "0")
            close(input_file)
            return input_values
        end

        !in(string(condition_id), split(condition_ids, ',')) && continue
        input_values = HDF5.read_dataset(group_input_id, condition_ids)
        close(input_file)
        return input_values
    end
    throw(PEtabInputError("The file $(file_path) which contains initial values for \
        neural network input ID $(input_id) does not provide any data for condition \
        $(condition_id)"))
end

function _input_isfile(
        input_variable::Union{String, Symbol}, yaml_file::Dict, paths::Dict{Symbol, String}
    )::Bool
    # When defined in Julia
    if isempty(yaml_file)
        return false
    end

    return !isnothing(_get_input_path(input_variable, yaml_file, paths))
end

# For simulation hybridization ML models
function _pase_array_inputs!(
        ml_models::MLModels, petab_tables::PEtabTables, paths::Dict{Symbol, String}
    )::Nothing
    for ml_model in ml_models.ml_models
        _pase_array_inputs!(ml_model, petab_tables, paths)
    end
    return nothing
end
function _pase_array_inputs!(
        ml_model::MLModel, petab_tables::PEtabTables, paths::Dict{Symbol, String}
    )::Nothing
    ml_model.pre_initialization == true && return nothing

    hybridization_df, mappings_df, conditions_df, experiments_df = _get_petab_tables(
        petab_tables, [:hybridization, :mapping, :conditions, :experiments]
    )

    arg_is_array = _get_array_args(ml_model, mappings_df, hybridization_df)

    has_pre_eq = !all(ismissing.(petab_tables[:measurements].preequilibrationConditionId))
    if has_pre_eq && any(arg_is_array .== true)
        throw(PEtabInputError("Array inputs via PEtab array files for hybridized ML \
            models (ml-model $(ml_model.ml_id)) are not supported in PEtab.jl for problems \
            with pre-equilibration (steady-state initialization). If you need this feature, \
            please open an issue: https://github.com/PEtab-dev/PEtab.jl/issues"))
    end

    # If is array, store the arrays in MLModel so it can be extracted when computing the
    # solution
    input_ids = _get_ml_model_io_petab_ids(mappings_df, ml_model.ml_id, :inputs)
    for arg_idx in eachindex(arg_is_array)
        arg_is_array[arg_idx] == false && continue

        input_id = input_ids[arg_idx][1]
        path_data = _get_input_path(input_id, petab_tables[:yaml], paths)

        input_file = HDF5.h5open(path_data, "r")
        group_data = input_file["inputs"][input_id]
        input_data_values = Dict{String, Array{Float64}}()
        for condition_ids in keys(group_data)
            input_data = HDF5.read_dataset(group_data, condition_ids)
            input_data = _reshape_input_array(input_data)

            input_keys = split(condition_ids, ';')
            for input_key in input_keys
                input_data_values[input_key] = input_data
            end
        end
        close(input_file)

        for condition_id in conditions_df.conditionId
            condition_v2_id = string(
                _get_petab_v2_condition_id(Symbol(condition_id), experiments_df)
            )

            key_array_inputs = Symbol("__arg$(arg_idx)_$(condition_id)")
            if haskey(input_data_values, condition_v2_id)
                ml_model.array_inputs[key_array_inputs] = input_data_values[condition_v2_id]
                continue
            elseif haskey(input_data_values, "0")
                ml_model.array_inputs[key_array_inputs] = input_data_values["0"]
                continue
            end
            throw(PEtabInputError("Could not assign array input for ml-model \
                $(ml_model.ml_id) in PEtab condition $(condition_v2_id). This occurs when: \
                (1) no value is assigned for this condition in the PEtab array files, or \
                (2) the condition is non-initial (not the first in a PEtab experiment). \
                Array input values for non-initial conditions are not supported in in PEtab.jl \
                simulation hybridized ML models."))
        end
    end
    return nothing
end
function _parse_array_input!(ml_models::MLModels, petab_tables::PEtabTables)
    for ml_model in ml_models.ml_models
        _parse_array_input!(ml_model, petab_tables)
    end
    return nothing
end
function _parse_array_input!(ml_model::MLModel, petab_tables::PEtabTables)
    ml_model.pre_initialization == true && return nothing

    hybridization_df, mappings_df, conditions_df = _get_petab_tables(
        petab_tables, [:hybridization, :mapping, :conditions]
    )

    arg_is_array = _get_array_args(ml_model, mappings_df, hybridization_df)

    for arg_idx in eachindex(arg_is_array)
        arg_is_array[arg_idx] == false && continue

        for condition_id in conditions_df.conditionId
            # In case of global array input
            key_condition = Symbol("__arg$(arg_idx)_$(condition_id)")
            key_global = Symbol("__arg$(arg_idx)")
            if haskey(ml_model.array_inputs, key_global)
                ml_model.array_inputs[key_condition] = ml_model.array_inputs[key_global]
                continue
            end

            # Should already be parsed along with condition table
            @assert haskey(ml_model.array_inputs, key_condition) "Could not parse \
                array input for ml-model $(ml_model.ml_id), please file an issue on \
                GitHub"
        end
    end
    return nothing
end

function _reshape_input_array(input_data::T)::T where T <: Vector{<:Real}
    return input_data
end
function _reshape_input_array(input_data::Array{<:Real})
    input_data = permutedims(input_data, reverse(1:ndims(input_data)))
    input_data = _reshape_io_data(input_data)
    return reshape(input_data, (size(input_data)..., 1))
end

function _get_array_args(
        ml_model::MLModel, mappings_df::DataFrame, hybridization_df::DataFrame
    )::Vector{Bool}
    input_ids = _get_ml_model_io_petab_ids(mappings_df, ml_model.ml_id, :inputs)

    arg_is_array = fill(false, length(input_ids))
    for (i, input_id) in pairs(input_ids)
        idx_row = findfirst(x -> x == input_id[1], hybridization_df.targetId)
        hybridization_df.targetValue[idx_row] != "array" && continue

        if length(input_id) > 1
            ml_id = ml_model.ml_id
            throw(PEtabInputError("If input to neural net is a file, only one \
                input can be provided in the mapping table. This does not \
                hold for $ml_id"))
        end

        arg_is_array[i] = true
    end
    return arg_is_array
end
