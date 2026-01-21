function _get_ps_path(ml_id::Symbol, paths::Dict{Symbol, String})::String
    yaml_file = YAML.load_file(paths[:yaml])
    array_files = yaml_file["extensions"]["sciml"]["array_files"]
    path_ps_file = String[]
    for array_file in array_files
        _path_ps_file = joinpath(paths[:dirmodel], array_file)
        if !isfile(_path_ps_file)
            throw(PEtab.PEtabInputError("The provided SciML extension array file \
                $(array_file) does not exist at $(_path_ps_file)"))
        end
        hdf5_file = HDF5.h5open(_path_ps_file, "r")
        if haskey(hdf5_file["parameters"], "$(ml_id)")
            push!(path_ps_file, _path_ps_file)
            break
        end
    end

    if isempty(path_ps_file)
        throw(PEtab.PEtabInputError("Parameters for neural network $(ml_id) has not \
            been provided in an array file"))
    end
    return path_ps_file[1]
end

function _get_input_file_path(input_variable::Union{String, Symbol}, yaml_file::Dict, paths::Dict{Symbol, String})::Union{String, Nothing}
    if !haskey(paths, :dirmodel)
        return nothing
    end

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

function _get_input_file_values(input_id::String, file_path::Symbol, condition_id::Symbol)::Array{Float64}
    input_file = HDF5.h5open(string(file_path), "r")
    # Find the correct dataset associated with the provided simulation condition
    for condition_ids in keys(input_file["inputs"][input_id])
        group_input_id = input_file["inputs"][input_id]

        # "0" encoding means the data applies across all simulation conditions
        if condition_ids == "0" && length(keys(input_file["inputs"][input_id])) == 1
            input_values = HDF5.read_dataset(group_input_id, "0")
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

function _input_isfile(input_variable::Union{String, Symbol}, yaml_file::Dict, paths::Dict{Symbol, String})::Bool
    # When defined in Julia
    if isempty(yaml_file)
        return false
    end

    input_file_path = _get_input_file_path(input_variable, yaml_file, paths)
    return !isnothing(input_file_path)
end
