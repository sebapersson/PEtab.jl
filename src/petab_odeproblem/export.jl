"""
    export_petab(dir, prob::PEtabODEProblem, res) -> path_yaml

Export in the PEtab standard format `prob` to the directory `dir`, using the parameters in
`res` to populate the PEtab parameter table. `res` may be a parameter-estimation result
(e.g. `PEtabMultistartResult`) or a parameter vector in the order expected by `prob` (see
[`get_x`](@ref)).

Returns the path to the exported PEtab problem YAML file.

The exporter currently supports only problems that were provided in the PEtab and
PEtab-SciML standard formats (PEtab tables), and exported tables keep the same filenames as
in the original PEtab problem. Problems constructed via the Julia interface are not yet
exportable.
"""
function export_petab(
        dir_export::AbstractString, prob::PEtabODEProblem, res::EstimationResult
    )::String
    model_info = prob.model_info
    @unpack model, xindices = model_info

    if model.defined_in_julia == true
        throw(ArgumentError("Cannot export this `PEtabODEProblem`: as `export_petab` \
            currently supports only problems provided in the PEtab standard format"))
    end

    if !isdir(dir_export)
        mkpath(dir_export)
    end

    # Export unchanged tables
    _cp_petab_file(dir_export, model.paths[:yaml])
    _cp_petab_file(dir_export, model.paths[:SBML])
    _cp_petab_file(dir_export, model.paths[:conditions])
    _cp_petab_file(dir_export, model.paths[:observables])
    _cp_petab_file(dir_export, model.paths[:measurements])
    if haskey(model.paths, :experiments)
        _cp_petab_file(dir_export, model.paths[:experiments])
    end

    # PEtab SciML extension
    path_yaml_export = joinpath(dir_export, basename(model.paths[:yaml]))
    yaml_file = YAML.load_file(path_yaml_export)
    if _is_sciml_problem(yaml_file)
        _cp_petab_file(dir_export, model.paths[:mapping])
        _cp_petab_file(dir_export, model.paths[:hybridization])
        for ml_path in _get_ml_paths(model.paths[:yaml], yaml_file)
            _cp_petab_file(dir_export, ml_path)
        end
        for path_array_file in _get_array_paths(model.paths[:yaml], yaml_file)
            _cp_petab_file(dir_export, path_array_file)
        end
    end

    # PEtab parameters are always exported on linear scale
    x_transformed = deepcopy(prob.xnominal)
    x_transformed .= _get_x(res)
    ids_mech = filter(id -> !in(id, xindices.ids[:ml_est]), prob.xnames)
    x_transformed[ids_mech] .= transform_x(
        _get_x(res)[xindices.indices_est[:est_to_mech]], ids_mech, xindices
    )

    parameters_df = CSV.read(model.paths[:parameters], DataFrame; stringtype = String)
    @unpack nominal_value, parameter_id = model_info.petab_parameters
    for (i, id) in pairs(prob.xnames)
        id in  prob.model_info.xindices.ids[:ml_est] && continue
        ix = findfirst(x -> x == id, parameter_id)
        if parameters_df.nominalValue[ix] isa AbstractString
            parameters_df.nominalValue[ix] = string(x_transformed[i])
        else
            parameters_df.nominalValue[ix] = x_transformed[i]
        end
    end
    path_parameters = joinpath(dir_export, basename(model.paths[:parameters]))
    CSV.write(path_parameters, parameters_df; delim = '\t')

    # ML parameters are stored in HDF5-files
    if _is_sciml_problem(yaml_file)
        for ml_model in prob.model_info.model.ml_models.ml_models
            !in(ml_model.ml_id, prob.model_info.xindices.ids[:ml_est]) && continue

            path_array_file = _get_array_file_path(
                path_yaml_export, yaml_file, ml_model.ml_id
            )
            ml_ps_to_hdf5(path_array_file, ml_model, x_transformed[ml_model.ml_id])
        end
    end
    return path_yaml_export
end

function _cp_petab_file(dir_export::String, path_original::String)::Nothing
    if isempty(path_original)
        return nothing
    end
    path_new = joinpath(dir_export, basename(path_original))
    cp(path_original, path_new; force = true)
    return nothing
end

function _is_sciml_problem(yaml_file::Dict)::Bool
    return haskey(yaml_file, "extensions") && haskey(yaml_file["extensions"], "sciml")
end

function _get_ml_paths(path_yaml::String, yaml_file::Dict)::Vector{String}
    yaml_models = yaml_file["extensions"]["sciml"]["neural_nets"]
    ml_paths = String[]
    for ml_model_info in values(yaml_models)
        push!(ml_paths, joinpath(dirname(path_yaml), ml_model_info["location"]))
    end
    return ml_paths
end

function _get_array_paths(path_yaml::String, yaml_file::Dict)::Vector{String}
    path_array_files = String[]
    if !haskey(yaml_file["extensions"]["sciml"], "array_files")
        return path_array_files
    end
    return joinpath.(dirname(path_yaml), yaml_file["extensions"]["sciml"]["array_files"])
end

function _get_array_file_path(path_yaml::String, yaml_file::Dict, ml_id::Symbol)::String
    ml_id = string(ml_id)
    path_array_files = _get_array_paths(path_yaml, yaml_file)

    for path_array_file in path_array_files
        array_file = HDF5.h5open(path_array_file, "r")
        if haskey(array_file, "parameters") && haskey(array_file["parameters"], ml_id)
            close(array_file)
            return path_array_file
        end
        close(array_file)
    end

    path_array_file = joinpath(dirname(path_yaml), "parameters_$(ml_id).hdf5")
    push!(yaml_file["extensions"]["sciml"]["array_files"], path_array_file)
    YAML.write_file(path_yaml, yaml_file)
    return path_array_file
end
