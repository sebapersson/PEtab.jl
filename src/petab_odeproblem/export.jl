"""
    export_petab(dir, prob::PEtabODEProblem, res) -> yaml_path

Export in the PEtab standard format `prob` to the directory `dir`, using the parameters in
`res` to populate the PEtab parameter table. `res` may be a parameter-estimation result
(e.g. `PEtabMultistartResult`) or a parameter vector in the order expected by `prob` (see
[`get_x`](@ref)).

Returns the path to the exported PEtab problem YAML file.

The exporter currently supports only problems that were provided in the PEtab standard
format (PEtab tables), and exported tables keep the same filenames as in the original
PEtab problem. Problems constructed via the Julia interface are not yet exportable.
"""
function export_petab(dir_export::AbstractString, prob::PEtabODEProblem, res::EstimationResult)::String
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

    # PEtab parameters are always exported on linear scale
    x_transformed = transform_x(_get_x(res), xindices.ids[:estimate], xindices)

    parameters_df = CSV.read(model.paths[:parameters], DataFrame)
    @unpack nominal_value, parameter_id = model_info.petab_parameters
    for (i, id) in pairs(prob.xnames)
        ix = findfirst(x -> x == id, parameter_id)
        parameters_df.nominalValue[ix] = x_transformed[i]
    end
    path_parameters = joinpath(dir_export, basename(model.paths[:parameters]))
    CSV.write(path_parameters, parameters_df; delim = '\t')

    return joinpath(dir_export, basename(model.paths[:yaml]))
end

function _cp_petab_file(dir_export::String, path_original::String)::Nothing
    if isempty(path_original)
        return nothing
    end
    path_new = joinpath(dir_export, basename(path_original))
    cp(path_original, path_new; force = true)
    return nothing
end
