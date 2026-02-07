function _get_ml_model_io_petab_ids(
        ml_models::MLModels, mappings_df::DataFrame
    )::Vector{String}
    out = String[]
    for ml_model in ml_models.ml_models
        input_ids = _get_ml_model_io_petab_ids(mappings_df, ml_model.ml_id, :inputs)
        isempty(input_ids) && continue
        out = vcat(out, reduce(vcat, input_ids))
    end
    return out
end
function _get_ml_model_io_petab_ids(
        mappings_df::DataFrame, ml_id::Symbol, type::Symbol
    )::Vector{Vector{String}}
    @assert type in [:inputs, :outputs]

    # Sort to get PEtab Id inputs/outputs in correct order (both within and between
    # arguments). File inputs are on the format netId.inputs[ix], while other inputs
    # are on the format netId.inputs[ix][jx]
    out = Vector{Vector{String}}(undef, 0)
    for i in 0:100
        # File input id
        regex = "$(ml_id).$(type)[$(i)]" * r"$"
        matches = match.(regex, mappings_df.modelEntityId)
        if !all(isnothing.(matches))
            @assert sum(.!isnothing(matches)) == 1 "Duplicates of $(ml_id).$(type)[$(i)] \
                in mapping table"
            push!(out, [mappings_df.petabEntityId[findfirst(!isnothing, matches)]])
            continue
        end

        regex = "$(ml_id).$(type)[$(i)]" * r"\[(\d+)\]$"
        matches = match.(regex, string.(mappings_df[!, "modelEntityId"]))
        all(isnothing.(matches)) && break

        df_match = mappings_df[.!isnothing.(matches), :]
        is = sortperm(
            df_match.modelEntityId, by = x -> parse(Int, match(regex, x).captures[1])
        )
        push!(out, df_match[is, :petabEntityId])
    end
    return out
end

function _get_ml_model_input_values(
        input_petab_ids::Vector{Symbol}, ml_id::Symbol, ml_model::MLModel,
        conditions_df::DataFrame, petab_tables::PEtabTables, paths::Dict{Symbol, String},
        petab_parameters::PEtabParameters, sys::ModelSystem; keep_numbers::Bool = false
    )::Vector{Symbol}
    hybridization_df, yaml_file = _get_petab_tables(petab_tables, [:hybridization, :yaml])

    input_values = Symbol[]
    for input_id in input_petab_ids
        # Condition-specific array input in Julia
        if isempty(yaml_file) && input_id == :array
            return [:_array_julia]
        end

        # If the input variable is a file, the complete path is added here, which simplifies
        # downstream processing
        if input_id in Symbol.(hybridization_df.targetId)
            ix = findfirst(x -> x == input_id, Symbol.(hybridization_df.targetId))
            if hybridization_df.targetValue[ix] == "array"
                # If problem is defined in Julia, the array input is stored elsewhere,
                # which the _array_julia value flags
                if isempty(yaml_file)
                    return [:_array_julia]
                end

                if _input_isfile(input_id, yaml_file, paths) == false
                    throw(PEtabInputError("ML model input variable '$(input_id)' is \
                        marked as an array-file input, but no matching input was found \
                        among the array files in the PEtab problem. Check the \
                        `array_files` entries in problem YAML-file"))
                end
                path = _get_input_path(input_id, yaml_file, paths)
                push!(input_values, Symbol(path))
                continue
            end
            push!(input_values, Symbol.(hybridization_df.targetValue[ix]))
            continue
        end

        # This can be triggered via recursion (condition table can have numbers)
        if is_number(input_id)
            if keep_numbers == true
                push!(input_values, input_id)
            end
            continue
        end

        # Julia user provided array input
        if input_id == :_ARRAY_INPUT
            push!(input_values, input_id)
            continue
        end

        if input_id in petab_parameters.parameter_id
            push!(input_values, input_id)
            continue
        end

        if input_id in Symbol.(hybridization_df.targetId)
            ix = findfirst(x -> x == input_id, Symbol.(hybridization_df.targetId))
            push!(input_values, Symbol.(hybridization_df.targetValue[ix]))
            continue
        end

        # When input is assigned via the conditions table. Recursion needed to find the
        # the potential parameter assigning the input
        if input_id in propertynames(conditions_df)
            for condition_value in Symbol.(conditions_df[!, input_id])
                _input_values = _get_ml_model_input_values(
                    [condition_value], ml_id, ml_model, conditions_df, petab_tables,
                    paths, petab_parameters, sys; keep_numbers = keep_numbers
                )
                input_values = vcat(input_values, _input_values)
            end
            continue
        end

        throw(PEtabInputError("Input $(input_id) to neural-network cannot be found \
            among ODE variables, PEtab parameters, array files or in the conditions table"))
    end
    return input_values
end
