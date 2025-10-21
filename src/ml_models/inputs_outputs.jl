function get_ml_model_petab_variables(mappings_df::DataFrame, ml_model_id::Symbol, type::Symbol)::Union{Vector{String}, Vector{Vector{String}}}
    entity_col = string.(mappings_df[!, "modelEntityId"])
    idf = startswith.(entity_col, "$(ml_model_id).$(type)")
    if type == :parameters
        return mappings_df[idf, :petabEntityId]
    end

    # Sort to get PEtab Id inputs/outputs in correct order (both within and between
    # arguments). File inputs are on the format netId.inputs[ix], while other inputs
    # are on the format netId.inputs[ix][jx]
    out = Vector{Vector{String}}(undef, 0)
    for i in 0:100
        # Check if file input, and handle separately
        str_match = "$(ml_model_id).$(type)[$(i)]" * r"$"
        matches = match.(str_match, string.(mappings_df[!, "modelEntityId"]))
        if !all(isnothing.(matches))
            @assert sum(.!isnothing(matches)) == 1 "Duplicates of \
                $(ml_model_id).$(type)[$(i)] in mapping table"
            is = findfirst(x -> !isnothing(x), matches)
            push!(out, [string.(mappings_df[is, "petabEntityId"])])
            continue
        end

        str_match = "$(ml_model_id).$(type)[$(i)]" * r"\[(\d+)\]$"
        matches = match.(str_match, string.(mappings_df[!, "modelEntityId"]))
        all(isnothing.(matches)) && break
        df = mappings_df[.!isnothing.(matches), :]
        is = sortperm(string.(df[!, "modelEntityId"]), by = x -> parse(Int, match(str_match, x).captures[1]))
        if string.(df[is, "petabEntityId"]) isa Vector{String}
            push!(out, string.(df[is, "petabEntityId"]))
        else
            push!(out, [string.(df[is, "petabEntityId"])])
        end
    end
    return out
end

function _get_ml_model_input_values(input_variables::Vector{Symbol}, ml_model_id::Symbol, ml_model::MLModel, conditions_df::DataFrame, petab_tables::PEtabTables, paths::Dict{Symbol, String}, petab_parameters::PEtabParameters, sys::ModelSystem; keep_numbers::Bool = false)::Vector{Symbol}
    input_values = Symbol[]
    hybridization_df = petab_tables[:hybridization]
    for input_variable in input_variables
        # This can be triggered via recursion (condition table can have numbers)
        if is_number(input_variable)
            if keep_numbers == true
                push!(input_values, input_variable)
            end
            continue
        end

        if input_variable in petab_parameters.parameter_id
            push!(input_values, input_variable)
            continue
        end

        if input_variable in Symbol.(hybridization_df.targetId)
            ix = findfirst(x -> x == input_variable, Symbol.(hybridization_df.targetId))
            push!(input_values, Symbol.(hybridization_df.targetValue[ix]))
            continue
        end

        # When input is assigned via the conditions table. Recursion needed to find the
        # the potential parameter assigning the input
        if input_variable in propertynames(conditions_df)
            for condition_value in Symbol.(conditions_df[!, input_variable])
                _input_values = _get_ml_model_input_values([condition_value], ml_model_id, ml_model, conditions_df, petab_tables, paths, petab_parameters, sys; keep_numbers = keep_numbers)
                input_values = vcat(input_values, _input_values)
            end
            continue
        end

        # If the input variable is a file, the complete path is added here, which simplifies
        # downstream processing
        if haskey(petab_tables, :yaml) && _input_isfile(input_variable, petab_tables[:yaml], paths)
            path = _get_input_file_path(input_variable, petab_tables[:yaml], paths)
            push!(input_values, Symbol(path))
            continue
        end

        throw(PEtabInputError("Input $(input_variable) to neural-network cannot be found \
            among ODE variables, PEtab parameters, array files or in the conditions table"))
    end
    return input_values
end
