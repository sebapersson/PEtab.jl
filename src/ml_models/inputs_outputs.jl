function get_ml_model_petab_variables(mappings_df::DataFrame, ml_model_id::Symbol, type::Symbol)::Vector{String}
    entity_col = string.(mappings_df[!, "modelEntityId"])
    if type == :outputs
        idf = startswith.(entity_col, "$(ml_model_id).outputs")
    elseif type == :inputs
        idf = startswith.(entity_col, "$(ml_model_id).inputs")
    elseif type == :parameters
        idf = startswith.(entity_col, "$(ml_model_id).parameters")
        return mappings_df[idf, :petabEntityId]
    end
    df = mappings_df[idf, :]
    # Sort to get inputs in order output1, output2, ...
    is = sortperm(string.(df[!, "modelEntityId"]),
                  by = x -> parse(Int, match(r".*\[(\d+)\]$", x).captures[1]))
    return string.(df[is, "petabEntityId"])
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
