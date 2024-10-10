function _parse_table_column!(out::AbstractVector, dfcol, type)::Nothing
    @assert type in [Symbol, string, Float64, Bool]
    for (i, val) in pairs(dfcol)
        ismissing(val) && continue
        out[i] = val |> type
    end
    return nothing
end

function _parse_bound_column!(out::Vector{Float64}, dfcol, estimate::Vector{Bool})::Nothing
    for (i, val) in pairs(dfcol)
        if ismissing(val) && estimate[i] == false
            continue
        elseif ismissing(val)
            throw(PEtabInputError("If a parameter does not have a bound " *
                                  "then estimate = false for said parameter"))
        end
        out[i] = Float64(val)
    end
    return nothing
end

function _check_values_column(df::DataFrame, valid_values, colname::Symbol,
                              table::String; allow_missing::Bool = false)::Nothing
    for val in df[!, colname]
        allow_missing == true && ismissing(val) && continue
        val in valid_values && continue
        throw(PEtabFileError("Invalid value $val in $colname column in the $table table. " *
                             "Allowed values are $valid_values"))
    end
    return nothing
end

function _estimate_parameter(id::Symbol, petab_parameters::PEtabParameters)::Bool
    @unpack estimate, parameter_id = petab_parameters
    ip = findfirst(x -> x == id, parameter_id)
    return isnothing(ip) ? false : estimate[ip]
end

function _get_functions_as_str(path::String, nfunctions::Int64;
                               asstr::Bool = false)::Vector{String}
    functions = fill("", nfunctions)
    if asstr == false
        bodyfile = open(path, "r") do f
            read(f, String)
        end
        bodyfile = split(bodyfile, '\n')
    else
        bodyfile = split(path, '\n')
    end

    ifunction, istart, infunction = 1, 1, false
    for (i, line) in pairs(bodyfile)
        if length(line) ≥ 8 && line[1:8] == "function"
            istart = i
            infunction = true
        end
        if infunction && length(line) ≥ 3 && line[1:3] == "end"
            functions[ifunction] = prod(bodyfile[istart:i] .* '\n')
            infunction == false
            ifunction += 1
        end
    end
    return functions
end
