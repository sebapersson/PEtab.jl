
function _parse_table_column!(out::AbstractVector, dfcol, type)::Nothing
    @assert type in [Symbol, string, Float64, Bool]
    for (i, val) in pairs(dfcol)
        ismissing(val) && continue
        out[i] = val |> type
    end
    return nothing
end

function _check_values_column(df::DataFrame, valid_values, colname::Symbol, table::String)::Nothing
    for val in df[!, colname]
        val in valid_values && continue
        throw(PEtabFileError("Invalid value $val in $colname column in the $table table. " *
                             "Allowed values are $valid_values"))
    end
    return nothing
end

function _estimate_parameter(id::Symbol, parameter_info::ParametersInfo)::Bool
    @unpack estimate, parameter_id = parameter_info
    return estimate[findfirst(x -> x == id, parameter_id)]
end
