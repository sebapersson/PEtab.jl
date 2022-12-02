"""
    isNumber(x::String)::Bool

    Check if a string x is a number (Float).
"""
function isNumber(x::AbstractString)::Bool
    re1 = r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)$" # Picks up scientific notation
    re2 = r"^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$"
    return (occursin(re1, x) || occursin(re2, x))
end
"""
    isNumber(x::SubString{String})::Bool
"""
function isNumber(x::SubString{String})::Bool
    re1 = r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)$" # Picks up scientific notation
    re2 = r"^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$"
    return (occursin(re1, x) || occursin(re2, x))
end


"""
    transformYobsOrYmodArr!(vals, transformationArr::Array{Symbol, 1})

    Transform the Yobs or Ymod arrays (vals) in place using for each value 
    in vals the transformation specifed in transformationArr.

    Currently :lin, :log10 and :log transforamtions are supported, see
    `setUpCostFunc`.
"""
function transformYobsOrYmodArr!(vals, transformationArr::Array{Symbol, 1})
    for i in eachindex(vals)
        vals[i] = transformObsOrData(vals[i], transformationArr[i])
    end
end


"""
    transformYobsOrYmodArr!(vals, transformationArr::Array{Symbol, 1})

    Transform val using either :lin (identify), :log10 and :log transforamtions.
"""
function transformObsOrData(val, transform::Symbol)
    if transform == :lin
        return val
    elseif transform == :log10
        return val > 0 ? log10(val) : Inf
    elseif transform == :log
        return val > 0 ? log(val) : Inf
    else
        println("Error : $transform is not an allowed transformation")
        println("Only :lin, :log10 and :log are supported.")
    end
end
