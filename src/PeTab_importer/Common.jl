
# function isNumber(x::AbstractString)::Bool
#     re1 = r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)$" # Picks up scientific notation
#     re2 = r"^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$"
#     return (occursin(re1, x) || occursin(re2, x))
# end

"""
    isNumber(x::AbstractString)::Bool

Check if a string x is a number (Float).
"""
isNumber(x::AbstractString) = tryparse(Float64, x) !== nothing



"""
    transformYobsOrYmodArr!(vals, transformationArr::Vector{Symbol})

Transform the Yobs or Ymod arrays (vals) in place using for each value
in vals the transformation specifed in transformationArr.

Currently :lin, :log10 and :log transforamtions are supported, see
`setUpCostFunc`.
"""
function transformYobsOrYmodArr!(vals, transformationArr::AbstractVector{<:Symbol})
    for i in eachindex(vals)
        vals[i] = transformObsOrData(vals[i], transformationArr[i])
    end
end


"""
    transformObsOrData(val, transform::Symbol)

Transform val using either :lin (identity), :log10 and :log transforamtions.
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
