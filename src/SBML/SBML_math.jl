"""
    parse_SBML_math(math)::String

Parse a SBML math expression via recursion to a string    
"""
function parse_SBML_math(math)::String
    math_parsed, _ = _parse_SBML_math(math)
    return math_parsed
end
"""
    parse_SBML_math(math::SBML.MathApply, inequality_to_julia::Bool)::String

Parse a SBML math expression via recursion to a str with inequality rewritten to 
Julia syntax, geq(x, 2) -> x ≥ 2 (instead of keeping geq-syntax)
"""
function parse_SBML_math(math::SBML.MathApply, inequality_to_julia::Bool)::String
    math_parsed, _ = _parse_SBML_math(math; inequality_to_julia=inequality_to_julia)
    return math_parsed
end
function parse_SBML_math(math, inequality_to_julia::Bool)::String
    math_parsed, _ = _parse_SBML_math(math)
    return math_parsed
end
function parse_SBML_math(math::Nothing)::String
    return ""
end


function _parse_SBML_math(math::SBML.MathApply; inequality_to_julia::Bool=false)::Tuple{String, Bool}

    # Single argument times allowed according to MathML standard
    if math.fn == "*" && length(math.args) == 0
        return "1", false
    end

    # Single argument addition allowed according to MathML standard
    if math.fn == "+" && length(math.args) == 0
        return "0", false
    end

    if math.fn ∈ ["*", "/", "+", "-", "power"] && length(math.args) == 2
        fn = math.fn == "power" ? "^" : math.fn
        _part1, add_parenthesis1 = _parse_SBML_math(math.args[1])
        _part2, add_parenthesis2 = _parse_SBML_math(math.args[2])
        part1 = add_parenthesis1 ?  '(' * _part1 * ')' : _part1
        part2 = add_parenthesis2 ?  '(' * _part2 * ')' : _part2

        # For power always have the exponential in parenthesis for 
        # ensuring corectness
        if fn == "^"
            return part1 * fn * "(" * _part2 * ")", false
        elseif fn ∈ ["+", "-", "/"]
            return part1 * fn * part2, true
        elseif fn == "*"
            if any(occursin.(["+", "-", "/"], part1)) || any(occursin.(["+", "-", "/"], part2))
                return part1 * fn * part2, true
            else
                return part1 * fn * part2, false
            end
        end
    end

    if math.fn == "log" && length(math.args) == 2
        base, add_parenthesis1 = _parse_SBML_math(math.args[1])
        arg, add_parenthesis2 = _parse_SBML_math(math.args[2])
        part1 = add_parenthesis1 ?  '(' * base * ')' : base
        part2 = add_parenthesis2 ?  '(' * arg * ')' : arg
        return "log(" * part1 * ", " * part2 * ")", false
    end

    if math.fn == "root" && length(math.args) == 2
        base, add_parenthesis1 = _parse_SBML_math(math.args[1])
        arg, add_parenthesis2 = _parse_SBML_math(math.args[2])
        part1 = add_parenthesis1 ?  '(' * base * ')' : base
        part2 = add_parenthesis2 ?  '(' * arg * ')' : arg
        return  part2 * "^(1 / " * part1 * ")", false
    end

    if math.fn ∈ ["+", "-"] && length(math.args) == 1
        _formula, add_parenthesis = _parse_SBML_math(math.args[1])
        formula = add_parenthesis ? '(' * _formula * ')' : _formula
        return math.fn * formula, false
    end

    if math.fn == "quotient" 
        arg1, _ = _parse_SBML_math(math.args[1])
        arg2, _ = _parse_SBML_math(math.args[2])
        return "div(" * arg1 * ", " * arg2 * ")", false
    end

    # Piecewise can have arbibrary number of arguments 
    if math.fn == "piecewise"
        formula = "piecewise("
        for arg in math.args
            _formula, _ = _parse_SBML_math(arg) 
            formula *= _formula * ", "
        end
        return formula[1:end-2] * ')', false
    end

    if math.fn ∈ ["lt", "gt", "leq", "geq", "eq"] && inequality_to_julia == false
        @assert length(math.args) == 2
        part1, _ = _parse_SBML_math(math.args[1]) 
        part2, _ = _parse_SBML_math(math.args[2])
        return math.fn * "(" * part1 * ", " * part2 * ')', false
    end

    if math.fn ∈ ["lt", "gt", "leq", "geq", "eq"] && inequality_to_julia == true
        @assert length(math.args) == 2
        part1, _ = _parse_SBML_math(math.args[1]) 
        part2, _ = _parse_SBML_math(math.args[2])
        if math.fn == "lt"
            operator = "<"
        elseif math.fn == "gt"
            operator = ">"
        elseif math.fn == "geq"
            operator = "≥"
        elseif math.fn == "leq"
            operator = "≤"
        elseif math.fn == "eq"
            operator = "=="
        end

        return "(" * part1 * operator * part2 * ')', false
    end

    if math.fn ∈ ["exp", "log", "log2", "log10", "sin", "cos", "tan"]
        @assert length(math.args) == 1
        formula, _ = _parse_SBML_math(math.args[1])
        return math.fn * '(' * formula * ')', false
    end

    if math.fn ∈ ["arctan", "arcsin", "arccos", "arcsec", "arctanh", "arcsinh", "arccosh", 
                  "arccsc", "arcsech", "arccoth", "arccot", "arccot", "arccsch"]
        @assert length(math.args) == 1
        formula, _ = _parse_SBML_math(math.args[1])
        return "a" * math.fn[4:end] * '(' * formula * ')', false
    end

    if math.fn ∈ ["exp", "log", "log2", "log10", "sin", "cos", "tan", "csc", "ln"]
        fn = math.fn == "ln" ? "log" : math.fn
        @assert length(math.args) == 1
        formula, _ = _parse_SBML_math(math.args[1])
        return fn * '(' * formula * ')', false
    end

    # Special function which must be rewritten to Julia syntax 
    if math.fn == "ceiling"
        formula, _ = _parse_SBML_math(math.args[1])
        return "ceil" * '(' * formula * ')', false
    end

    # Factorials are, naturally, very challenging for ODE solvers. In case against the odds they 
    # are provided we compute the factorial via the gamma-function (to handle Num type). 
    if math.fn == "factorial"
        @warn "Factorial in the ODE model. PEtab.jl can handle factorials, but, solving the ODEs with factorial is 
            numerically challenging, and thus if possible should be avioded"
        formula, _ = _parse_SBML_math(math.args[1])
        return "SpecialFunctions.gamma" * '(' * formula * " + 1.0)", false
    end

    # At this point the only feasible option left is a SBML_function
    formula = math.fn * '('
    if length(math.args) == 0
        return formula * ')', false
    end
    for arg in math.args
        _formula, _ = _parse_SBML_math(arg) 
        formula *= _formula * ", "
    end
    return formula[1:end-2] * ')', false
end
function _parse_SBML_math(math::SBML.MathVal)::Tuple{String, Bool}
    return string(math.val), false
end
function _parse_SBML_math(math::SBML.MathIdent)::Tuple{String, Bool}
    return string(math.id), false
end
function _parse_SBML_math(math::SBML.MathTime)::Tuple{String, Bool}
    # Time unit is consistently in models refered to as time 
    return "t", false
end
function _parse_SBML_math(math::SBML.MathAvogadro)::Tuple{String, Bool}
    # Time unit is consistently in models refered to as time 
    return "6.02214179e23", false
end
function _parse_SBML_math(math::SBML.MathConst)::Tuple{String, Bool}
    if math.id == "exponentiale"
        return "2.718281828459045", false
    elseif math.id == "pi"
        return "3.1415926535897", false
    else
        return math.id, false
    end
end