function parse_SBML_functions!(model_dict::Dict, model_SBML::SBML.Model)::Nothing
    
    for (function_name, SBML_function) in model_SBML.function_definitions
        
        if isnothing(SBML_function.body)
            continue
        end
        
        args = get_SBML_function_args(SBML_function)
        function_formula = parse_SBML_math(SBML_function.body.body, true)
        model_dict["SBML_functions"][function_name] = [args, function_formula]
    end
    return nothing
end


function get_SBML_function_args(SBML_function::SBML.FunctionDefinition)::String
    if isempty(SBML_function.body.args)
        return ""
    end
    args = prod([arg * ", " for arg in SBML_function.body.args])[1:end-2]
    return args
end


"""
    SBML_function_to_math(formula::T, model_functions::Dict)::T where T<:AbstractString

Substitutes any potential SBML functions recursively with math expressions, by 
inserting the function arguments into the function.

# Example

```
formula = fun1(fun2(a,b),fun3(c,d))
model_functions = Dict("fun1" => ["a,b","a^b"], 
                       "fun2" => ["a,b","a*b"], 
                       "fun3" => ["a,b","a+b"])
println(SBML_function_to_math(formula, model_functions))                       
```
```
"((a*b)^(c+d))"
```
"""
function SBML_function_to_math(formula::T, model_functions::Dict)::T where T<:AbstractString

    _formula = formula
    outside_comma_regex = Regex(",(?![^()]*\\))")
    match_parentheses_regex = Regex("\\((?:[^)(]*(?R)?)*+\\)")

    for (key, value) in model_functions
        # Find commas not surrounded by parentheses.
        # Used to split function arguments
        # If input argument are "pow(a,b),c" the list becomes ["pow(a,b)","c"]
        # Finds the old input arguments, removes spaces and puts them in a list
        replace_from = split(replace(value[1], " " => ""), outside_comma_regex)

        # Finds all functions on the form "funName("
        n_functions = Regex("\\b" * key * "\\(")
        # Finds the offset after the parenthesis in "funName("
        function_start_regex = Regex("\\b" * key * "\\(\\K")
        # Matches parentheses pairs to grab the arguments of the "funName(" function
        while !isnothing(match(n_functions, _formula))
            # The string we wish to insert when the correct
            # replacement has been made.
            # Must be resetted after each pass.
            replace_str = value[2]
            # Extracts the function arguments
            function_start = match(function_start_regex, _formula)
            function_start_position = function_start.offset
            inside_function = match(match_parentheses_regex, _formula[function_start_position-1:end]).match
            inside_function = inside_function[2:end-1]
            replace_to = split(replace(inside_function,", "=>","), outside_comma_regex)

            # Replace each variable used in the formula with the
            # variable name used as input for the function.
            replace_dict = Dict()
            for ind in eachindex(replace_to)
                replace_from_regex = Regex("(\\b" * replace_from[ind] * "\\b)")
                if !isempty(replace_from[ind])
                    replace_dict[replace_from_regex] = '(' * replace_to[ind] * ')'                        
                else
                    replace_dict[replace_from_regex] = ""
                end
            end
            replace_str = replace(replace_str, replace_dict...)

            if key != "pow"
                # Replace function(input) with formula where each variable in formula has the correct name.
                _formula = replace(_formula, key * "(" * inside_function * ")" => "(" * replace_str * ")")
            else
                # Same as above, but skips extra parentheses around the entire power.
                _formula = replace(_formula, key * "(" * inside_function * ")" => replace_str)
            end

        end
    end

    # In case of nested function recursively processes nested functions
    _model_functions = collect(keys(model_functions))
    if any(occursin.(_model_functions, _formula))
        _formula = SBML_function_to_math(_formula, model_functions)
    end

    return _formula
end
