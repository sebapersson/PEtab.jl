# Extracts the argument from a function.
# If a dictionary (with functions) is supplied, will also check if there are nested functions and will
# include the arguments of these nested functions as arguments of the first function.
# The returned string will only contain unique arguments.
function get_arguments(function_as_str, base_functions::Array{String, 1})
    parts = split(function_as_str, ['(', ')', '/', '+', '-', '*', ' ', '~', '>', '<', '=', ','], keepempty = false)
    arguments = Dict()
    for part in parts
        if isdigit(part[1])
            nothing
        else
            if (part in values(arguments)) == false && !(part in base_functions)
                arguments[length(arguments)+1] = part
            end
        end
    end
    if length(arguments) > 0
        argument_str = arguments[1]
        for i = 2:length(arguments)
            argument_str = argument_str * ", " * arguments[i]
        end
    else
        argument_str = ""
    end
    return argument_str
end
function get_arguments(function_as_str, dictionary::Dict, base_functions::Vector{String})
    parts = split(function_as_str, ['(', ')', '/', '+', '-', '*', ' ', '~', '>', '<', '=', ','], keepempty = false)
    existing_functions = keys(dictionary)
    includes_function = false
    arguments = Dict()
    for part in parts
        if isdigit(part[1])
            nothing
        else
            if part in existing_functions
                includes_function = true
                function_arguments = dictionary[part][1]
                function_arguments = split(function_arguments, [',', ' '], keepempty = false)
                for arg in function_arguments
                    if (arg in values(arguments)) == false
                        arguments[length(arguments)+1] = arg
                    end
                end
            else
                if (part in values(arguments)) == false && !(part in base_functions)
                    arguments[length(arguments)+1] = part
                end
            end
        end
    end
    if length(arguments) > 0
        argument_str = arguments[1]
        for i = 2:length(arguments)
            argument_str = argument_str * ", " * arguments[i]
        end
    else
        argument_str = ""
    end
    return [argument_str, includes_function]
end


# Replaces a word, "replace_from" in functions with another word, "replace_to".
# Often used to change "time" to "t"
# Makes sure not to change for example "time1" or "shift_time"
function replace_whole_word(old_str, replace_from, replace_to)

    replace_from_regex = Regex("(\\b" * replace_from * "\\b)")
    new_str = replace(old_str, replace_from_regex => replace_to)
    return new_str
end



# Replaces words in old_str given a dictionary replace_dict.
# In the Dict, the key  is the word to replace and the second
# value is the value to replace with.
# Makes sure to only change whole words.
function replace_whole_word_dict(old_str, replace_dict)

    new_str = old_str
    regex_replace_dict = Dict()
    for (key, value) in replace_dict
        replace_from_regex = Regex("(\\b" * key * "\\b)")
        regex_replace_dict[replace_from_regex] = "(" * value[2] * ")"
    end
    new_str = replace(new_str, regex_replace_dict...)

    return new_str

end


# Substitutes the function with the formula given by the model, but replaces
# the names of the variables in the formula with the input variable names.
# e.g. If fun(a) = a^2 then "constant * fun(b)" will be rewritten as
# "constant * b^2"
# Main goal, insert model formulas when producing the model equations.
# Example "fun1(fun2(a,b),fun3(c,d))" and Dict
# test["fun1"] = ["a,b","a^b"]
# test["fun2"] = ["a,b","a*b"]
# test["fun3"] = ["a,b","a+b"]
# Gives ((a*b)^(c+d))
function replace_function_with_formula(function_as_str, function_arguments)

    new_function_str = function_as_str
    outside_comma_regex = Regex(",(?![^()]*\\))")
    match_parentheses_regex = Regex("\\((?:[^)(]*(?R)?)*+\\)")

    for (key, value) in function_arguments
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
        while !isnothing(match(n_functions, new_function_str))
            # The string we wish to insert when the correct
            # replacement has been made.
            # Must be resetted after each pass.
            replace_str = value[2]
            # Extracts the function arguments
            function_start = match(function_start_regex, new_function_str)
            function_start_position = function_start.offset
            inside_function = match(match_parentheses_regex, new_function_str[function_start_position-1:end]).match
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
                new_function_str = replace(new_function_str, key * "(" * inside_function * ")" => "(" * replace_str * ")")
            else
                # Same as above, but skips extra parentheses around the entire power.
                new_function_str = replace(new_function_str, key * "(" * inside_function * ")" => replace_str)
            end

        end
    end

    model_functions = collect(keys(function_arguments))
    if any(occursin.(model_functions, new_function_str))
        new_function_str = replace_function_with_formula(new_function_str, function_arguments)
    end

    return new_function_str
end

# Rewrites pow(base,exponent) into (base)^(exponent), which Julia can handle
function remove_power_functions(oldStr)

    if !occursin(Regex("(\\bpow\\b)"), oldStr)
        return oldStr
    end

    power_dict = Dict()
    power_dict["pow"] = ["base, exponent","(base)^(exponent)"]
    new_str = replace_function_with_formula(oldStr, power_dict)
    return new_str

end


function get_SBML_function_args(SBML_function)::String
    if isempty(SBML_function.body.args)
        return ""
    end
    args = prod([arg * ", " for arg in SBML_function.body.args])[1:end-2]
    return args
end
