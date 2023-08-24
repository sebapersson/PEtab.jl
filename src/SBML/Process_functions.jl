# Extracts the argument from a function.
# If a dictionary (with functions) is supplied, will also check if there are nested functions and will
# include the arguments of these nested functions as arguments of the first function.
# The returned string will only contain unique arguments.
function getArguments(functionAsString, baseFunctions::Array{String, 1})
    parts = split(functionAsString, ['(', ')', '/', '+', '-', '*', ' ', '~', '>', '<', '=', ','], keepempty = false)
    arguments = Dict()
    for part in parts
        if isdigit(part[1])
            nothing
        else
            if (part in values(arguments)) == false && !(part in baseFunctions)
                arguments[length(arguments)+1] = part
            end
        end
    end
    if length(arguments) > 0
        argumentString = arguments[1]
        for i = 2:length(arguments)
            argumentString = argumentString * ", " * arguments[i]
        end
    else
        argumentString = ""
    end
    return argumentString
end
function getArguments(functionAsString, dictionary::Dict, baseFunctions::Vector{String})
    parts = split(functionAsString, ['(', ')', '/', '+', '-', '*', ' ', '~', '>', '<', '=', ','], keepempty = false)
    existingFunctions = keys(dictionary)
    includesFunction = false
    arguments = Dict()
    for part in parts
        if isdigit(part[1])
            nothing
        else
            if part in existingFunctions
                includesFunction = true
                funcArgs = dictionary[part][1]
                funcArgs = split(funcArgs, [',', ' '], keepempty = false)
                for arg in funcArgs
                    if (arg in values(arguments)) == false
                        arguments[length(arguments)+1] = arg
                    end
                end
            else
                if (part in values(arguments)) == false && !(part in baseFunctions)
                    arguments[length(arguments)+1] = part
                end
            end
        end
    end
    if length(arguments) > 0
        argumentString = arguments[1]
        for i = 2:length(arguments)
            argumentString = argumentString * ", " * arguments[i]
        end
    else
        argumentString = ""
    end
    return [argumentString, includesFunction]
end


# Replaces a word, "replaceFrom" in functions with another word, "replaceTo".
# Often used to change "time" to "t"
# Makes sure not to change for example "time1" or "shift_time"
function replaceWholeWord(oldString, replaceFrom, replaceTo)

    replaceFromRegex = Regex("(\\b" * replaceFrom * "\\b)")
    newString = replace(oldString, replaceFromRegex => replaceTo)
    return newString

end



# Replaces words in oldString given a dictionary replaceDict.
# In the Dict, the key  is the word to replace and the second
# value is the value to replace with.
# Makes sure to only change whole words.
function replaceWholeWordDict(oldString, replaceDict)

    newString = oldString
    regexReplaceDict = Dict()
    for (key,value) in replaceDict
        replaceFromRegex = Regex("(\\b" * key * "\\b)")
        regexReplaceDict[replaceFromRegex] = "(" * value[2] * ")"
    end
    newString = replace(newString, regexReplaceDict...)

    return newString

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
function replaceFunctionWithFormula(functionAsString, funcNameArgFormula)

    newFunctionsAsString = functionAsString

    for (key, value) in funcNameArgFormula
        # Find commas not surrounded by parentheses.
        # Used to split function arguments
        # If input argument are "pow(a,b),c" the list becomes ["pow(a,b)","c"]
        findOutsideCommaRegex = Regex(",(?![^()]*\\))")
        # Finds the old input arguments, removes spaces and puts them in a list
        replaceFrom = split(replace(value[1]," "=>""),findOutsideCommaRegex)

        # Finds all functions on the form "funName("
        numberOfFuns = Regex("\\b" * key * "\\(")
        # Finds the offset after the parenthesis in "funName("
        funStartRegex = Regex("\\b" * key * "\\(\\K")
        # Matches parentheses pairs to grab the arguments of the "funName(" function
        matchParenthesesRegex = Regex("\\((?:[^)(]*(?R)?)*+\\)")
        while !isnothing(match(numberOfFuns, newFunctionsAsString))
            # The string we wish to insert when the correct
            # replacement has been made.
            # Must be resetted after each pass.
            replaceStr = value[2]
            # Extracts the function arguments
            funStart = match(funStartRegex, newFunctionsAsString)
            funStartPos = funStart.offset
            insideOfFun = match(matchParenthesesRegex, newFunctionsAsString[funStartPos-1:end]).match
            insideOfFun = insideOfFun[2:end-1]
            replaceTo = split(replace(insideOfFun,", "=>","),findOutsideCommaRegex)

            # Replace each variable used in the formula with the
            # variable name used as input for the function.
            replaceDict = Dict()
            for ind in eachindex(replaceTo)
                replaceFromRegex = Regex("(\\b" * replaceFrom[ind] * "\\b)")
                replaceDict[replaceFromRegex] = '(' * replaceTo[ind] * ')'
            end
            replaceStr = replace(replaceStr, replaceDict...)

            if key != "pow"
                # Replace function(input) with formula where each variable in formula has the correct name.
                newFunctionsAsString = replace(newFunctionsAsString, key * "(" * insideOfFun * ")" => "(" * replaceStr * ")")
            else
                # Same as above, but skips extra parentheses around the entire power.
                newFunctionsAsString = replace(newFunctionsAsString, key * "(" * insideOfFun * ")" => replaceStr)
            end

        end
    end
    return newFunctionsAsString
end

# Rewrites pow(base,exponent) into (base)^(exponent), which Julia can handle
function removePowFunctions(oldStr)

    powDict = Dict()
    powDict["pow"] = ["base, exponent","(base)^(exponent)"]
    newStr = replaceFunctionWithFormula(oldStr, powDict)
    return newStr

end


function getSBMLFunctionArgs(SBMLFunction)::String
    if isempty(SBMLFunction.body.args)
        return ""
    end
    args = prod([arg * ", " for arg in SBMLFunction.body.args])[1:end-2]
    return args
end
