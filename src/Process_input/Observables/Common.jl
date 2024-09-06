
"""
    petab_formula_to_Julia(formula::String, state_names, parameter_info::ParametersInfo, namesParamDyn::Vector{String}, namesNonDynParam::Vector{String})::String

    Translate a peTab formula (e.g for observable or for sd-parameter) into Julia syntax and output the result
    as a string.
"""
function petab_formula_to_Julia(formula::String,
                                model_state_names::Vector{String},
                                parameter_info::ParametersInfo,
                                xdynamic_names::Vector{String},
                                xnondynamic_names::Vector{String})::String

    # Characters directly translate to Julia and characters that also are assumed to terminate a word (e.g state
    # and parameter)
    char_directly_translate = ['(', ')', '+', '-', '/', '*', '^', ',']
    len_formula = length(formula)

    i, julia_formula = 1, ""
    while i <= len_formula
        # In case character i of the string can be translated directly
        if formula[i] in char_directly_translate
            julia_formula *= formula[i]
            i += 1

            # In case character i cannot be translated directly (is part of a word)
        else
            # Get word (e.g param, state, math-operation or number)
            word, i_new = get_word(formula, i, char_directly_translate)
            # Translate word to Julia syntax
            julia_formula *= word_to_julia(word, model_state_names, parameter_info,
                                           xdynamic_names, xnondynamic_names)
            i = i_new

            # Special case where we have multiplication
            if is_number(word) && i <= len_formula && isletter(formula[i])
                julia_formula *= "*"
            end
        end
    end

    return julia_formula
end

"""
    get_word(str::String, i_start::Int, char_terminate::Vector{Char})

In a string starting from position i_start extract the next "word", which is the longest
concurent occurance of characters that are not in the character list with word termination
characters. Returns the word and i_end (the position where the word ends).

For example, if char_terminate = ['(', ')', '+', '-', '/', '*', '^'] abc123 is
considered a word but not abc123*.
"""
function get_word(str::String, i_start::Int, char_terminate::Vector{Char})
    word_str = ""
    i_end = i_start

    # If the first character is a numberic the termination occurs when
    # the first non-numeric character (not digit or dot .) is reached.
    start_is_numeric = isnumeric(str[i_end])

    while i_end <= length(str)

        # In case str[i_end] is not a termination character we need to be careful with numbers
        # so that we handle sciencetific notations correctly, e.g we do not consider
        # 1.2e-3 to be two words [1.2e, 3] but rather a single workd.
        if !(str[i_end] in char_terminate)

            # Parase sciencetific notation for number
            if start_is_numeric == true && str[i_end] == 'e'
                if length(str) > i_end &&
                   (str[i_end + 1] == '-' || isnumeric(str[i_end + 1]))
                    if str[i_end + 1] == '-'
                        i_end += 2
                        word_str *= "e-"
                    else
                        i_end += 1
                        word_str *= "e"
                    end

                else
                    break
                end
            end

            if start_is_numeric == true && !(isnumeric(str[i_end]) || str[i_end] == '.')
                break
            end
            word_str *= str[i_end]
        else
            break
        end
        i_end += 1
    end
    # Remove all spaces from the word
    word_str = replace(word_str, " " => "")
    return word_str, i_end
end

function word_to_julia(word_translate::String,
                       model_state_names::Vector{String},
                       parameter_info::ParametersInfo,
                       xdynamic_names::Vector{String},
                       xnondynamic_names::Vector{String})::String

    # List of mathemathical operations that are accpeted and will be translated
    # into Julia syntax (t is assumed to be time)
    list_operations = ["exp", "sin", "cos", "t"]

    word_julia = ""
    # If word_translate is a constant parameter
    if word_translate ∈ string.(parameter_info.parameter_id) &&
       word_translate ∉ xdynamic_names && word_translate ∉ xnondynamic_names
        # Constant parameters get a _C appended to tell them apart
        word_julia *= word_translate * "_C"
    end

    if word_translate ∈ xdynamic_names
        word_julia *= word_translate
    end

    if word_translate ∈ xnondynamic_names
        word_julia *= word_translate
    end

    if word_translate ∈ model_state_names
        word_julia *= word_translate
    end

    if is_number(word_translate)
        # In case there is not a . in the number add to ensure function returns floats
        word_julia *= occursin('.', word_translate) ? word_translate : word_translate * ".0"
    end

    if word_translate in list_operations
        word_julia *= list_operations[word_translate .== list_operations][1]
        return word_julia # Not allowed to follow with a space
    end

    if length(word_translate) >= 19 && word_translate[1:19] == "observableParameter"
        word_julia *= word_translate
    end

    if length(word_translate) >= 14 && word_translate[1:14] == "noiseParameter"
        word_julia *= word_translate
    end

    if isempty(word_translate)
        println("Warning : When creating observation function $word_translate could not be processed")
    end

    word_julia *= " "

    return word_julia
end

"""
    get_observable_parameters(formula::String)::String

Helper function to extract all observableParameter in the observableFormula in the PEtab-file.
"""
function get_observable_parameters(formula::String)::String

    # Find all words on the form observableParameter
    _observable_parameters = sort(unique([match.match
                                          for match in eachmatch(r"observableParameter[0-9]_\w+",
                                                                 formula)]))
    observable_parameters = ""
    for i in eachindex(_observable_parameters)
        if i != length(_observable_parameters)
            observable_parameters *= _observable_parameters[i] * ", "
        else
            observable_parameters *= _observable_parameters[i]
        end
    end

    return observable_parameters
end

"""
    get_noise_parameters(formula::String)::String

Helper function to extract all the noiseParameter in noiseParameter formula in the PEtab file.
"""
function get_noise_parameters(formula::String)::String

    # Find all words on the form observableParameter
    _noise_parameters = [match.match
                         for match in eachmatch(r"noiseParameter[0-9]_\w+", formula)]
    noise_parameters = ""
    for i in eachindex(_noise_parameters)
        if i != length(_noise_parameters)
            noise_parameters *= _noise_parameters[i] * ", "
        else
            noise_parameters *= _noise_parameters[i]
        end
    end

    return noise_parameters
end

"""
    variables_to_array_index(formula,state_names,parameter_names,namesNonDynParam,parameter_info)::String

Replaces any state or parameter from formula with their corresponding index in the ODE system
Symbolics can return strings without multiplication sign, e.g. 100.0STAT5 instead of 100.0*STAT5
so replace_variable cannot be used here
"""
function variables_to_array_index(formula::String,
                                  model_state_names::Vector{String},
                                  parameter_info::ParametersInfo,
                                  pNames::Vector{String},
                                  xnondynamic_names::Vector{String};
                                  p_ode_problem::Bool = false)::String
    for (i, stateName) in pairs(model_state_names)
        formula = replace_word_number_prefix(formula, stateName, "u[" * string(i) * "]")
    end

    if p_ode_problem == true
        for (i, pName) in pairs(pNames)
            formula = replace_word_number_prefix(formula, pName,
                                                 "p_ode_problem[" * string(i) * "]")
        end
    else
        for (i, pName) in pairs(pNames)
            formula = replace_word_number_prefix(formula, pName,
                                                 "xdynamic[" * string(i) * "]")
        end
    end

    for (i, xnondynamicName) in pairs(xnondynamic_names)
        formula = replace_word_number_prefix(formula, xnondynamicName,
                                             "xnondynamic[" * string(i) * "]")
    end

    for i in eachindex(parameter_info.parameter_id)
        if parameter_info.estimate[i] == false
            formula = replace_word_number_prefix(formula,
                                                 string(parameter_info.parameter_id[i]) *
                                                 "_C",
                                                 "nominal_value[" *
                                                 string(i) * "]")
        end
    end

    return formula
end

"""
    replace_explicit_variable_rule(formula::String, model_SBML::SBMLImporter.ModelSBML)::String

Replace the explicit rule variable with the explicit rule
"""
function replace_explicit_variable_rule(formula::String,
                                        model_SBML::SBMLImporter.ModelSBML)::String
    _formula = deepcopy(formula)
    while true
        for (specie_id, specie) in model_SBML.species
            if specie.assignment_rule == false
                continue
            end
            _formula = SBMLImporter._replace_variable(_formula, specie_id,
                                                     "(" * specie.formula * ")")
        end
        _formula == formula && break
        formula = deepcopy(_formula)
    end
    while true
        if isempty(model_SBML.parameters)
            break
        end
        for (parameter_id, parameter) in model_SBML.parameters
            if parameter.assignment_rule == false
                continue
            end
            _formula = SBMLImporter._replace_variable(_formula, parameter_id,
                                                     "(" * parameter.formula * ")")
        end
        _formula == formula && break
        formula = deepcopy(_formula)
    end

    return _formula
end

"""
    replace_word_number_prefix(formula, from, to)::String

Replaces variables that can be prefixed with numbers, e.g.,
replace_word_number_prefix("4STAT5 + 100.0STAT5 + RE*STAT5 + STAT5","STAT5","u[1]")
gives 4u[1] + 100.0u[1] + RE*u[1] + u[1]
"""
function replace_word_number_prefix(old_str, replace_from, replace_to)
    replace_from_regex = Regex("\\b(\\d+\\.?\\d*+)*(" * replace_from * ")\\b")
    replace_to_regex = SubstitutionString("\\1" * replace_to)
    sleep(0.001)
    new_str = replace(old_str, replace_from_regex => replace_to_regex)
    return new_str
end
