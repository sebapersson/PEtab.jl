"""
    PEtabModel(path_yaml::String;
               build_julia_files::Bool=false,
               verbose::Bool=true,
               ifelse_to_event::Bool=true,
               write_to_file::Bool=true,
               jlfile_path::String="")::PEtabModel

Create a PEtabModel from a PEtab specified problem with a YAML-file located at `path_yaml`.

When parsing a PEtab problem, several things happen under the hood:

1. The SBML file is translated into `ModelingToolkit.jl` format to allow for symbolic computations of the ODE-model Jacobian. Piecewise and model events are further written into `DifferentialEquations.jl` callbacks.
2. The observable PEtab table is translated into a Julia file with functions for computing the observable (`h`), noise parameter (`σ`), and initial values (`u0`).
3. To allow gradients via adjoint sensitivity analysis and/or forward sensitivity equations, the gradients of `h` and `σ` are computed symbolically with respect to the ODE model's states (`u`) and parameters (`odeProblem.p`).

All of this happens automatically, and resulting files are stored under `petab_model.dir_julia` assuming write_to_file=true. To save time, `forceBuildJlFiles=false` by default, which means that Julia files are not rebuilt if they already exist.

In case a Julia model files is provided instead of a SBML file provide file path under `jlfile_path`.

# Arguments
- `path_yaml::String`: Path to the PEtab problem YAML file.
- `build_julia_files::Bool=false`: If `true`, forces the creation of Julia files for the problem even if they already exist.
- `verbose::Bool=true`: If `true`, displays verbose output during parsing.
- `ifelse_to_event::Bool=true`: If `true`, rewrites `if-else` statements in the SBML model as event-based callbacks.
- `jlfile_path::String=""`: Path to an existing Julia file. Should only be provided if a Julia model file is availble.

# Example
```julia
petab_model = PEtabModel("path_to_petab_problem_yaml")
```
"""
function PEtabModel(path_yaml::String;
                    build_julia_files::Bool=false,
                    verbose::Bool=true,
                    ifelse_to_event::Bool=true,
                    jlfile_path::String="",
                    custom_parameter_values::Union{Nothing, Dict}=nothing, 
                    write_to_file::Bool=true)::PEtabModel

    jlFile = isempty(jlfile_path) ? false : true

    path_SBML, path_parameters, path_conditions, path_observables, path_measurements, dir_julia, dir_model, model_name = readPEtabYamlFile(path_yaml, jlFile=jlFile)

    verbose == true && @info "Building PEtabModel for $model_name"

    if jlFile == false
        pathModelJlFile = joinpath(dir_julia, model_name * ".jl")
        if !isfile(pathModelJlFile) || build_julia_files == true
            verbose == true && printstyled("[ Info:", color=123, bold=true)
            verbose == true && build_julia_files && print(" By user option rebuilds Julia ODE model ...")
            verbose == true && !build_julia_files && print(" Building Julia model file as it does not exist ...")

            bBuild = @elapsed modelDict, modelStr = XmlToModellingToolkit(path_SBML, pathModelJlFile, model_name, 
                ifelse_to_event=ifelse_to_event, write_to_file=write_to_file)
            verbose == true && @printf(" done. Time = %.1es\n", bBuild)
        end

        if isfile(pathModelJlFile) && build_julia_files == false
            verbose == true && printstyled("[ Info:", color=123, bold=true)
            verbose == true && print(" Julia model file exists and will not be rebuilt\n")
            modelStr = getFunctionsAsString(pathModelJlFile, 1)[1]
        end
    end

    if jlFile == true
        if isempty(jlfile_path) || !isfile(jlfile_path)
            @error "In case jlFile=true you must provide the path to a valid Julia model model. $jlfile_path is not a valid file"
        end

        modelDict, pathModelJlFile, modelStr = JLToModellingToolkit(jlfile_path, dir_julia, model_name, 
            ifelse_to_event=ifelse_to_event, write_to_file=write_to_file)
    end

    modelStr = addParameterForConditionSpecificInitialValues(modelStr, path_conditions, path_parameters, pathModelJlFile, write_to_file)

    # For down the line processing model dict is required 
    if !@isdefined(modelDict)
        modelDict, _ = XmlToModellingToolkit(path_SBML, pathModelJlFile, model_name, write_to_file=false, 
            onlyGetSBMLDict=true, ifelse_to_event=ifelse_to_event)
    end

    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Symbolically processes ODE-system ...")
    timeTake = @elapsed begin
        _getODESystem = @RuntimeGeneratedFunction(Meta.parse(modelStr))
        _odeSystem, state_map, parameter_map = _getODESystem("https://xkcd.com/303/") # Argument needed by @RuntimeGeneratedFunction
        if "algebraicRules" ∉ keys(modelDict) || isempty(modelDict["algebraicRules"])
            odeSystem = structural_simplify(_odeSystem)
        # DAE requires special processing
        else
            odeSystem = structural_simplify(dae_index_lowering(_odeSystem))
        end
        parameter_names = parameters(odeSystem)
        state_names = states(odeSystem)
    end
    verbose == true && @printf(" done. Time = %.1es\n", timeTake)

    path_u0_h_sigma = joinpath(dir_julia, model_name * "_h_sd_u0.jl")
    if !isfile(path_u0_h_sigma) || build_julia_files == true
        verbose == true && printstyled("[ Info:", color=123, bold=true)
        verbose == true && !isfile(path_u0_h_sigma) && print(" Building u0, h and σ file as it does not exist ...")
        verbose == true && isfile(path_u0_h_sigma) && print(" By user option rebuilds u0, h and σ file ...")
        if !@isdefined(modelDict)
            modelDict, _ = XmlToModellingToolkit(path_SBML, pathModelJlFile, model_name, write_to_file=false, 
                onlyGetSBMLDict=true, ifelse_to_event=ifelse_to_event)
        end
        bBuild = @elapsed hStr, u0!Str, u0Str, σStr = create_σ_h_u0_File(model_name, path_yaml, dir_julia, odeSystem, 
            parameter_map, state_map, modelDict, jlFile=jlFile, custom_parameter_values=custom_parameter_values, write_to_file=write_to_file)
        verbose == true && @printf(" done. Time = %.1es\n", bBuild)
    else
        verbose == true && printstyled("[ Info:", color=123, bold=true)
        verbose == true && print(" u0, h and σ file exists and will not be rebuilt\n")
        hStr, u0!Str, u0Str, σStr = getFunctionsAsString(path_u0_h_sigma, 4)
    end
    compute_h = @RuntimeGeneratedFunction(Meta.parse(hStr))
    compute_u0! = @RuntimeGeneratedFunction(Meta.parse(u0!Str))
    compute_u0 = @RuntimeGeneratedFunction(Meta.parse(u0Str))
    compute_σ = @RuntimeGeneratedFunction(Meta.parse(σStr))
    

    path_D_h_sd = joinpath(dir_julia, model_name * "_D_h_sd.jl")
    if !isfile(path_D_h_sd) || build_julia_files == true
        verbose == true && printstyled("[ Info:", color=123, bold=true)
        verbose == true && !isfile(path_u0_h_sigma) && print(" Building ∂h∂p, ∂h∂u, ∂σ∂p and ∂σ∂u file as it does not exist ...")
        verbose == true && isfile(path_u0_h_sigma) && print(" By user option rebuilds ∂h∂p, ∂h∂u, ∂σ∂p and ∂σ∂u file ...")
        if !@isdefined(modelDict)
            modelDict, _ = XmlToModellingToolkit(path_SBML, pathModelJlFile, model_name, write_to_file=false, 
                onlyGetSBMLDict=true, ifelse_to_event=ifelse_to_event)
        end
        bBuild = @elapsed ∂h∂uStr, ∂h∂pStr, ∂σ∂uStr, ∂σ∂pStr = createDerivative_σ_h_File(model_name, path_yaml, dir_julia, odeSystem, parameter_map, state_map, modelDict, jlFile=jlFile, custom_parameter_values=custom_parameter_values, write_to_file=write_to_file)
        verbose == true && @printf(" done. Time = %.1es\n", bBuild)
    else verbose == true
        verbose == true && printstyled("[ Info:", color=123, bold=true)
        verbose == true && print(" ∂h∂p, ∂h∂u, ∂σ∂p and ∂σ∂u file exists and will not be rebuilt\n")
        ∂h∂uStr, ∂h∂pStr, ∂σ∂uStr, ∂σ∂pStr = getFunctionsAsString(path_D_h_sd, 4)
    end
    compute_∂h∂u! = @RuntimeGeneratedFunction(Meta.parse(∂h∂uStr))
    compute_∂h∂p! = @RuntimeGeneratedFunction(Meta.parse(∂h∂pStr))
    compute_∂σ∂σu! = @RuntimeGeneratedFunction(Meta.parse(∂σ∂uStr))
    compute_∂σ∂σp! = @RuntimeGeneratedFunction(Meta.parse(∂σ∂pStr))

    pathCallback = joinpath(dir_julia, model_name * "_callbacks.jl")
    if !isfile(pathCallback) || build_julia_files == true
        verbose == true && printstyled("[ Info:", color=123, bold=true)
        verbose == true && !isfile(pathCallback) && print(" Building callback file as it does not exist ...")
        verbose == true && isfile(pathCallback) && print(" By user option rebuilds callback file ...")
        if !@isdefined(modelDict)
            modelDict, _ = XmlToModellingToolkit(path_SBML, pathModelJlFile, model_name, write_to_file=false, 
                onlyGetSBMLDict=true, ifelse_to_event=ifelse_to_event)
        end
        bBuild = @elapsed callbackStr, tstopsStr = createCallbacksForTimeDepedentPiecewise(odeSystem, parameter_map, 
            state_map, modelDict, model_name, path_yaml, dir_julia, jlFile = jlFile, 
            custom_parameter_values=custom_parameter_values, write_to_file=write_to_file)
        verbose == true && @printf(" done. Time = %.1es\n", bBuild)
    else
        verbose == true && printstyled("[ Info:", color=123, bold=true)
        verbose == true && print(" Callback file exists and will not be rebuilt\n")
        callbackStr, tstopsStr = getFunctionsAsString(pathCallback, 2)
    end
    getCallbackFunction = @RuntimeGeneratedFunction(Meta.parse(callbackStr))
    cbSet, checkCbActive, convert_tspan = getCallbackFunction("https://xkcd.com/2694/") # Argument needed by @RuntimeGeneratedFunction
    computeTstops = @RuntimeGeneratedFunction(Meta.parse(tstopsStr))

    petab_model = PEtabModel(model_name,
                            compute_h,
                            compute_u0!,
                            compute_u0,
                            compute_σ,
                            compute_∂h∂u!,
                            compute_∂σ∂σu!,
                            compute_∂h∂p!,
                            compute_∂σ∂σp!,
                            computeTstops,
                            convert_tspan,
                            odeSystem,
                            parameter_map,
                            state_map,
                            parameter_names,
                            state_names,
                            dir_model,
                            dir_julia,
                            CSV.File(path_measurements, stringtype=String),
                            CSV.File(path_conditions, stringtype=String),
                            CSV.File(path_observables, stringtype=String),
                            CSV.File(path_parameters, stringtype=String),
                            path_SBML,
                            path_yaml,
                            cbSet,
                            checkCbActive)

    return petab_model
end


# For reading the run-time generated PEtab-related functions which via Meta.parse are passed
# on to @RuntimeGeneratedFunction to build the PEtab related functions without world-problems.
function getFunctionsAsString(filePath::AbstractString, nFunctions::Int64)::Vector{String}

    fStart, fEnd = zeros(Int64, nFunctions), zeros(Int64, nFunctions)
    iFunction = 1
    inFunction::Bool = false
    nLines = open(filePath, "r") do f countlines(f) end
    bodyStr = Vector{String}(undef, nLines)

    f = open(filePath, "r")
    for (iLine, line) in pairs(readlines(f))

        if length(line) ≥ 8 && line[1:8] == "function"
            fStart[iFunction] = iLine
            inFunction = true
        end

        if length(line) ≥ 3 && line[1] != '#' && line[1:3] == "end"
            fEnd[iFunction] = iLine
            inFunction = false
            iFunction += 1
        end

        bodyStr[iLine] = string(line)
    end
    close(f)

    out = Vector{String}(undef, nFunctions)
    for i in eachindex(out)

        # Runtime generated functions requrie at least on function argument input, hence if missing we
        # add a foo argument
        if bodyStr[fStart[i]][end-1:end] == "()"
            bodyStr[fStart[i]] = bodyStr[fStart[i]][1:end-2] * "(foo)"
        end

        out[i] = prod([bodyStr[j] * '\n' for j in fStart[i]:fEnd[i]])
    end
    return out
end


# The PEtab standard allows the condition table to have headers which corresponds to states. In order for this to
# be compatible with gradient compuations we add such initial values as an additional parameter in odeProblem.p
# by overwriting the Julia-model file
function addParameterForConditionSpecificInitialValues(modelStr::String,
                                                       path_conditions::String,
                                                       path_parameters::String, 
                                                       pathJuliaFile::String,
                                                       write_to_file::Bool)

    # Load necessary data
    experimentalConditionsFile = CSV.File(path_conditions)
    parametersFile = CSV.File(path_parameters)

    # Extract return line to find names of trueParameterValues and ODESystem
    returnLine = filter(line -> occursin(r"\s*return",line), split(modelStr, "\n"))[1]
    returnLine = replace(returnLine, r"\s*return"=>"")
    returnLine = replace(returnLine, " "=>"")
    returnOutputs = split(returnLine,",")
    odeSystemName = returnOutputs[1]
    trueParameterValuesName = returnOutputs[end]

    # Extract line with  ODESystem to find name of stateArray and parameterArray
    odeLine = filter(line -> occursin(Regex(odeSystemName * "\\s*=\\s*ODESystem\\("), line), split(modelStr, "\n"))[1]    
    # Finds the offset after the parenthesis in "ODESystem("
    funStartRegex = Regex("\\bODESystem\\(\\K")
    # Matches parentheses pairs to grab the arguments of the "ODESystem(" function
    matchParenthesesRegex = Regex("\\((?:[^)(]*(?R)?)*+\\)")
    funStart = match(funStartRegex, odeLine)
    funStartPos = funStart.offset
    insideOfFun = match(matchParenthesesRegex, odeLine[funStartPos-1:end]).match
    insideOfFun = insideOfFun[2:end-1]
    insideOfFun = replace(insideOfFun, " "=>"")
    returnOutputs = split(insideOfFun,",")
    parameterArrayName = string(returnOutputs[end])
    stateArrayName = string(returnOutputs[end-1])

    # Extract state and parameter names
    state_names = getStateOrParameterNamesFromJlFunction(modelStr, stateArrayName, parameterArrayName, getStates=true)
    parameter_names = getStateOrParameterNamesFromJlFunction(modelStr, stateArrayName, parameterArrayName, getStates=false)
    
    # Check if the condition table contains states to map initial values
    colNames = string.(experimentalConditionsFile.names)
    length(colNames) == 1 && return modelStr
    iStart = colNames[2] == "conditionName" ? 3 : 2 # Sometimes PEtab file does not include column conditionName
    # Only change model file in case on of the experimental conditions map to a state (that is add an init parameter)
    if any(name -> name ∈ state_names, colNames[iStart:end]) == false
        return modelStr
    end

    # Find states and create new parameter names and values
    whichStates = (colNames[iStart:end])[findall(x -> x ∈ state_names, colNames[iStart:end])]
    newParameterNames = "__init__" .* whichStates .* "__"
    newParameterValues = Vector{String}(undef, length(newParameterNames))

    # Check if the columns for which the states are assigned contain parameters. If these parameters are not a part
    # of the ODE-system they have to be assigned to the ODE-system (since they determine an initial value they must
    # be considered dynamic parameters).
    for state in whichStates
        for rowValue in experimentalConditionsFile[Symbol(state)]
            if typeof(rowValue) <: Real
                continue
            elseif ismissing(rowValue)
                continue
            elseif isNumber(rowValue) == true || string(rowValue) ∈ parameter_names
                continue
            elseif rowValue ∈ parametersFile[:parameterId]
                # Must be a parameter which did not appear in the SBML file
                newParameterNames = vcat(newParameterNames, rowValue)
                newParameterValues = vcat(newParameterValues, "0.0")
            else
                @error "The condition table value $rowValue does not correspond to any parameter in the SBML file parameters file"
            end
        end
    end

    # Check if the function has already been rewritten
    if any(x -> x ∈ parameter_names, newParameterNames)
        return modelStr
    end

    # Update function lines with new parameters and values
    functionLineByLine = split(modelStr, '\n')
    linesAdd = 0:0
    for i in eachindex(functionLineByLine)
        lineNoWhiteSpace = replace(functionLineByLine[i], r"\s+" => "")

        # Check which lines new initial value parameters should be added to the parametersMap
        if length(lineNoWhiteSpace) ≥ 19 && lineNoWhiteSpace[1:19] == trueParameterValuesName
            linesAdd = (i+1):(i+length(newParameterNames))
        end

        # Add new parameters for ModelingToolkit.@parameters line
        if length(lineNoWhiteSpace) ≥ 27 && lineNoWhiteSpace[1:27] == "ModelingToolkit.@parameters"
            functionLineByLine[i] *= (" "*prod([str * " " for str in newParameterNames]))[1:end-1]
        end

        # Add new parameters in parameterArray
        if length(lineNoWhiteSpace) ≥ 14 && lineNoWhiteSpace[1:14] == parameterArrayName
            functionLineByLine[i] = functionLineByLine[i][1:end-1] * ", " * (" "*prod([str * ", " for str in newParameterNames]))[1:end-2] * "]"
        end

        # Move through state array
        for j in eachindex(whichStates)
            if startsWithx(lineNoWhiteSpace, whichStates[j])
                # Extract the default value
                _, defaultValue = split(lineNoWhiteSpace, "=>")
                newParameterValues[j] = defaultValue[end] == ',' ? defaultValue[1:end-1] : defaultValue[:]
                functionLineByLine[i] = "\t" * whichStates[j] * " => " * newParameterNames[j] * ","
            end
        end
    end

    functionLineByLineNew = Vector{String}(undef, length(functionLineByLine) + length(newParameterNames))
    k = 1
    for i in eachindex(functionLineByLineNew)
        if i ∈ linesAdd
            continue
        end
        functionLineByLineNew[i] = functionLineByLine[k]
        k += 1
    end
    # We need to capture default values
    functionLineByLineNew[linesAdd] .= "\t" .* newParameterNames .* " => " .* newParameterValues .* ","

    newFunctionString = functionLineByLineNew[1]
    newFunctionString *= prod(row * "\n" for row in functionLineByLineNew[2:end])
    # Write the new function to the Julia file
    if write_to_file == true
        open(pathJuliaFile, "w") do f
            write(f, newFunctionString)
            flush(f)
        end
    end

    return getFunctionsAsString(pathJuliaFile, 1)[1]
end

# Extract model state names from stateArray in the JL-file (and also parameter names)
function getStateOrParameterNamesFromJlFunction(fAsString::String, stateArrayName::String, parameterArrayName::String; getStates::Bool=false)

    functionLineByLine = split(fAsString, '\n')
    for i in eachindex(functionLineByLine)
        lineNoWhiteSpace = replace(functionLineByLine[i], " " => "")
        lineNoWhiteSpace = replace(lineNoWhiteSpace, "\t" => "")

        # Add new parameters in parameterArray
        if getStates == true
            lenStateArrayName = length(stateArrayName)
            if length(lineNoWhiteSpace) ≥ lenStateArrayName && lineNoWhiteSpace[1:lenStateArrayName] == stateArrayName
                locOfArrayStart = findfirst("=[",lineNoWhiteSpace)[end]
                return split(lineNoWhiteSpace[locOfArrayStart+1:end-1], ",")
            end
        end

        if getStates == false
            lenParameterArrayName = length(parameterArrayName)
            if length(lineNoWhiteSpace) ≥ lenParameterArrayName && lineNoWhiteSpace[1:lenParameterArrayName] == parameterArrayName
                locOfArrayStart = findfirst("=[",lineNoWhiteSpace)[end]
                return split(lineNoWhiteSpace[locOfArrayStart+1:end-1], ",")
            end
        end
    end

end


# Check if a str starts with x
function startsWithx(str, x)
    if length(str) < length(x)
        return false
    end

    if str[1:length(x)] == x && str[length(x)+1] ∈ [' ', '=']
        return true
    end
    return false
end
