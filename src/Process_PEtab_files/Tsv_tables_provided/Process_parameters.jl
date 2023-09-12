"""
    processParameters(parametersFile::CSV.File)::ParameterInfo

Process the PeTab parametersFile file into a type-stable Julia struct.
"""
function processParameters(parametersFile::CSV.File; customParameterValues::Union{Nothing, Dict}=nothing)::ParametersInfo

    nParameters = length(parametersFile[:estimate])

    # Pre-allocate arrays to hold data
    lowerBound::Vector{Float64} = zeros(Float64, nParameters)
    upperBound::Vector{Float64} = zeros(Float64, nParameters)
    nominalValue::Vector{Float64} = zeros(Float64, nParameters) # Vector with Nominal value in PeTab-file
    parameterScale::Vector{Symbol} = Vector{Symbol}(undef, nParameters)
    parameterId::Vector{Symbol} = Vector{Symbol}(undef, nParameters) # Faster to do comparisons with Symbols than Strings
    estimate::Vector{Bool} = Vector{Bool}(undef, nParameters) #

    for i in eachindex(estimate)

        # If upper or lower bounds are missing assume +Inf and -Inf respectively.
        if ismissing(parametersFile[:lowerBound][i])
            lowerBound[i] = -Inf
        else
            lowerBound[i] = parametersFile[:lowerBound][i]
        end
        if ismissing(parametersFile[:upperBound][i])
            upperBound[i] = Inf
        else
            upperBound[i] = parametersFile[:upperBound][i]
        end

        nominalValue[i] = parametersFile[:nominalValue][i]
        parameterId[i] = Symbol(string(parametersFile[:parameterId][i]))
        # Currently only supports parameters on log10-scale -> TODO: Change this
        if parametersFile[:parameterScale][i] == "log10"
            parameterScale[i] = :log10
        elseif parametersFile[:parameterScale][i] == "log"
            parameterScale[i] = :log
        elseif parametersFile[:parameterScale][i] == "lin"
            parameterScale[i] = :lin
        else
            errorStr = "Parameter scale " * parametersFile[:parameterScale][i] * "not supported. Only log10, log and lin are supported in the Parameters PEtab file under the parameterScale column."
            throw(PEtabFileError(errorStr))
        end

        estimate[i] = parametersFile[:estimate][i] == 1 ? true : false

        # In some case when working with the model the user might want to change model parameters but not go the entire
        # way to the PEtab-files. This ensure ParametersInfo gets its parameters correct.
        if isnothing(customParameterValues)
            continue
        end
        keysDict = collect(keys(customParameterValues))
        iKey = findfirst(x -> x == parameterId[i], keysDict)
        isnothing(iKey) && continue
        valueChangeTo = customParameterValues[keysDict[iKey]]
        if typeof(valueChangeTo) <: Real
            estimate[i] = false
            nominalValue[i] = valueChangeTo
        elseif isNumber(valueChangeTo)
            estimate[i] = false
            nominalValue[i] = parse(Float64, valueChangeTo)
        elseif valueChangeTo == "estimate"
            estimate[i] = true
        else
            PEtabFileError("For PEtab select a parameter must be set to either a estimate or a number not $valueChangeTo")
        end
    end

    nParametersToEstimate::Int64 = Int64(sum(estimate))
    return ParametersInfo(nominalValue, lowerBound, upperBound, parameterId, parameterScale, estimate, nParametersToEstimate)
end


function processPriors(θ_indices::ParameterIndices, parametersFile::CSV.File)::PriorInfo

    # In case there are no model priors
    if :objectivePriorType ∉ parametersFile.names
        return PriorInfo(Dict{Symbol, Function}(), Dict{Symbol, Bool}(), false)
    end

    priorLogpdf::Dict{Symbol, Function} = Dict()
    priorOnParameterScale::Dict{Symbol, Bool} = Dict()
    for θ_name in θ_indices.θ_estNames

        whichParameter = findfirst(x -> x == string(θ_name), string.(parametersFile[:parameterId]))
        prior = parametersFile[whichParameter][:objectivePriorType]

        # In case the parameter lacks prior
        if ismissing(prior) || isempty(prior)
            priorLogpdf[θ_name] = noPrior
            priorOnParameterScale[θ_name] = false
            continue
        end

        # In case a Julia prior is provided via Catalyst importer
        if occursin("__Julia__", prior)
            priorParsed = eval(Meta.parse(parsePrior(prior)))
            priorLogpdf[θ_name] = (x) -> logpdf(priorParsed, x)
            priorOnParameterScale[θ_name] = !parametersFile[whichParameter][:priorOnLinearScale]
            continue
        end

        # In case there is a prior is has associated parameters
        priorParameters = parse.(Float64, split(parametersFile[whichParameter][:objectivePriorParameters], ";"))
        if prior == "parameterScaleNormal"
            priorLogpdf[θ_name] = (x) -> logpdf(Normal(priorParameters[1], priorParameters[2]), x)
            priorOnParameterScale[θ_name] = true

        elseif prior == "parameterScaleLaplace"
            priorLogpdf[θ_name] = (x) -> logpdf(Laplace(priorParameters[1], priorParameters[2]), x)
            priorOnParameterScale[θ_name] = true

        elseif prior == "normal"
            priorLogpdf[θ_name] = (x) -> logpdf(Normal(priorParameters[1], priorParameters[2]), x)
            priorOnParameterScale[θ_name] = false

        elseif prior == "laplace"
            priorLogpdf[θ_name] = (x) -> logpdf(Laplace(priorParameters[1], priorParameters[2]), x)
            priorOnParameterScale[θ_name] = false

        elseif prior == "logNormal"
            priorLogpdf[θ_name] = (x) -> logpdf(LogNormal(priorParameters[1], priorParameters[2]), x)
            priorOnParameterScale[θ_name] = false

        elseif prior == "logLaplace"
            @error "Error : Julia does not yet have support for log-laplace"
        else
            @error "Error : PeTab standard does not support a prior of type $priorF"
        end
    end

    return PriorInfo(priorLogpdf, priorOnParameterScale, true)
end
# Helper function in case there is not any parameter priors
function noPrior(p::Real)::Real
    return 0.0
end


# Helper funciton to parse if prior has been provided via Catalyst interface 
function parsePrior(str::String)
    _str = replace(str, "__Julia__" => "")
    strParse = ""
    insideParenthesis::Bool=false
    doNotAdd::Bool=false
    for char in _str
        if char == '('
            insideParenthesis = true
            doNotAdd = true
            strParse *= char
            continue
        end

        if insideParenthesis == true && doNotAdd == true && char == '='
            doNotAdd = false
            continue
        end

        if insideParenthesis == true && char == ','
            doNotAdd = true
            strParse *= char
            continue
        end

        if doNotAdd == true
            continue
        end
        strParse *= char 
    end
    return strParse
end
