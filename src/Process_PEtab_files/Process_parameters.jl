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
        if !isnothing(customParameterValues)
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
    end
    nParametersToEstimate::Int64 = Int64(sum(estimate))
    return ParametersInfo(nominalValue, lowerBound, upperBound, parameterId, parameterScale, estimate, nParametersToEstimate)
end


function processPriors(θ_indices::ParameterIndices, parametersFile::CSV.File)::PriorInfo

    # In case there are no model priors
    if :objectivePriorType ∉ parametersFile.names
        return PriorInfo(NamedTuple(), NamedTuple(), false)
    end

    # To track Priors we employ NamedTuples
    θ_estNames = string.(θ_indices.θ_estNames)
    priorLogpdf = Vector{Function}(undef, length(θ_estNames))
    priorOnParameterScale = Vector{Bool}(undef, length(θ_estNames))

    for i in eachindex(θ_estNames)

        whichParameter = findfirst(x -> x == θ_estNames[i], string.(parametersFile[:parameterId]))
        prior = parametersFile[whichParameter][:objectivePriorType]

        # In case the parameter lacks prior
        if ismissing(prior)
            priorLogpdf[i] = noPrior
            priorOnParameterScale[i] = false
            continue
        end

        # In case there is a prior is has associated parameters
        priorParameters = parse.(Float64, split(parametersFile[whichParameter][:objectivePriorParameters], ";"))
        if prior == "parameterScaleNormal"
            priorLogpdf[i] = (x) -> logpdf(Normal(priorParameters[1], priorParameters[2]), x)
            priorOnParameterScale[i] = true

        elseif prior == "parameterScaleLaplace"
            priorLogpdf[i] = (x) -> logpdf(Laplace(priorParameters[1], priorParameters[2]), x)
            priorOnParameterScale[i] = true

        elseif prior == "normal"
            priorLogpdf[i] = (x) -> logpdf(Normal(priorParameters[1], priorParameters[2]), x)
            priorOnParameterScale[i] = false

        elseif prior == "laplace"
            priorLogpdf[i] = (x) -> logpdf(Laplace(priorParameters[1], priorParameters[2]), x)
            priorOnParameterScale[i] = false

        elseif prior == "logNormal"
            priorLogpdf[i] = (x) -> logpdf(LogNormal(priorParameters[1], priorParameters[2]), x)
            priorOnParameterScale[i] = false

        elseif prior == "logLaplace"
            println("Error : Julia does not yet have support for log-laplace")
        else
            println("Error : PeTab standard does not support a prior of type ", priorF)
        end
    end

    _priorLogpdf = NamedTuple{Tuple(name for name in θ_indices.θ_estNames)}(Tuple(logpdf for logpdf in priorLogpdf))
    _priorOnParameterScale = NamedTuple{Tuple(name for name in θ_indices.θ_estNames)}(Tuple(scale for scale in priorOnParameterScale))

    return PriorInfo(_priorLogpdf, _priorOnParameterScale, true)
end
# Helper function in case there is not any parameter priors
function noPrior(p::Real)::Real
    return 0.0
end
