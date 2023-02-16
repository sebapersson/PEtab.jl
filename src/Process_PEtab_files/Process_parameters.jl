"""
    processParameters(parametersFile::DataFrame)::ParameterInfo

    Process the PeTab parametersFile file into a type-stable Julia struct.
"""
function processParameters(parametersFile::DataFrame)::ParametersInfo

    nParameters = length(parametersFile[!, "estimate"])

    # Pre-allocate arrays to hold data
    lowerBound::Vector{Float64} = zeros(Float64, nParameters)
    upperBound::Vector{Float64} = zeros(Float64, nParameters)
    nominalValue::Vector{Float64} = zeros(Float64, nParameters) # Vector with Nominal value in PeTab-file
    parameterScale::Vector{Symbol} = Vector{Symbol}(undef, nParameters)
    parameterId::Vector{Symbol} = Vector{Symbol}(undef, nParameters) # Faster to do comparisons with Symbols than Strings
    estimate::Vector{Bool} = Vector{Bool}(undef, nParameters) #

    for i in eachindex(estimate)

        # If upper or lower bounds are missing assume +Inf and -Inf respectively.
        if ismissing(parametersFile[i, "lowerBound"])
            lowerBound[i] = -Inf
        else
            lowerBound[i] = parametersFile[i, "lowerBound"]
        end
        if ismissing(parametersFile[i, "upperBound"])
            upperBound[i] = Inf
        else
            upperBound[i] = parametersFile[i, "upperBound"]
        end

        nominalValue[i] = parametersFile[i, "nominalValue"]
        parameterId[i] = Symbol(string(parametersFile[i, "parameterId"]))
        # Currently only supports parameters on log10-scale -> TODO: Change this
        if parametersFile[i, "parameterScale"] == "log10"
            parameterScale[i] = :log10
        elseif parametersFile[i, "parameterScale"] == "log"
            parameterScale[i] = :log
        elseif parametersFile[i, "parameterScale"] == "lin"
            parameterScale[i] = :lin
        else
            errorStr = "Parameter scale " * parametersFile[i, "parameterScale"] * "not supported. Only log10, log and lin are supported in the Parameters PEtab file under the parameterScale column."
            throw(PEtabFileError(errorStr))
        end

        estimate[i] = parametersFile[i, "estimate"] == 1 ? true : false
    end
    nParametersToEstimate::Int64 = Int64(sum(estimate))

    return ParametersInfo(nominalValue, lowerBound, upperBound, parameterId, parameterScale, estimate, nParametersToEstimate)
end


function processPriors(θ_indices::ParameterIndices, parametersFile::DataFrame)::PriorInfo

    # In case there are no model priors
    if "objectivePriorType" ∉ names(parametersFile)
        return PriorInfo(NamedTuple(), NamedTuple(), false)
    end

    # To track Priors we employ NamedTuples
    θ_estNames = string.(θ_indices.θ_estNames)
    priorLogpdf = Vector{Function}(undef, length(θ_estNames))
    priorOnParameterScale = Vector{Bool}(undef, length(θ_estNames))

    for i in eachindex(θ_estNames)

        whichParameter = findfirst(x -> x == θ_estNames[i], string.(parametersFile[!, "parameterId"]))
        prior = parametersFile[whichParameter, "objectivePriorType"]

        # In case the parameter lacks prior
        if ismissing(prior)
            priorLogpdf[i] = noPrior
            priorOnParameterScale[i] = false
            continue
        end

        # In case there is a prior is has associated parameters
        priorParameters = parse.(Float64, split(parametersFile[whichParameter, "objectivePriorParameters"], ";"))
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
