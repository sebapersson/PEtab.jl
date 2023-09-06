using ModelingToolkit
using Catalyst
using OrdinaryDiffEq
using Distributions
using DataFrames
using Plots
using PEtab
using CSV
using Tables
using Printf


import PEtab: getObsOrSdParam


# The conditiosn for an experiment.
struct PEtabExperimentalCondition
    parameter_values::Dict{Union{Num, Symbol}, Union{Float64, Symbol}}
end
# An observable value.
struct PEtabObservable
    obs::Num
    transformation::Union{Nothing, Symbol}
    noiseFormula::Union{Nothing, Num}
end
# A parameter.
struct PEtabParameter
    parameter::Union{Num, Symbol}
    estimate::Bool
    value::Union{Nothing,Float64}
    lb::Union{Nothing,Float64}
    ub::Union{Nothing,Float64}
    prior::Union{Nothing,Distribution{Univariate, Continuous}}
    scale::Union{Nothing,Symbol} # :log10, :linear and :log supported.
end
function PEtabParameter(parameter::Union{Num, Symbol};
                        estimate::Bool=true,
                        value::Union{Nothing, Float64}=nothing,
                        lb::Union{Nothing, Float64}=1e-3,
                        ub::Union{Nothing, Float64}=1e3,
                        prior::Union{Nothing,Distribution{Univariate, Continuous}}=nothing,
                        scale::Union{Nothing, Symbol}=:log10)

    return PEtabParameter(parameter, estimate, value, lb, ub, prior, scale)
end

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

# ------------------------------------------------------------------
# Functions for making reactions systems talk with everything else
# ------------------------------------------------------------------


# Parse PEtabParameter into PEtab-parameters file. In case of nothing a default value is set
# for a specific row.
function parsePEtabParameters(petab_parameters::Vector{PEtabParameter})::DataFrame
    df = DataFrame()
    for parameter in petab_parameters
        parameterId = string(parameter.parameter)

        # Often performance is improvied by estimating parameters on log-scale
        if isnothing(parameter.scale)
            parameterScale = "log10"
        else
            parameterScale = string(parameter.scale)
            @assert parameterScale ∈ ["lin", "log", "log10"]
        end

        # Default upper and lowers bounds are [1e-3, 1e3]
        if isnothing(parameter.lb)
            lowerBound = 1e-3
        else
            lowerBound = parameter.lb
        end
        if isnothing(parameter.ub)
            upperBound = 1e3
        else
            upperBound = parameter.ub
        end

        # Symbolically setting nominal value if not given
        if isnothing(parameter.value)
            nominalValue = (lowerBound + upperBound) / 2.0
        else
            nominalValue = parameter.value
        end

        estimate = parameter.estimate == true ? 1 : 0

        row = DataFrame(parameterId = parameterId,
                        parameterScale = parameterScale,
                        lowerBound = lowerBound,
                        upperBound = upperBound,
                        nominalValue = nominalValue,
                        estimate = estimate)
        append!(df, row)
    end
    return df
end


# Parse PEtabExperimentalCondition into PEtab conditions file. A lot of work is going to be needed
# here to make sure that the user does not provide any bad values.
function parsePEtabExperimentalCondition(experimental_conditions::Dict{String,PEtabExperimentalCondition}, 
                                         system::ReactionSystem)::DataFrame

    
    df = DataFrame()
    for (conditionId, experimental_condition) in experimental_conditions
        row = DataFrame(conditionId = conditionId)
        # TODO: We should @assert here to not allow the user to provide any illegal value, but aja
        for (id, parameterOrSpeciesOrCompartment) in experimental_condition.parameter_values
            row[!, string(id)] = [parameterOrSpeciesOrCompartment]
        end
        append!(df, row)
    end

    # Check if initial value is set in condition table (must then add a parameter to the reaction-system 
    # to correctly compute gradients)
    stateNames = replace.(string.(states(system)), "(t)" => "")
    for id in names(df)
        if string(id) ∈ stateNames
            newParameter = "__init__" * string(id) * "__"
            eval(Meta.parse("@parameters " * newParameter))
            Catalyst.addparam!(system, eval(Meta.parse(newParameter)))

        end
    end

    return df
end


function updateStateMap(stateMap, system::ReactionSystem, experimentalConditions::CSV.File)
    
    # Check if initial value is set in condition table (must then add a parameter to the reaction-system 
    # to correctly compute gradients)
    stateNames = replace.(string.(states(system)), "(t)" => "")
    for id in experimentalConditions.names
        if string(id) ∉ stateNames
            continue
        end
        newParameter = "__init__" * string(id) * "__"
        if isnothing(stateMap)
            stateMap = [Symbol(id) => Symbol(newParameter)]
        else
            stateMap = vcat(stateMap, Symbol(id) => Symbol(newParameter))
        end
    end
    return stateMap
end


# The measurements will be rewritten into a DataFrame which follows the correct format
function parsePEtabMeasurements(petabMeasurements::DataFrame, 
                                observables::Dict{String,PEtabObservable})::DataFrame
    df = DataFrame()
    df[!, "observableId"] = petabMeasurements[!, "obs_id"]
    df[!, "simulationConditionId"] = petabMeasurements[!, "exp_id"]
    df[!, "measurement"] = petabMeasurements[!, "value"]
    df[!, "time"] = petabMeasurements[!, "time_point"]

    if "noise_parameter" ∉ names(petabMeasurements)
        noiseFormulas = string.([observable.noiseFormula for (_, observable) in observables])
        @assert all(PEtab.isNumber.(split(noiseFormulas, ";")))
    else
        df[!, "noiseParameters"] = petabMeasurements[!, "noise_parameter"]
    end
    if "observable_parameters" ∈ names(petabMeasurements)
        df[!, "observableParameters"] = petabMeasurements[!, "observable_parameters"]
    end
    if "pre_eq_id" in names(petabMeasurements)
        df[!, "preequilibrationConditionId"] = petabMeasurements[!, "pre_eq_id"]
    end

    return df
end


function parsePEtabObservable(observables::Dict{String,PEtabObservable})::DataFrame
    df = DataFrame()

    for (observableId, observable) in observables

        if isnothing(observable.transformation)
            transformation = "lin"
        else
            transformation = string(observable.transformation)
            @assert transformation ∈ ["lin", "log", "log10"]
        end
        row = DataFrame(observableId = observableId,
                        observableFormula = replace(string(observable.obs), "(t)" => ""),
                        observableTransformation = transformation,
                        noiseFormula = observable.noiseFormula,
                        noiseDistribution = "normal")
        append!(df, row)
    end
    return df
end


function dataFrameToCSVFile(df::DataFrame)
    io = IOBuffer()
    io = CSV.write(io, df)
    str = String(take!(io))
    return CSV.File(IOBuffer(str), stringtype=String)
end


function create_σ_h_u0_File(modelName::String,
                            system::ReactionSystem,
                            experimentalConditions::CSV.File,
                            measurementsData::CSV.File,
                            parametersData::CSV.File,
                            observablesData::CSV.File,
                            stateMap)::NTuple{4, String}

    pODEProblemNames = string.(parameters(system))
    modelStateNames = replace.(string.(states(system)), "(t)" => "")
    parameterMap = [p => 0.0 for p in parameters(system)]

    parameterInfo = PEtab.processParameters(parametersData)
    measurementInfo = PEtab.processMeasurements(measurementsData, observablesData)

    # Indices for keeping track of parameters in θ
    θ_indices = PEtab.computeIndicesθ(parameterInfo, measurementInfo, system, parameterMap, stateMap, experimentalConditions)

    # Dummary variables to keep PEtab importer happy even as we are not providing any PEtab files
    SBMLDict = Dict(); SBMLDict["assignmentRulesStates"] = Dict()

    hStr = PEtab.create_h_Function(modelName, @__DIR__, modelStateNames, parameterInfo, pODEProblemNames,
                                   string.(θ_indices.θ_nonDynamicNames), observablesData, SBMLDict, false)
    u0!Str = PEtab.create_u0_Function(modelName, @__DIR__, parameterInfo, pODEProblemNames, stateMap, false,
                                      SBMLDict, inPlace=true)
    u0Str = PEtab.create_u0_Function(modelName, @__DIR__, parameterInfo, pODEProblemNames, stateMap, false,
                                     SBMLDict, inPlace=false)
    σStr = PEtab.create_σ_Function(modelName, @__DIR__, parameterInfo, modelStateNames, pODEProblemNames,
                                   string.(θ_indices.θ_nonDynamicNames), observablesData, SBMLDict, false)

    return hStr, u0!Str, u0Str, σStr
end


function createDerivative_σ_h_File(modelName::String,
                                   system::ReactionSystem,
                                   experimentalConditions::CSV.File,
                                   measurementsData::CSV.File,
                                   parametersData::CSV.File,
                                   observablesData::CSV.File,
                                   stateMap)

    pODEProblemNames = string.(parameters(system))
    modelStateNames = replace.(string.(states(system)), "(t)" => "")
    parameterMap = [p => 0.0 for p in parameters(system)]

    parameterInfo = PEtab.processParameters(parametersData)
    measurementInfo = PEtab.processMeasurements(measurementsData, observablesData)

    # Indices for keeping track of parameters in θ
    θ_indices = PEtab.computeIndicesθ(parameterInfo, measurementInfo, system, parameterMap, stateMap, experimentalConditions)

    # Dummary variables to keep PEtab importer happy even as we are not providing any PEtab files
    SBMLDict = Dict(); SBMLDict["assignmentRulesStates"] = Dict()

    ∂h∂uStr, ∂h∂pStr = PEtab.create∂h∂_Function(modelName, @__DIR__, modelStateNames, parameterInfo, pODEProblemNames, string.(θ_indices.θ_nonDynamicNames), observablesData, SBMLDict, false)
    ∂σ∂uStr, ∂σ∂pStr = PEtab.create∂σ∂_Function(modelName, @__DIR__, parameterInfo, modelStateNames, pODEProblemNames, string.(θ_indices.θ_nonDynamicNames), observablesData, SBMLDict, false)

    return ∂h∂uStr, ∂h∂pStr, ∂σ∂uStr, ∂σ∂pStr
end


function readPEtabModel(system::ReactionSystem,
                        experimental_conditions::Dict{String,PEtabExperimentalCondition},
                        observables::Dict{String,PEtabObservable},
                        meassurments::DataFrame,
                        petabParameters::Vector{PEtabParameter};
                        stateMap::Union{Nothing, Vector{Pair{T, Float64}}}=nothing,
                        verbose::Bool=false)::PEtabModel where T<:Union{Symbol, Num}

    modelName = "ReactionSystemModel"

    @info "Building PEtabModel for $modelName"

    # Extract model parameters and names
    parameterNames = parameters(system)
    stateNames = states(system)

    # Extract relevant PEtab-files, convert to CSV.File
    parametersData = parsePEtabParameters(petabParameters) |> dataFrameToCSVFile
    observablesData = parsePEtabObservable(observables) |> dataFrameToCSVFile
    experimentalConditions = parsePEtabExperimentalCondition(experimental_conditions, system) |> dataFrameToCSVFile
    measurementsData = parsePEtabMeasurements(meassurments, observables) |> dataFrameToCSVFile

    # Build the initial value map (initial values as parameters are set in the reaction system)
    stateMap = updateStateMap(stateMap, system, experimentalConditions) # Parameters in condition table
    defaultValues = Catalyst.get_defaults(system)
    _stateMap = [Symbol(replace(string(S), "(t)" => "")) => S ∈ keys(defaultValues) ? string(defaultValues[S]) : "0.0" for S in states(system)]
    if !isnothing(stateMap)
        stateMapNames = [Symbol(_S.first) for _S in stateMap]
        for (i, S) in pairs(_stateMap)
            if S.first ∉ stateMapNames
                continue
            end
            _stateMap[i] = _stateMap[i].first => string(stateMap[findfirst(x -> x == S.first, stateMapNames)].second)
        end
    end

    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building u0, h and σ functions ...")
    timeTaken = @elapsed begin
    hStr, u0!Str, u0Str, σStr = create_σ_h_u0_File(modelName, system, experimentalConditions, measurementsData,
                                                parametersData, observablesData, _stateMap)
    compute_h = @RuntimeGeneratedFunction(Meta.parse(hStr))
    compute_u0! = @RuntimeGeneratedFunction(Meta.parse(u0!Str))
    compute_u0 = @RuntimeGeneratedFunction(Meta.parse(u0Str))
    compute_σ = @RuntimeGeneratedFunction(Meta.parse(σStr))
    end
    verbose == true && @printf(" done. Time = %.1e\n", timeTaken)

    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building ∂h∂p, ∂h∂u, ∂σ∂p and ∂σ∂u functions ...")
    timeTaken = @elapsed begin
    ∂h∂uStr, ∂h∂pStr, ∂σ∂uStr, ∂σ∂pStr = createDerivative_σ_h_File(modelName, system, experimentalConditions,
                                                                   measurementsData, parametersData, observablesData,
                                                                   _stateMap)
    compute_∂h∂u! = @RuntimeGeneratedFunction(Meta.parse(∂h∂uStr))
    compute_∂h∂p! = @RuntimeGeneratedFunction(Meta.parse(∂h∂pStr))
    compute_∂σ∂σu! = @RuntimeGeneratedFunction(Meta.parse(∂σ∂uStr))
    compute_∂σ∂σp! = @RuntimeGeneratedFunction(Meta.parse(∂σ∂pStr))
    end
    verbose == true && @printf(" done. Time = %.1e\n", timeTaken)

    # For Callbacks. These function are needed by SBML generated PEtab-files, as for those we as an example rewrite
    # piecewise expressions into events
    stringWriteCallbacks = "function getCallbacks_" * modelName * "(foo)\n"
    stringWriteTstops = "\nfunction computeTstops(u::AbstractVector, p::AbstractVector)\n"
    stringWriteTstops *= "\t return Float64[]\nend\n"
    stringWriteCallbacks *= "\treturn CallbackSet(), Function[], false\nend"
    getCallbackFunction = @RuntimeGeneratedFunction(Meta.parse(stringWriteCallbacks))
    cbSet, checkCbActive, convertTspan = getCallbackFunction("https://xkcd.com/2694/") # Argument needed by @RuntimeGeneratedFunction
    computeTstops = @RuntimeGeneratedFunction(Meta.parse(stringWriteTstops))

    parameterMap = [Num(p) => 0.0 for p in parameters(system)]
    petabModel = PEtabModel(modelName,
                            compute_h,
                            compute_u0!,
                            compute_u0,
                            compute_σ,
                            compute_∂h∂u!,
                            compute_∂σ∂σu!,
                            compute_∂h∂p!,
                            compute_∂σ∂σp!,
                            computeTstops,
                            false,
                            system,
                            parameterMap,
                            _stateMap,
                            parameterNames,
                            stateNames,
                            "",
                            "",
                            measurementsData,
                            experimentalConditions,
                            observablesData,
                            parametersData,
                            "",
                            "",
                            cbSet,
                            checkCbActive)
    return petabModel
end