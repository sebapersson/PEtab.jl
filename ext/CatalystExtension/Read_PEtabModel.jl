"""
    readPEtabModel(system::ReactionSystem,
                   simulationConditions::Dict{String, Dict},
                   observables::Dict{String, PEtab.PEtabObservable},
                   measurements::DataFrame,
                   petabParameters::Vector{PEtab.PEtabParameter};
                   stateMap::Union{Nothing, Vector{Pair}=nothing,
                   parameterMap::Union{Nothing, Vector{Pair}=nothing,
                   verbose::Bool=false)::PEtabModel

Create a PEtabModel directly in Julia from a Catalyst reaction system.

For additional information on the input format, see the main documentation. 

# Arguments 
- `system::ReactionSystem`: A Catalyst reaction system.
- `simulationConditions::Dict{String, T}`: A dictionary specifying values for control parameters/species per simulation condition.
- `observables::Dict{String, PEtab.PEtabObservable}`: A dictionary specifying the observable and noise formulas linking the model to data.
- `measurements::DataFrame`: Measurement data to calibrate the model against.
- `petabParameters::Vector{PEtab.PEtabParameter}`: Parameters to estimate in PEtabParameter format.
- `stateMap=nothing`: An optional state-map to set initial species values to be constant across all simulation conditions.
- `parameterMap=nothing`: An optional state-map to set parameter values to be constant across all simulation conditions.
- `verbose::Bool=false`: Whether to print progress when building the model.

# Example
```julia
# Define a reaction network model 
rn = @reaction_network begin
    @parameters a0 b0
    @species A(t)=a0 B(t)=b0
    (k1, k2), A <--> B
end

# Measurement data 
measurements = DataFrame(
    simulation_id=["c0", "c0"],
    obs_id=["obs_a", "obs_a"],
    time=[0, 10.0],
    measurement=[0.7, 0.1],
    noise_parameters=0.5
)

# Single experimental condition                          
simulation_conditions = Dict("c0" => Dict())

# PEtab-parameters to estimate
petab_parameters = [
    PEtabParameter(:a0, value=1.0, scale=:lin),
    PEtabParameter(:b0, value=0.0, scale=:lin),
    PEtabParameter(:k1, value=0.8, scale=:lin),
    PEtabParameter(:k2, value=0.6, scale=:lin)
]

# Observable equation                     
@unpack A = rn
observables = Dict("obs_a" => PEtabObservable(A, 0.5))

# Create a PEtabODEProblem 
petab_model = readPEtabModel(
    rn, simulation_conditions, observables, measurements,
    petab_parameters, verbose=false
)
```
"""
function PEtab.readPEtabModel(system::ReactionSystem,
                              simulationConditions::Dict{String, T},
                              observables::Dict{String, PEtab.PEtabObservable},
                              measurements::DataFrame,
                              petabParameters::Vector{PEtab.PEtabParameter};
                              stateMap::Union{Nothing, Vector{Pair{T1, Float64}}}=nothing,
                              parameterMap::Union{Nothing, Vector{Pair{T2, Float64}}}=nothing,
                              verbose::Bool=false)::PEtab.PEtabModel where {T1<:Union{Symbol, Num}, T2<:Union{Symbol, Num}, T<:Dict}

    modelName = "ReactionSystemModel"
    verbose == true && @info "Building PEtabModel for $modelName"

    # Extract model parameters and names
    parameterNames = parameters(system)
    stateNames = states(system)

    # Extract relevant PEtab-files, convert to CSV.File
    measurementsData = PEtab.parsePEtabMeasurements(measurements, observables, simulationConditions, petabParameters) |> PEtab.dataFrameToCSVFile
    parametersData = PEtab.parsePEtabParameters(petabParameters, system, simulationConditions, observables, measurements) |> PEtab.dataFrameToCSVFile
    observablesData = PEtab.parsePEtabObservable(observables) |> PEtab.dataFrameToCSVFile
    experimentalConditions = PEtab.parsePEtabExperimentalCondition(simulationConditions, petabParameters, system) |> PEtab.dataFrameToCSVFile
    

    # Build the initial value map (initial values as parameters are set in the reaction system)
    stateMap = PEtab.updateStateMap(stateMap, system, experimentalConditions) # Parameters in condition table
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
    hStr, u0!Str, u0Str, σStr = PEtab.create_σ_h_u0_File(modelName, system, experimentalConditions, measurementsData,
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
    ∂h∂uStr, ∂h∂pStr, ∂σ∂uStr, ∂σ∂pStr = PEtab.createDerivative_σ_h_File(modelName, system, experimentalConditions,
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

    _parameterMap = [Num(p) => 0.0 for p in parameters(system)]
    for i in eachindex(_parameterMap)
        if isnothing(parameterMap)
            continue
        end
        for j in eachindex(parameterMap)
            if string(_parameterMap[i].first) != string(parameterMap[j].first)
                continue
            end
            _parameterMap[i] = _parameterMap[i].first => parameterMap[j].second
        end
    end

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
                            _parameterMap,
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


function PEtab.addModelParameter!(system::ReactionSystem, newParameter)
    eval(Meta.parse("@parameters " * newParameter))
    Catalyst.addparam!(system, eval(Meta.parse(newParameter)))
end
