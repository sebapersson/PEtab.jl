
# Parse PEtabParameter into PEtab-parameters file. In case of nothing a default value is set
# for a specific row.
function parsePEtabParameters(petab_parameters::Vector{PEtabParameter}, 
                              system, 
                              simulationConditions::Dict{String, T}, 
                              observables::Dict{String,PEtabObservable},
                              measurements::DataFrame)::DataFrame where T<:Dict

    # Extract any parameter that appears in the model or in simulationConditions 
    modelParameters = string.(parameters(system))
    conditionValues = unique(reduce(vcat, (string.(collect(values(dict))) for dict in values(simulationConditions))))
    nonDynamicParameters = vcat([string(obs.obs) for obs in values(observables)], [string(obs.noiseFormula) for obs in values(observables)])
    if "observable_parameters" ∈ names(measurements)
        observableParameters = reduce(vcat, split.(string.(measurements[!, "observable_parameters"]), ';'))
    else
        observableParameters = [""]
    end
    if "noise_parameters" ∈ names(measurements)
        noiseParameters = reduce(vcat, split.(string.(measurements[!, "noise_parameters"]), ';'))
    else
        noiseParameters = [""]
    end


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

        if (parameterId ∉ modelParameters && 
            parameterId ∉ conditionValues && 
            parameterId ∉ noiseParameters && 
            parameterId ∉ observableParameters && 
            !any(occursin.(parameterId, nonDynamicParameters)))

            if parameter.estimate == true
                strWrite = "Parameter $parameterId set to be estimated does not appear as a parameter in the reaction system, "
                strWrite *= "as a parameter in the simulation-conditions dict, or as a noise/observable parameter in the  "
                strWrite *= "measurement data. The parameter does not have any effect on the objective function."
                throw(PEtab.PEtabFormatError(strWrite))
            end
            strWrite = "Parameter $parameterId set to not be estimated does not appear as a parameter in the reaction system "
            strWrite *= "as a parameter in the simulation-conditions dict, or as a noise/observable parameter in the  "
            strWrite *= "measurement data. The parameter does not have any effect on the objective function."
            @warn "$strWrite"
        end

        row = DataFrame(parameterId = parameterId,
                        parameterScale = parameterScale,
                        lowerBound = lowerBound,
                        upperBound = upperBound,
                        nominalValue = nominalValue,
                        estimate = estimate)
        append!(df, row)
    end

    priorsPresent = !all([isnothing(petab_parameters[i].prior) for i in eachindex(petab_parameters)])
    if priorsPresent == false
        return df
    end

    priorList = Vector{String}(undef, length(petab_parameters))
    isOnLinearScale = Vector{Union{Bool, String}}(undef, length(petab_parameters))
    for i in eachindex(priorList)
        if isnothing(petab_parameters[i].prior)
            priorList[i] = ""
            isOnLinearScale[i] = ""
            continue
        end

        priorList[i] = "__Julia__" * string(petab_parameters[i].prior)
        isOnLinearScale[i] = petab_parameters[i].prior_on_linear_scale
    end
    df[!, :objectivePriorType] = priorList
    df[!, :priorOnLinearScale] = isOnLinearScale
    return df
end


# Parse PEtabExperimentalCondition into PEtab conditions file. A lot of work is going to be needed
# here to make sure that the user does not provide any bad values.
function parsePEtabExperimentalCondition(simulationConditions::Dict{String, T},
                                         petabParameters::Vector{PEtabParameter},
                                         system)::DataFrame where {T<:Dict}

    # Sanity check that parameters/states set for a specific experimental condition correspond
    # to states or parameters that appear in the model
    modelStateNames = replace.(string.(states(system)), "(t)" => "")
    modelParameterNames = string.(parameters(system))
    petabParameterIds = [string(petabParameter.parameter) for petabParameter in petabParameters]
    for conditionId in keys(simulationConditions)
        for id in keys(simulationConditions[conditionId])
            idStr = string(id)
            if idStr ∈ modelStateNames || idStr ∈ modelParameterNames
                continue
            end
            strWrite = "Parameter/state $idStr specifed to change between simulation-conditions (set as dictionary \n" 
            strWrite *= "key for simulation-conditions) does not as required correspond to one of the states or \n"
            strWrite *= "parameters in the model"
            throw(PEtab.PEtabFormatError(strWrite))
        end
    end

    # Check that for each condition the same states/parameters are assigned values
    if length(keys(simulationConditions)) > 1
        _keys = collect(keys(simulationConditions))
        referenceKeys = keys(simulationConditions[_keys[1]])
        for i in 2:length(keys(simulationConditions))
            keysCheck = collect(keys(simulationConditions[_keys[i]]))
            if all(sort(string.(referenceKeys)) .== sort(string.(keysCheck))) == true
                continue
            end
            strWrite = "Not all simulation conditions have the same parameter/states which are assigned. If "
            strWrite *= "for example parameter is assigned under condition c1, it must also be assigned under "
            strWrite *= "condition c2"
            throw(PEtab.PEtabFormatError(strWrite))
        end
    end

    df = DataFrame()
    for (conditionId, simulationCondition) in simulationConditions
        row = DataFrame(conditionId = conditionId)
        for (id, parameterOrSpeciesOrCompartment) in simulationCondition
            parameterOrSpeciesOrCompartmentStr = string(parameterOrSpeciesOrCompartment)
            if PEtab.isNumber(parameterOrSpeciesOrCompartmentStr)
                row[!, string(id)] = [string(parameterOrSpeciesOrCompartment)]
                continue
            end

            tmp = parameterOrSpeciesOrCompartmentStr
            if tmp ∈ modelParameterNames || tmp ∈ petabParameterIds
                row[!, string(id)] = [string(parameterOrSpeciesOrCompartment)]
                continue
            end

            strWrite = "For simulation-condition $conditionId $id is set to map to $tmp. However, as\n"
            strWrite *= " required $tmp does not correspond to a number, model parameter or a PEtabParameter."
            throw(PEtab.PEtabFormatError(strWrite))
        end
        append!(df, row)
    end

    # Check if initial value is set in condition table (must then add a parameter to the reaction-system
    # to correctly compute gradients)
    state_names = replace.(string.(states(system)), "(t)" => "")
    for id in names(df)
        if string(id) ∈ state_names
            newParameter = "__init__" * string(id) * "__"
            addModelParameter!(system, newParameter)
        end
    end

    return df
end


function addModelParameter!(system, newParameter)
    return nothing
end


function updateStateMap(state_map, system, experimentalConditions::CSV.File)

    # Check if initial value is set in condition table (must then add a parameter to the reaction-system
    # to correctly compute gradients)
    state_names = replace.(string.(states(system)), "(t)" => "")
    for id in experimentalConditions.names
        if string(id) ∉ state_names
            continue
        end
        newParameter = "__init__" * string(id) * "__"
        if isnothing(state_map)
            state_map = [Symbol(id) => Symbol(newParameter)]
        else
            state_map = vcat(state_map, Symbol(id) => Symbol(newParameter))
        end
    end
    return state_map
end


# The measurements will be rewritten into a DataFrame which follows the correct format
function parsePEtabMeasurements(petabMeasurements::DataFrame,
                                observables::Dict{String,PEtabObservable}, 
                                simulationConditions, 
                                PEtabParameters)::DataFrame

    allowedColumnNames = ["time", "obs_id", "observable_id", "noise_parameters", "measurement",
                          "simulation_id", "pre_equilibration_id", "pre_eq_id", "observable_parameters"]
    columnNames = names(petabMeasurements)
    for name in columnNames
        if name ∈ allowedColumnNames
            continue
        end
        allowedNamesF = prod([tmp * ", " for tmp in allowedColumnNames])[1:end-2]
        strWrite = "$name is not an allowed column name for PEtab measurements, allowed names are : $allowedNamesF"
        throw(PEtab.PEtabFormatError(strWrite))
    end
    df = DataFrame()

    #= 
        Check that input data is valid 
    =#
    checkMeasurementDataColumn(petabMeasurements, "measurement", simulationConditions, observables, PEtabParameters)
    df[!, "measurement"] = petabMeasurements[!, "measurement"]

    checkMeasurementDataColumn(petabMeasurements, "time", simulationConditions, observables, PEtabParameters)
    df[!, "time"] = petabMeasurements[!, "time"]

    checkMeasurementDataColumn(petabMeasurements, "simulation_id", simulationConditions, observables, PEtabParameters)
    df[!, "simulationConditionId"] = petabMeasurements[!, "simulation_id"]

    _name = "observable_id" ∈ columnNames ? "observable_id" : "obs_id"
    checkMeasurementDataColumn(petabMeasurements, _name, simulationConditions, observables, PEtabParameters)
    df[!, "observableId"] = petabMeasurements[!, _name]

    if "pre_equilibration_id" ∈ columnNames || "pre_eq_id" ∈ columnNames # Optional column
        _name = "pre_equilibration_id" ∈ columnNames ? "pre_equilibration_id" : "pre_eq_id"
        checkMeasurementDataColumn(petabMeasurements, _name, simulationConditions, observables, PEtabParameters)
        df[!, "preequilibrationConditionId"] = petabMeasurements[!, _name]
    end

    if "noise_parameters" ∈ columnNames
        checkMeasurementDataColumn(petabMeasurements, "noise_parameters", simulationConditions, observables, PEtabParameters)
        df[!, "noiseParameters"] = replace.(string.(petabMeasurements[!, "noise_parameters"]), "missing" => "")
    end

    if "observable_parameters" ∈ columnNames
        checkMeasurementDataColumn(petabMeasurements, "observable_parameters", simulationConditions, observables, PEtabParameters)
        df[!, "observableParameters"] = replace.(string.(petabMeasurements[!, "observable_parameters"]), "missing" => "")
    end

    return df
end


function checkMeasurementDataColumn(measurements::DataFrame,
                                    columnName::String,
                                    simulationConditions,
                                    observables::Dict{String,PEtabObservable},
                                    petabParameters::Vector{PEtabParameter})::Bool

    if columnName == "time" || columnName == "measurement"
        column = measurements[!, columnName]
        if typeof(column) <: Vector{<:Real}
            return true
        end
        strWrite = "In the measurement data the $columnName column must only contain numerical values (e.g. Floats) "
        strWrite *= "Currently column $columnName contains a none-numerical value"
        throw(PEtab.PEtabFormatError(strWrite))
    end

    if columnName == "simulation_id" || columnName == "pre_equilibration_id" || columnName == "pre_eq_id"
        column = measurements[!, columnName]
        if !(typeof(column) <: Vector{<:AbstractString})
            strWrite = "In the measurement data the $columnName column must only contain names (non-numerical values). "
            strWrite *= "Currently column $columnName contains a numerical value"
            throw(PEtab.PEtabFormatError(strWrite))
        end

        # Check that each simulation condition also is defined in the simulation-conditions struct
        conditionsIds = string.(collect(keys(simulationConditions)))
        local _id
        idInSimulationConditions = true
        for id in column
            if id ∉ conditionsIds
                idInSimulationConditions = false
                _id = id
                break
            end
        end
        if idInSimulationConditions == true
            return true
        end
        strWrite = "In the measurement data in the column $columnName the condition id $_id is not specifed "
        strWrite *= "in the simulation-conditions. For each measurement a valid simulation-condition must be specified."
        throw(PEtab.PEtabFormatError(strWrite))
    end

    if columnName == "observable_id" || columnName == "obs_id"
        column = measurements[!, columnName]
        if !(typeof(column) <: Vector{<:AbstractString})
            strWrite = "In the measurement data the $columnName column must only contain names (non-numerical values). "
            strWrite *= "Currently column $columnName contains a numerical value"
            throw(PEtab.PEtabFormatError(strWrite))
        end

        # Check that each simulation condition also is defined in the simulation-conditions struct
        obsIds = collect(keys(observables))
        local _id
        idInObservables = true
        for id in column
            if id ∉ obsIds
                idInObservables = false
                _id = id
                break
            end
        end
        if idInObservables == true
            return true
        end
        strWrite = "In the measurement data in the column $columnName the observable id $_id is not specifed "
        strWrite *= "as a PEtab observable. For each measurement a valid observable must be specified."
        throw(PEtab.PEtabFormatError(strWrite))
    end

    if columnName == "observable_parameters" || columnName == "noise_parameters"
        column = measurements[!, columnName]
        if !all(typeof.(column) .<: Union{<:Real, <:AbstractString, <:Missing})
            strWrite = "In the measurement data the $columnName column must only contain names or numerical values "
            strWrite *= "Currently column $columnName does not conform to this."
            throw(PEtab.PEtabFormatError(strWrite))
        end

        parameterIds = string.([parameter.parameter for parameter in petabParameters])
        for value in string.(column)

            if isempty(value) || value == "missing"
                continue
            end

            for part in split(value, ';')
                if PEtab.isNumber(part)
                    continue
                end

                if part ∈ parameterIds
                    continue
                end

                strWrite = "In column $columnName the parameter $part is specifed as a noise or observable parameter. "
                strWrite *= "However, as required the parameter is not specifed as a model PEtabParameter."
                throw(PEtab.PEtabFormatError(strWrite))
            end
        end
        return true
    end
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
                        noiseFormula = replace(string(observable.noiseFormula), "(t)" => ""),
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