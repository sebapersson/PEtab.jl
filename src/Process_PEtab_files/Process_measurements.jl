"""
    processMeasurements(measurementsFile::DataFrame, observableData::DataFrame)::MeasurementData

    Process the PeTab measurementsFile file into a type-stable Julia struct.
"""
function processMeasurements(measurementsFile::DataFrame, observablesFile::DataFrame)::MeasurementsInfo

    # Arrays for storing observed time-points and measurment values (yObs)
    measurement::Vector{Float64} = convert(Vector{Float64}, measurementsFile[!, "measurement"])
    time::Vector{Float64} = convert(Vector{Float64}, measurementsFile[!, "time"])
    nMeasurements = length(measurement)

    # In case preEquilibrationConditionId is not present in the file we assume no such conditions by default 
    preEquilibrationConditionId::Vector{Symbol} = Vector{Symbol}(undef, nMeasurements)
    if !("preequilibrationConditionId" in names(measurementsFile))
        preEquilibrationConditionId .= :None
    else
        for i in eachindex(preEquilibrationConditionId)
            preEquilibrationConditionId[i] = Symbol.(string(measurementsFile[i, "preequilibrationConditionId"]))
        end
    end
    preEquilibrationConditionId[findall(x -> x == :missing, preEquilibrationConditionId)] .= :None
    simulationConditionId::Vector{Symbol} = Symbol.(string.(measurementsFile[!, "simulationConditionId"]))
    
    observableId::Vector{Symbol} = Symbol.(string.(measurementsFile[!, "observableId"]))

    # Noise parameters in the PeTab file either have a parameter ID, or they have 
    # a value (fixed). Here the values are mapped to a Union{String, Float} vector, 
    # correct handling of noise parameters when computing h (yMod) is handled when  
    # building the ParameterIndices-struct.
    if !("noiseParameters" in names(measurementsFile))
        _noiseParameters = [missing for i in 1:nMeasurements]
    else
        _noiseParameters = measurementsFile[!, "noiseParameters"]
    end
    noiseParameters::Vector{Union{String, Float64}} = Vector{Union{String, Float64}}(undef, nMeasurements)
    for i in 1:nMeasurements
        if ismissing(_noiseParameters[i]) 
            noiseParameters[i] = ""
        # In case of a single constant value
        elseif typeof(_noiseParameters[i]) <:AbstractString && isNumber(_noiseParameters[i])
            noiseParameters[i] = parse(Float64, _noiseParameters[i])
        # Here there might be several noise parameters, or it is a variable. Correctly handled 
        # when building ParameterIndices struct
        else
            noiseParameters[i] = string(_noiseParameters[i])
        end
    end

    # observableParameters[i] can store more than one parameter. This is handled correctly 
    # when building the ParameterIndices struct.
    if !("observableParameters" in names(measurementsFile))
        _observableParameters = [missing for i in 1:nMeasurements]
    else
        _observableParameters = measurementsFile[!, "observableParameters"]
    end
    observableParameters::Vector{String} = Vector{String}(undef, nMeasurements)
    for i in 1:nMeasurements
        if ismissing(_observableParameters[i]) 
            observableParameters[i] = ""
        else
            observableParameters[i] = string(_observableParameters[i])
        end
    end

    # Often we work with transformed data (e.g log-normal measurement errors). To aviod repeating this 
    # calculation we hare pre-compute the transformed measurements, and vector with corresponding transformations 
    # for each observation.
    measurementTransformation::Vector{Symbol} = Vector{Symbol}(undef, nMeasurements)
    # Default linear
    if !("observableTransformation" in names(observablesFile))
        measurementTransformation .= :lin
    else
        for i in 1:nMeasurements
            iRow = findfirst(x -> x == string(observableId[i]), observablesFile[!, "observableId"])
            measurementTransformation[i] = Symbol(observablesFile[iRow, "observableTransformation"]) 
        end
    end
    measurementT::Vector{Float64} = [transformMeasurementOrH(measurement[i], measurementTransformation[i]) for i in eachindex(measurement)]
    
    return MeasurementsInfo(measurement, measurementT, measurementTransformation, time, observableId, 
                           preEquilibrationConditionId, simulationConditionId, noiseParameters, 
                           observableParameters)                           
end
