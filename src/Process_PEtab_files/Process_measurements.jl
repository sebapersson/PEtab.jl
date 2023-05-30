"""
    processMeasurements(measurementsFile::CSV.File, observableData::CSV.File)::MeasurementData

    Process the PeTab measurementsFile file into a type-stable Julia struct.
"""
function processMeasurements(measurementsFile::CSV.File, observablesFile::CSV.File)::MeasurementsInfo

    # Arrays for storing observed time-points and measurment values (yObs)
    measurement::Vector{Float64} = convert(Vector{Float64}, measurementsFile[:measurement])
    time::Vector{Float64} = convert(Vector{Float64}, measurementsFile[:time])
    nMeasurements = length(measurement)
    chi2Values::Vector{Float64} = zeros(Float64, nMeasurements)
    simulatedValues::Vector{Float64} = zeros(Float64, nMeasurements)
    residuals::Vector{Float64} = zeros(Float64, nMeasurements)

    # In case preEquilibrationConditionId is not present in the file we assume no such conditions by default
    preEquilibrationConditionId::Vector{Symbol} = Vector{Symbol}(undef, nMeasurements)
    if :preequilibrationConditionId ∉ measurementsFile.names
        preEquilibrationConditionId .= :None
    end
    if :preequilibrationConditionId ∈ measurementsFile.names
        for i in eachindex(preEquilibrationConditionId)
            preEquilibrationConditionId[i] = Symbol.(string(measurementsFile[:preequilibrationConditionId][i]))
        end
    end
    preEquilibrationConditionId[findall(x -> x == :missing, preEquilibrationConditionId)] .= :None
    simulationConditionId::Vector{Symbol} = Symbol.(string.(measurementsFile[:simulationConditionId]))

    observableId::Vector{Symbol} = Symbol.(string.(measurementsFile[:observableId]))

    # Noise parameters in the PeTab file either have a parameter ID, or they have
    # a value (fixed). Here the values are mapped to a Union{String, Float} vector,
    # correct handling of noise parameters when computing h (yMod) is handled when
    # building the ParameterIndices-struct.
    if :noiseParameters ∉ measurementsFile.names
        _noiseParameters = [missing for i in 1:nMeasurements]
    end
    if :noiseParameters ∈ measurementsFile.names
        _noiseParameters = measurementsFile[:noiseParameters]
    end
    noiseParameters::Vector{Union{String, Float64}} = Vector{Union{String, Float64}}(undef, nMeasurements)
    for i in 1:nMeasurements
        if ismissing(_noiseParameters[i])
            noiseParameters[i] = ""
            continue
        end

        # In case of a single constant value
        if typeof(_noiseParameters[i]) <:AbstractString && isNumber(_noiseParameters[i])
            noiseParameters[i] = parse(Float64, _noiseParameters[i])
            continue
        end

        # Here there might be several noise parameters, or it is a variable. Correctly handled
        # when building ParameterIndices struct
        noiseParameters[i] = string(_noiseParameters[i])
    end

    # observableParameters[i] can store more than one parameter. This is handled correctly
    # when building the ParameterIndices struct.
    if :observableParameters ∉ measurementsFile.names
        _observableParameters = [missing for i in 1:nMeasurements]
    end
    if :observableParameters ∈ measurementsFile.names
        _observableParameters = measurementsFile[:observableParameters]
    end
    observableParameters::Vector{String} = Vector{String}(undef, nMeasurements)
    for i in 1:nMeasurements
        if ismissing(_observableParameters[i])
            observableParameters[i] = ""
            continue
        end
        observableParameters[i] = string(_observableParameters[i])
    end

    # Often we work with transformed data (e.g log-normal measurement errors). To aviod repeating this
    # calculation we hare pre-compute the transformed measurements, and vector with corresponding transformations
    # for each observation.
    measurementTransformation::Vector{Symbol} = Vector{Symbol}(undef, nMeasurements)
    if :observableTransformation ∉ observablesFile.names
        measurementTransformation .= :lin
    end
    if :observableTransformation ∈ observablesFile.names
        for i in 1:nMeasurements
            iRow = findfirst(x -> x == string(observableId[i]), observablesFile[:observableId])
            measurementTransformation[i] = Symbol(observablesFile[:observableTransformation][iRow])
        end
    end
    measurementT::Vector{Float64} = [transformMeasurementOrH(measurement[i], measurementTransformation[i]) for i in eachindex(measurement)]

    return MeasurementsInfo(measurement, measurementT, simulatedValues, chi2Values, residuals, measurementTransformation, time, 
                            observableId,preEquilibrationConditionId, simulationConditionId, noiseParameters, observableParameters)
end
