"""
    process_measurements(measurements_file::CSV.File, observableData::CSV.File)::MeasurementData

    Process the PeTab measurements_file file into a type-stable Julia struct.
"""
function process_measurements(measurements_file::CSV.File, observablesFile::CSV.File)::MeasurementsInfo

    # Arrays for storing observed time-points and measurment values (y_obs)
    measurement::Vector{Float64} = convert(Vector{Float64}, measurements_file[:measurement])
    time::Vector{Float64} = convert(Vector{Float64}, measurements_file[:time])
    n_measurements = length(measurement)
    chi2_values::Vector{Float64} = zeros(Float64, n_measurements)
    simulated_values::Vector{Float64} = zeros(Float64, n_measurements)
    residuals::Vector{Float64} = zeros(Float64, n_measurements)

    # In case pre_equilibration_condition_id is not present in the file we assume no such conditions by default
    pre_equilibration_condition_id::Vector{Symbol} = Vector{Symbol}(undef, n_measurements)
    if :preequilibrationConditionId ∉ measurements_file.names
        pre_equilibration_condition_id .= :None
    end
    if :preequilibrationConditionId ∈ measurements_file.names
        for i in eachindex(pre_equilibration_condition_id)
            pre_equilibration_condition_id[i] = Symbol.(string(measurements_file[:preequilibrationConditionId][i]))
        end
    end
    pre_equilibration_condition_id[findall(x -> x == :missing, pre_equilibration_condition_id)] .= :None
    simulation_condition_id::Vector{Symbol} = Symbol.(string.(measurements_file[:simulationConditionId]))

    observable_Iid::Vector{Symbol} = Symbol.(string.(measurements_file[:observableId]))

    # Noise parameters in the PeTab file either have a parameter ID, or they have
    # a value (fixed). Here the values are mapped to a Union{String, Float} vector,
    # correct handling of noise parameters when computing h (y_model) is handled when
    # building the ParameterIndices-struct.
    if :noiseParameters ∉ measurements_file.names
        _noise_parameters = [missing for i in 1:n_measurements]
    end
    if :noiseParameters ∈ measurements_file.names
        _noise_parameters = measurements_file[:noiseParameters]
    end
    noise_parameters::Vector{Union{String, Float64}} = Vector{Union{String, Float64}}(undef, n_measurements)
    for i in 1:n_measurements
        if ismissing(_noise_parameters[i])
            noise_parameters[i] = ""
            continue
        end

        # In case of a single constant value
        if typeof(_noise_parameters[i]) <:AbstractString && is_number(_noise_parameters[i])
            noise_parameters[i] = parse(Float64, _noise_parameters[i])
            continue
        end

        # Here there might be several noise parameters, or it is a variable. Correctly handled
        # when building ParameterIndices struct
        noise_parameters[i] = string(_noise_parameters[i])
    end

    # observable_parameters[i] can store more than one parameter. This is handled correctly
    # when building the ParameterIndices struct.
    if :observableParameters ∉ measurements_file.names
        _observable_parameters = [missing for i in 1:n_measurements]
    end
    if :observableParameters ∈ measurements_file.names
        _observable_parameters = measurements_file[:observableParameters]
    end
    observable_parameters::Vector{String} = Vector{String}(undef, n_measurements)
    for i in 1:n_measurements
        if ismissing(_observable_parameters[i])
            observable_parameters[i] = ""
            continue
        end
        observable_parameters[i] = string(_observable_parameters[i])
    end

    # Often we work with transformed data (e.g log-normal measurement errors). To aviod repeating this
    # calculation we hare pre-compute the transformed measurements, and vector with corresponding transformations
    # for each observation.
    measurement_transformation::Vector{Symbol} = Vector{Symbol}(undef, n_measurements)
    if :observableTransformation ∉ observablesFile.names
        measurement_transformation .= :lin
    end
    if :observableTransformation ∈ observablesFile.names
        for i in 1:n_measurements
            i_row = findfirst(x -> x == string(observable_Iid[i]), observablesFile[:observableId])
            measurement_transformation[i] = Symbol(observablesFile[:observableTransformation][i_row])
        end
    end
    measurementT::Vector{Float64} = [transform_measurement_or_h(measurement[i], measurement_transformation[i]) for i in eachindex(measurement)]

    return MeasurementsInfo(measurement, measurementT, simulated_values, chi2_values, residuals, measurement_transformation, time,
                            observable_Iid,pre_equilibration_condition_id, simulation_condition_id, noise_parameters, observable_parameters)
end
