"""
    parse_measurements(measurements_df, observables_df)::Measurements

    Process the PEtab measurements table into a type-stable Julia struct.
"""
function parse_measurements(measurements_df::DataFrame,
                            observables_df::DataFrame)::MeasurementsInfo
    if :observableTransformation in propertynames(observables_df)
        _check_values_column(observables_df, VALID_SCALES, :observableTransformation,
                             "observables")
    end

    nmeasurements = nrow(measurements_df)
    measurements = zeros(Float64, nmeasurements)
    time = zeros(Float64, nmeasurements)
    observable_ids = fill(Symbol(), nmeasurements)
    condition_ids = fill(Symbol(), nmeasurements)
    pre_equilibration_ids = fill(:None, nmeasurements)
    noise_parameters = fill("", nmeasurements)
    observable_parameters = fill("", nmeasurements)
    transformations = fill(:lin, nmeasurements)

    _parse_table_column!(measurements, measurements_df[!, :measurement], Float64)
    _parse_table_column!(time, measurements_df[!, :time], Float64)
    _parse_table_column!(observable_ids, measurements_df[!, :observableId], Symbol)
    _parse_table_column!(condition_ids, measurements_df[!, :simulationConditionId], Symbol)
    # Optional columns
    if :preequilibrationConditionId in propertynames(measurements_df)
        dfcol = measurements_df[!, :preequilibrationConditionId]
        _parse_table_column!(pre_equilibration_ids, dfcol, Symbol)
    end
    if :noiseParameters in propertynames(measurements_df)
        dfcol = measurements_df[!, :noiseParameters]
        _parse_table_column!(noise_parameters, dfcol, string)
    end
    if :observableParameters in propertynames(measurements_df)
        dfcol = measurements_df[!, :observableParameters]
        _parse_table_column!(observable_parameters, dfcol, string)
    end
    # Special handling as transformation must be obtained from observables_df
    if :observableTransformation in propertynames(observables_df)
        for (i, transformation) in pairs(observables_df[!, :observableTransformation])
            id = observables_df[i, :observableId] |> Symbol
            transformations[observable_ids .== id] .= Symbol(transformation)
        end
    end

    # To avoid computing the transformed measurment values, they are pre-computed
    measurements_t = similar(measurements)
    for (i, val) in pairs(measurements)
        measurements_t[i] = transform_observable(val, transformations[i])
    end

    # Values associated with the measurement values
    # TODO: These should be moved
    chi2_values = zeros(Float64, nmeasurements)
    simulated_values = zeros(Float64, nmeasurements)
    residuals = zeros(Float64, nmeasurements)

    return MeasurementsInfo(measurements, measurements_t, simulated_values, chi2_values,
                            residuals, transformations, time, observable_ids,
                            pre_equilibration_ids, condition_ids, noise_parameters,
                            observable_parameters)
end
