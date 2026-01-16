"""
    PEtabMeasurements(measurements_df, observables_df)::Measurements

    Process the PEtab measurements table into a type-stable Julia struct.
"""
function PEtabMeasurements(petab_tables::PEtabTables)
    measurements_df, observables_df = _get_petab_tables(
        petab_tables, [:measurements, :observables]
    )
    PEtabMeasurements(measurements_df, observables_df)
end
function PEtabMeasurements(
        measurements_df::DataFrame, observables_df::DataFrame
    )::PEtabMeasurements
    if :observableTransformation in propertynames(observables_df)
        _check_values_column(observables_df, VALID_SCALES, :observableTransformation,
                             "observables"; allow_missing = true)
    end

    nmeasurements = nrow(measurements_df)
    measurements = zeros(Float64, nmeasurements)
    time = zeros(Float64, nmeasurements)
    observable_ids = fill(Symbol(), nmeasurements)
    condition_ids = fill(Symbol(), nmeasurements)
    pre_equilibration_ids = fill(:None, nmeasurements)
    noise_parameters = fill("", nmeasurements)
    observable_parameters = fill("", nmeasurements)
    noise_distributions = fill(:Normal, nmeasurements)

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

    # Special handling as data distribution must be obtained from observables_df
    if :observableTransformation in propertynames(observables_df)
        for (row_idx, transformation) in pairs(observables_df[!, :observableTransformation])
            if :noiseDistribution in propertynames(observables_df) && !ismissing(observables_df.noiseDistribution[row_idx])
                _distribution = if ismissing(observables_df.noiseDistribution[row_idx])
                    :Normal
                else
                    observables_df.noiseDistribution[row_idx] |>
                        uppercasefirst |>
                        Symbol
                end
            else
                _distribution = :Normal
            end

            row_jdx = observable_ids .== Symbol(observables_df[row_idx, :observableId])
            if ismissing(transformation)
                noise_distributions[row_jdx] .= _distribution
            elseif transformation == "log"
                noise_distributions[row_jdx] .= Symbol("Log$(_distribution)")
            elseif transformation == "log10"
                noise_distributions[row_jdx] .= Symbol("Log10$(_distribution)")
            elseif transformation == "log2"
                noise_distributions[row_jdx] .= Symbol("Log2$(_distribution)")
            end
        end
    end

    # To avoid computing the transformed measurment values, they are pre-computed
    measurements_t = similar(measurements)
    for (i, val) in pairs(measurements)
        measurements_t[i] = _transform_h(val, noise_distributions[i])
    end

    # PEtab v2 introduces none-zero simulation start times. In the conversion of PEtab
    # v2 to PEtab v1 tables this additional information is encoded as an extra column
    # in the measurements table, which is then parsed in SimulationInfo
    if "simulationStartTime" in names(measurements_df)
        simulation_start_time = measurements_df.simulationStartTime
    else
        simulation_start_time = zeros(Float64, nmeasurements)
    end

    # Values associated with the measurement values
    # TODO: These should be moved
    chi2_values = zeros(Float64, nmeasurements)
    simulated_values = zeros(Float64, nmeasurements)
    residuals = zeros(Float64, nmeasurements)

    return PEtabMeasurements(measurements, measurements_t, simulated_values, chi2_values,
                             residuals, noise_distributions, time, observable_ids,
                             pre_equilibration_ids, condition_ids, noise_parameters,
                             observable_parameters, simulation_start_time)
end
