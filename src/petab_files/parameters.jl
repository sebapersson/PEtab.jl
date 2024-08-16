function parse_parameters(parameters_df::DataFrame; custom_values::Union{Nothing, Dict} = nothing)::ParametersInfo
    _check_values_column(parameters_df, VALID_SCALES, :parameterScale, "parameters")
    _check_values_column(parameters_df, [0, 1], :estimate, "parameters")

    nparameters = nrow(parameters_df)
    lower_bounds = fill(-Inf, nparameters)
    upper_bounds = fill(Inf, nparameters)
    nominal_values = zeros(Float64, nparameters) # Vector with Nominal value in PeTab-file
    parameter_ids = fill(Symbol(), nparameters)
    paramter_scales = fill(Symbol(), nparameters)
    estimate = fill(false, nparameters)

    _parse_table_column!(lower_bounds, parameters_df[!, :lowerBound], Float64)
    _parse_table_column!(upper_bounds, parameters_df[!, :upperBound], Float64)
    _parse_table_column!(nominal_values, parameters_df[!, :nominalValue], Float64)
    _parse_table_column!(parameter_ids, parameters_df[!, :parameterId], Symbol)
    _parse_table_column!(paramter_scales, parameters_df[!, :parameterScale], Symbol)
    _parse_table_column!(estimate, parameters_df[!, :estimate], Bool)
    nparameters_esimtate = Int64(sum(estimate))

    # When doing model selection it can be necessary to change the parameter values
    # without changing in the PEtab files. To get all subsequent parameter running
    # correct it must be done here. TODO: Refactor when time for PEtab-select
    if !isnothing(custom_values)
        for (id, value) in custom_values
            ip = findfirst(x -> x == id, parameter_ids)
            if value == "estimate"
                estimate[ip] = true
            elseif value isa Real
                estimate[ip] = false
                nominal_values[ip] = value
            elseif is_number(value)
                estimate[ip] = false
                nominal_values[ip] = parse(Float64, value)
            end
        end
    end

    return ParametersInfo(nominal_values, lower_bounds, upper_bounds, parameter_ids, paramter_scales, estimate, nparameters_esimtate)
end
