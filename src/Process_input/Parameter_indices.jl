#=
    Functions for creating the ParameterIndices struct. This index struct contains maps for how to
    split the θ_est into the dynamic, non-dynamic, sd and observable parameters. There are also
    indices for mapping parameters between ode_problem.p and θ_est for parameters which are constant
    constant accross experimental conditions, and parameters specific to experimental conditions.
=#

function compute_θ_indices(parameter_info::ParametersInfo,
                           measurements_info::MeasurementsInfo,
                           petab_model::PEtabModel)::ParameterIndices
    conditions_df = petab_model.conditions_df
    return compute_θ_indices(parameter_info, measurements_info, petab_model.system_mutated,
                             petab_model.parameter_map, petab_model.state_map,
                             conditions_df)
end
function compute_θ_indices(parameter_info::ParametersInfo,
                           measurements_info::MeasurementsInfo,
                           system,
                           parameter_map,
                           state_map,
                           conditions_df::DataFrame)::ParameterIndices
    θ_observable_names, θ_sd_names, θ_non_dynamic_names, θ_dynamic_names = compute_θ_names(parameter_info,
                                                                                           measurements_info,
                                                                                           system,
                                                                                           conditions_df)
    # When computing the gradient tracking parameters not part of ODE system is helpful
    iθ_not_ode_names::Vector{Symbol} = Symbol.(unique(vcat(θ_sd_names, θ_observable_names,
                                                           θ_non_dynamic_names)))
    # Names in the big θ_est vector
    θ_names::Vector{Symbol} = Symbol.(vcat(θ_dynamic_names, iθ_not_ode_names))

    # Indices for each parameter in the big θ_est vector
    iθ_dynamic::Vector{Int64} = [findfirst(x -> x == θ_dynamic_names[i], θ_names)
                                 for i in eachindex(θ_dynamic_names)]
    iθ_sd::Vector{Int64} = [findfirst(x -> x == θ_sd_names[i], θ_names)
                            for i in eachindex(θ_sd_names)]
    iθ_observable::Vector{Int64} = [findfirst(x -> x == θ_observable_names[i], θ_names)
                                    for i in eachindex(θ_observable_names)]
    iθ_non_dynamic::Vector{Int64} = [findfirst(x -> x == θ_non_dynamic_names[i], θ_names)
                                     for i in eachindex(θ_non_dynamic_names)]
    iθ_not_ode::Vector{Int64} = [findfirst(x -> x == iθ_not_ode_names[i], θ_names)
                                 for i in eachindex(iθ_not_ode_names)]

    # When extracting observable or sd parameter for computing the cost we use a pre-computed map to efficently
    # extract correct parameters
    mapθ_observable = build_θ_sd_observable_map(θ_observable_names, measurements_info,
                                                parameter_info, buildθ_observable = true)
    mapθ_sd = build_θ_sd_observable_map(θ_sd_names, measurements_info, parameter_info,
                                        buildθ_observable = false)

    # Compute a map to map parameters between θ_dynamic and ode_problem.p
    name_parameters_ode = Symbol.(string.(parameters(system)))
    _iθ_dynamic::Vector{Int64} = findall(x -> x ∈ name_parameters_ode, θ_dynamic_names)
    i_ode_problem_θ_dynamic::Vector{Int64} = [findfirst(x -> x == θ_dynamic_names[i],
                                                        name_parameters_ode)
                                              for i in _iθ_dynamic]
    map_ode_problem::Map_ode_problem = Map_ode_problem(_iθ_dynamic, i_ode_problem_θ_dynamic)

    # Set up a map for changing between experimental conditions
    maps_condition_id::Dict{Symbol, MapConditionId} = compute_maps_condition(system,
                                                                             parameter_map,
                                                                             state_map,
                                                                             parameter_info,
                                                                             conditions_df,
                                                                             θ_dynamic_names)

    # Set up a named tuple tracking the transformation of each parameter
    _θ_scale = [parameter_info.parameter_scale[findfirst(x -> x == θ_name,
                                                         parameter_info.parameter_id)]
                for θ_name in θ_names]
    θ_scale::Dict{Symbol, Symbol} = Dict([(θ_names[i], _θ_scale[i])
                                          for i in eachindex(θ_names)])

    θ_indices = ParameterIndices(iθ_dynamic,
                                 iθ_observable,
                                 iθ_sd,
                                 iθ_non_dynamic,
                                 iθ_not_ode,
                                 θ_dynamic_names,
                                 θ_observable_names,
                                 θ_sd_names,
                                 θ_non_dynamic_names,
                                 iθ_not_ode_names,
                                 θ_names,
                                 θ_scale,
                                 mapθ_observable,
                                 mapθ_sd,
                                 map_ode_problem,
                                 maps_condition_id)

    return θ_indices
end

function compute_θ_names(parameter_info::ParametersInfo,
                         measurements_info::MeasurementsInfo,
                         system,
                         conditions_df::DataFrame)::Tuple{Vector{Symbol},
                                                                        Vector{Symbol},
                                                                        Vector{Symbol},
                                                                        Vector{Symbol}}

    # Extract the name of all parameter types
    θ_observable_names::Vector{Symbol} = compute_names_obs_sd_parameters(measurements_info.observable_parameters,
                                                                         parameter_info)
    isθ_observable::Vector{Bool} = [parameter_info.parameter_id[i] in θ_observable_names
                                    for i in eachindex(parameter_info.parameter_id)]

    θ_sd_names::Vector{Symbol} = compute_names_obs_sd_parameters(measurements_info.noise_parameters,
                                                                 parameter_info)
    isθ_sd::Vector{Bool} = [parameter_info.parameter_id[i] in θ_sd_names
                            for i in eachindex(parameter_info.parameter_id)]

    # Non-dynamic parameters. This are parameters not entering the ODE system (or initial values), are not
    # noise-parameter or observable parameters, but appear in SD and/or OBS functions. We need to track these
    # as non-dynamic parameters since  want to compute gradients for these given a fixed ODE solution.
    # Non-dynamic parameters not allowed to be observable or sd parameters
    _isθ_non_dynamic = (parameter_info.estimate .&& .!isθ_observable .&& .!isθ_sd)
    _θ_non_dynamic_names = parameter_info.parameter_id[_isθ_non_dynamic]
    # Non-dynamic parameters not allowed to be part of the ODE-system
    _θ_non_dynamic_names = _θ_non_dynamic_names[findall(x -> x ∉
                                                             Symbol.(string.(parameters(system))),
                                                        _θ_non_dynamic_names)]
    # Non-dynamic parameters not allowed to be experimental condition specific parameters
    conditions_specific_θ_dynamic = identify_cond_specific_θ_dynamic(system, parameter_info,
                                                                     conditions_df)
    θ_non_dynamic_names::Vector{Symbol} = _θ_non_dynamic_names[findall(x -> x ∉
                                                                            conditions_specific_θ_dynamic,
                                                                       _θ_non_dynamic_names)]
    isθ_non_dynamic = [parameter_info.parameter_id[i] in θ_non_dynamic_names
                       for i in eachindex(parameter_info.parameter_id)]

    isθ_dynamic::Vector{Bool} = (parameter_info.estimate .&& .!isθ_non_dynamic .&&
                                 .!isθ_sd .&& .!isθ_observable)
    θ_dynamic_names::Vector{Symbol} = parameter_info.parameter_id[isθ_dynamic]

    return θ_observable_names, θ_sd_names, θ_non_dynamic_names, θ_dynamic_names
end

# Helper function for extracting ID:s for observable and noise parameters from the noise- and observable column
# in the PEtab file.
function compute_names_obs_sd_parameters(noise_obs_column::T1,
                                         parameter_info::ParametersInfo) where {
                                                                                T1 <:
                                                                                Vector{<:Union{<:String,
                                                                                               <:AbstractFloat}}
                                                                                }
    θ_names = Symbol[]
    for i in eachindex(noise_obs_column)
        if isempty(noise_obs_column[i]) || is_number(string(noise_obs_column[i]))
            continue
        end

        parameters_rowi = split(noise_obs_column[i], ';')
        for _parameter in parameters_rowi
            parameter = Symbol(_parameter)
            # Disregard Id if parameters should not be estimated, or
            i_parameter = findfirst(x -> x == parameter, parameter_info.parameter_id)
            if is_number(_parameter) || parameter in θ_names ||
               parameter_info.estimate[i_parameter] == false
                continue
            elseif isnothing(i_parameter)
                @error "Parameter $parameter could not be found in parameter file"
            end

            θ_names = vcat(θ_names, parameter)
        end
    end

    return θ_names
end

# Identifaying dynamic parameters to estimate, where the dynamic parameters are only used for some specific
# experimental conditions.
function identify_cond_specific_θ_dynamic(system,
                                          parameter_info::ParametersInfo,
                                          conditions_df::DataFrame)::Vector{Symbol}
    all_parameters_ode = string.(parameters(system))
    model_state_names = string.(states(system))
    model_state_names = replace.(model_state_names, "(t)" => "")
    parameters_estimate = parameter_info.parameter_id[parameter_info.estimate]

    # List of parameters which have specific values for specific experimental conditions, these can be extracted
    # from the rows of the conditions_df (where the column is the name of the parameter in the ODE-system,
    # and the rows are the corresponding names of the parameter value to estimate)
    conditions_specific_θ_dynamic = Vector{Symbol}(undef, 0)
    column_names = names(conditions_df)
    length(column_names) == 1 && return conditions_specific_θ_dynamic
    i_start = column_names[2] == "conditionName" ? 3 : 2 # Sometimes PEtab file does not include column conditionName
    for i in i_start:length(names(conditions_df))
        if column_names[i] ∉ all_parameters_ode && column_names[i] ∉ model_state_names
            @error "Problem : Parameter ", column_names[i],
                   " should be in the ODE model as it dicates an experimental condition"
        end

        for j in 1:nrow(conditions_df)
            if (_parameter = Symbol(string(conditions_df[j, i]))) ∈
               parameters_estimate
                conditions_specific_θ_dynamic = vcat(conditions_specific_θ_dynamic,
                                                     _parameter)
            end
        end
    end

    return unique(conditions_specific_θ_dynamic)
end

# For each observation build a map that correctly from either θ_observable or θ_sd map extract the correct value
# for the time-point specific observable and noise parameters when compuing σ or h (observable) value.
function build_θ_sd_observable_map(n_parameters_estimate::Vector{Symbol},
                                   measurements_info::MeasurementsInfo,
                                   parameter_info::ParametersInfo;
                                   buildθ_observable = true)
    parameter_map::Vector{θObsOrSdParameterMap} = Vector{θObsOrSdParameterMap}(undef,
                                                                               length(measurements_info.time))
    if buildθ_observable == true
        timepoint_values = measurements_info.observable_parameters
    else
        timepoint_values = measurements_info.noise_parameters
    end

    # For each time-point build an associated map which stores if i) noise/obserable parameters are constants, ii) should
    # be estimated, iii) and corresponding index in parameter vector
    for i in eachindex(parameter_map)
        # In case we do not have any noise/obserable parameter
        if isempty(timepoint_values[i])
            parameter_map[i] = θObsOrSdParameterMap(Vector{Bool}(undef, 0),
                                                    Vector{Int64}(undef, 0),
                                                    Vector{Float64}(undef, 0), Int64(0),
                                                    false)
        end

        # In case of a constant noise/obserable parameter encoded as a Float in the PEtab file.
        if typeof(timepoint_values[i]) <: Real
            parameter_map[i] = θObsOrSdParameterMap(Vector{Bool}(undef, 0),
                                                    Vector{Int64}(undef, 0),
                                                    Float64[timepoint_values[i]], Int64(0),
                                                    true)
        end

        # In case observable or noise parameter maps to a parameter
        if !isempty(timepoint_values[i]) && !(typeof(timepoint_values[i]) <: Real)

            # Parameter are delimited by ; in the PEtab files and they can be constant, or they can
            # be in the vector to estimate θ
            parameters_in_expression = split(timepoint_values[i], ';')
            n_parameters::Int = length(parameters_in_expression)
            should_estimate::Vector{Bool} = Vector{Bool}(undef, n_parameters)
            index_in_θ::Vector{Int64} = Vector{Int64}(undef, n_parameters)
            constant_values::Array{Float64, 1} = Vector{Float64}(undef, n_parameters)

            for j in eachindex(parameters_in_expression)
                # In case observable parameter in paramsRet[j] should be estimated save which index
                # it has in the θ vector
                if Symbol(parameters_in_expression[j]) ∈ n_parameters_estimate
                    should_estimate[j] = true
                    index_in_θ[j] = Int64(findfirst(x -> x ==
                                                         Symbol(parameters_in_expression[j]),
                                                    n_parameters_estimate))
                    continue
                end

                # In case observable parameter in paramsRet[j] is constant save its constant value.
                # The constant value can be found either directly in the measurements_infoFile, or in
                # in the parameters_df.
                should_estimate[j] = false
                # Hard coded in Measurement data file
                if is_number(parameters_in_expression[j])
                    constant_values[j] = parse(Float64, parameters_in_expression[j])
                    continue
                end
                # Hard coded in Parameters file
                if Symbol(parameters_in_expression[j]) in parameter_info.parameter_id
                    constant_values[j] = parameter_info.nominal_value[findfirst(x -> x ==
                                                                                     Symbol(parameters_in_expression[j]),
                                                                                parameter_info.parameter_id)]
                    continue
                end

                @error "Cannot find matching for parameter ", parameters_in_expression[j],
                       " when building map."
            end

            parameter_map[i] = θObsOrSdParameterMap(should_estimate,
                                                    index_in_θ[should_estimate],
                                                    constant_values[.!should_estimate],
                                                    Int64(length(parameters_in_expression)),
                                                    false)
        end
    end

    return parameter_map
end

# A map to accurately map parameters for a specific experimental conditionId to the ODE-problem
function compute_maps_condition(system,
                                parameter_map,
                                state_map,
                                parameter_info::ParametersInfo,
                                conditions_df::DataFrame,
                                _θ_dynamic_names::Vector{Symbol})::Dict{Symbol,
                                                                        MapConditionId}
    θ_dynamic_names = string.(_θ_dynamic_names)
    n_conditions = nrow(conditions_df)
    model_state_names = string.(states(system))
    model_state_names = replace.(model_state_names, "(t)" => "")
    all_parameters_ode = string.(parameters(system))

    i_start = "conditionName" in names(conditions_df) ? 3 : 2
    condition_specific_variables = string.(names(conditions_df)[i_start:end])

    maps_condition_id::Dict{Symbol, MapConditionId} = Dict()

    for i in 1:n_conditions
        constant_parameters::Vector{Float64} = Vector{Float64}(undef, 0)
        i_ode_constant_parameters::Vector{Int64} = Vector{Int64}(undef, 0)
        constant_states::Vector{Float64} = Vector{Float64}(undef, 0)
        i_ode_constant_states::Vector{Int64} = Vector{Int64}(undef, 0)
        iθ_dynamic::Vector{Int64} = Vector{Int64}(undef, 0)
        i_ode_problem_θ_dynamic::Vector{Int64} = Vector{Int64}(undef, 0)

        condition_id_name = Symbol(string(conditions_df[i, 1]))

        rowi = string.(collect(conditions_df[i, i_start:end]))
        for j in eachindex(rowi)

            # In case a condition specific ode-system parameter is mapped to constant number
            if is_number(rowi[j]) && condition_specific_variables[j] ∈ all_parameters_ode
                constant_parameters = vcat(constant_parameters, parse(Float64, rowi[j]))
                i_ode_constant_parameters = vcat(i_ode_constant_parameters,
                                                 findfirst(x -> x ==
                                                                condition_specific_variables[j],
                                                           all_parameters_ode))
                continue
            end
            if is_number(rowi[j]) && condition_specific_variables[j] ∈ model_state_names
                constant_parameters = vcat(constant_parameters, parse(Float64, rowi[j]))
                i_ode_constant_parameters = vcat(i_ode_constant_parameters,
                                                 findfirst(x -> x ==
                                                                "__init__" *
                                                                condition_specific_variables[j] *
                                                                "__", all_parameters_ode))
                continue
            end
            is_number(rowi[j]) &&
                @error "Error : Cannot build map for experimental condition variable",
                       condition_specific_variables[j]

            # In case we are trying to change one the θ_dynamic parameters we are estimating
            if rowi[j] ∈ θ_dynamic_names &&
               condition_specific_variables[j] ∈ all_parameters_ode
                iθ_dynamic = vcat(iθ_dynamic, findfirst(x -> x == rowi[j], θ_dynamic_names))
                i_ode_problem_θ_dynamic = vcat(i_ode_problem_θ_dynamic,
                                               findfirst(x -> x ==
                                                              condition_specific_variables[j],
                                                         all_parameters_ode))
                continue
            end
            if rowi[j] ∈ θ_dynamic_names &&
               condition_specific_variables[j] ∈ model_state_names
                iθ_dynamic = vcat(iθ_dynamic, findfirst(x -> x == rowi[j], θ_dynamic_names))
                i_ode_problem_θ_dynamic = vcat(i_ode_problem_θ_dynamic,
                                               findfirst(x -> x ==
                                                              "__init__" *
                                                              condition_specific_variables[j] *
                                                              "__", all_parameters_ode))
                continue
            end
            rowi[j] ∈ θ_dynamic_names &&
                @error "Could not map " * string(condition_specific_variables[j]) *
                       " when building condition map"

            # In case rowi is a parameter but we do not estimate said parameter
            if rowi[j] ∈ string.(parameter_info.parameter_id)
                i_value = findfirst(x -> x == rowi[j], string.(parameter_info.parameter_id))
                constant_parameters = vcat(constant_parameters,
                                           parameter_info.nominal_value[i_value])
                i_ode_constant_parameters = vcat(i_ode_constant_parameters,
                                                 findfirst(x -> x ==
                                                                condition_specific_variables[j],
                                                           all_parameters_ode))
                continue
            end

            # In case rowi is missing (specifically NaN) the default SBML-file value should be used. To this end we need to
            # have access to the parameter and state map to handle both states and parameters. Then must fix such that
            # __init__ parameters  take on the correct value.
            if rowi[j] == "missing" && condition_specific_variables[j] ∈ all_parameters_ode
                valueDefault = get_default_values_maps(string(condition_specific_variables[j]),
                                                       parameter_map, state_map)
                constant_parameters = vcat(constant_parameters, valueDefault)
                i_ode_constant_parameters = vcat(i_ode_constant_parameters,
                                                 findfirst(x -> x ==
                                                                condition_specific_variables[j],
                                                           all_parameters_ode))
                continue
            end
            if rowi[j] == "missing" && condition_specific_variables[j] ∈ model_state_names
                valueDefault = get_default_values_maps(string(condition_specific_variables[j]),
                                                       parameter_map, state_map)
                constant_parameters = vcat(constant_parameters, valueDefault)
                i_ode_constant_parameters = vcat(i_ode_constant_parameters,
                                                 findfirst(x -> x ==
                                                                "__init__" *
                                                                condition_specific_variables[j] *
                                                                "__", all_parameters_ode))
                continue
            end

            # NaN can only applie for states
            if rowi[j] == "NaN" && condition_specific_variables[j] ∈ model_state_names
                constant_parameters = vcat(constant_parameters, NaN)
                i_ode_constant_parameters = vcat(i_ode_constant_parameters,
                                                 findfirst(x -> x ==
                                                                "__init__" *
                                                                condition_specific_variables[j] *
                                                                "__", all_parameters_ode))
                continue
            else
                str_write = "If a row in conditions file is NaN then the column header must be a state"
                throw(PEtabFileError(str_write))
            end

            # If we reach this far something is off and an error must be thrown
            str_write = "Could not map parameters for condition " *
                        string(condition_id_name) * " for parameter " * string(rowi[j])
            throw(PEtabFileError(str_write))
        end

        maps_condition_id[condition_id_name] = MapConditionId(constant_parameters,
                                                              i_ode_constant_parameters,
                                                              constant_states,
                                                              i_ode_constant_states,
                                                              iθ_dynamic,
                                                              i_ode_problem_θ_dynamic)
    end

    return maps_condition_id
end

# Extract default parameter value from state, or parameter map
function get_default_values_maps(which_parameter_state, parameter_map, state_map)
    parameter_map_names = string.([parameter_map[i].first for i in eachindex(parameter_map)])
    state_map_names = replace.(string.([state_map[i].first for i in eachindex(state_map)]),
                               "(t)" => "")

    # Parameters are only allowed to map to concrete values
    if which_parameter_state ∈ parameter_map_names
        which_index = findfirst(x -> x == which_parameter_state, parameter_map_names)
        return parse(Float64, string(parameter_map[which_index].second))
    end

    # States can by default map to a parameter by one level of recursion
    @assert which_parameter_state ∈ state_map_names
    which_index = findfirst(x -> x == which_parameter_state, state_map_names)
    value_map_to = string(state_map[which_index].second)
    if value_map_to ∈ parameter_map_names
        which_index_parameter = findfirst(x -> x == value_map_to, parameter_map_names)
        _value_map_to = string(parameter_map[which_index_parameter].second)
        if _value_map_to ∈ parameter_map_names
            _which_index_parameter = findfirst(x -> x == _value_map_to, parameter_map_names)
            return parse(Float64, string.(parameter_map[_which_index_parameter].second))
        else
            return parse(Float64, _value_map_to)
        end
    end

    return parse(Float64, string(state_map[which_index].second))
end
