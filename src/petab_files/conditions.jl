#=
    Functions for creating the ParameterIndices struct. This index struct contains maps for how to
    split the θ_est into the dynamic, non-dynamic, sd and observable parameters. There are also
    indices for mapping parameters between ode_problem.p and θ_est for parameters which are constant
    constant accross experimental conditions, and parameters specific to experimental conditions.
=#
# ParameterIndices = ParameterMappings
# ParametersInfo = PEtabParameters
# MeasurementsInfo = PEtabMeasurements

"""
        parse_conditions

Parse conditions and build parameter maps.

There are four type of parameters in a PEtab problem
1. Dynamic (appear in ODE)
2. Noise (appear in noiseParameters columns of measurement table)
3. Observable (appear in observableParameters of measurement table)
4. NonDynamic (do not correspond to any above)
These parameter types need to be treated separately for computing efficient gradients.
This function extracts which parameter is what type, and builds maps for correctly mapping
the parameter during likelihood computations.
"""
function parse_conditions(parameter_info::ParametersInfo, measurements_info::MeasurementsInfo, petab_model::PEtabModel)::ParameterIndices
    conditions_df = petab_model.conditions_df
    return parse_conditions(parameter_info, measurements_info, petab_model.system_mutated, petab_model.parameter_map, petab_model.state_map, conditions_df)
end
function parse_conditions(parameter_info::ParametersInfo, measurements_info::MeasurementsInfo, sys, parameter_map, state_map, conditions_df::DataFrame)::ParameterIndices
    xids = _get_xids(parameter_info, measurements_info, sys, conditions_df)

    # xindices for mapping parameters from xest -> xdynamic, etc..
    xindices = _get_xindices(xids)
    mapθ_observable = build_θ_sd_observable_map(xids[:observable], measurements_info, parameter_info, buildθ_observable = true)
    mapθ_sd = build_θ_sd_observable_map(xids[:noise], measurements_info, parameter_info, buildθ_observable = false)

    # TODO: SII is going to make this much easier (but the reverse will be harder)
    # Compute a map to map parameters between dynamic and system parameters
    sys_to_dynamic = findall(x -> x in xids[:sys], xids[:dynamic]) |> Vector{Int64}
    dynamic_to_sys = Int64[findfirst(x -> x == id, xids[:sys]) for id in xids[:dynamic][sys_to_dynamic]]
    map_ode_problem::MapODEProblem = MapODEProblem(sys_to_dynamic, dynamic_to_sys)

    # Set up a map for changing between experimental conditions
    maps_condition_id::Dict{Symbol, MapConditionId} = compute_maps_condition(sys,
                                                                             parameter_map,
                                                                             state_map,
                                                                             parameter_info,
                                                                             conditions_df,
                                                                             xids[:dynamic])
    θ_scale = _get_xscales(xids, parameter_info)
    θ_indices = ParameterIndices(xindices, xids, θ_scale, mapθ_observable, mapθ_sd, map_ode_problem, maps_condition_id)
    return θ_indices
end

function _get_xids(parameter_info::ParametersInfo, measurements_info::MeasurementsInfo, sys, conditions_df::DataFrame)::Dict{Symbol, Vector{Symbol}}
    @unpack observable_parameters, noise_parameters = measurements_info

    xids_observable = _get_xids_observable_noise(observable_parameters, parameter_info)
    xids_noise = _get_xids_observable_noise(noise_parameters, parameter_info)
    # Non-dynamic parameters are those that only appear in the observable and noise
    # functions, but are not defined noise or observable column of the measurement file.
    # Need to be tracked separately for efficient gradient computations
    xids_nondynamic = _get_xids_nondynamic(xids_observable, xids_noise, sys, parameter_info, conditions_df)
    xids_dynamic = _get_xids_dynamic(xids_observable, xids_noise, xids_nondynamic, parameter_info)
    xids_sys = parameters(sys) .|> Symbol
    xids_not_system = unique(vcat(xids_observable, xids_noise, xids_nondynamic))
    xids_estimate = vcat(xids_dynamic, xids_not_system)

    return Dict(:dynamic => xids_dynamic, :noise => xids_noise, :observable => xids_observable, :nondynamic => xids_nondynamic, :not_system => xids_not_system, :sys => xids_sys, :estimate => xids_estimate)
end

function _get_xindices(xids::Dict{Symbol, Vector{Symbol}})::Dict{Symbol, Vector{Int32}}
    xids_est = xids[:estimate]
    xi_dynamic = Int32[findfirst(x -> x == id, xids_est) for id in xids[:dynamic]]
    xi_noise = Int32[findfirst(x -> x == id, xids_est) for id in xids[:noise]]
    xi_observable = Int32[findfirst(x -> x == id, xids_est) for id in xids[:observable]]
    xi_nondynamic = Int32[findfirst(x -> x == id, xids_est) for id in xids[:nondynamic]]
    xi_not_system = Int32[findfirst(x -> x == id, xids_est) for id in xids[:not_system]]
    return Dict(:dynamic => xi_dynamic, :noise => xi_noise, :observable => xi_observable, :nondynamic => xi_nondynamic, :not_system => xi_not_system)
end

function _get_xscales(xids::Dict{T, Vector{T}}, parameter_info::ParametersInfo)::Dict{T, T} where T<:Symbol
    @unpack parameter_scale, parameter_id = parameter_info
    s = [parameter_scale[findfirst(x -> x == id, parameter_id)] for id in xids[:estimate]]
    return Dict(xids[:estimate] .=> s)
end

function _get_xids_dynamic(observable_ids::T, noise_ids::T, xids_nondynamic::T, parameter_info::ParametersInfo)::T where T<:Vector{Symbol}
    dynamics_xids = Symbol[]
    for id in parameter_info.parameter_id
        if _estimate_parameter(id, parameter_info) == false
            continue
        end
        if id in Iterators.flatten((observable_ids, noise_ids, xids_nondynamic))
            continue
        end
        push!(dynamics_xids, id)
    end
    return dynamics_xids
end

function _get_xids_observable_noise(values, parameter_info::ParametersInfo)::Vector{Symbol}
    ids = Symbol[]
    for value in values
        isempty(value) && continue
        is_number(value) && continue
        # Multiple ids are split by ; in the PEtab table
        for id in Symbol.(split(value, ';'))
            is_number(id) && continue
            if !(id in parameter_info.parameter_id)
                throw(PEtabFileError("Parameter $id in measurement file does not appear " *
                                     "in the PEtab parameters table."))
            end
            id in ids && continue
            if _estimate_parameter(id, parameter_info) == false
                continue
            end
            push!(ids, id)
        end
    end
    return ids
end

function _get_xids_nondynamic(xids_observable::T, xids_noise::T, sys, parameter_info::ParametersInfo, conditions_df::DataFrame)::T where T<:Vector{Symbol}
    xids_condition = _get_xids_condition(sys, parameter_info, conditions_df)
    xids_sys = parameters(sys) .|> Symbol
    xids_nondynamic = Symbol[]
    for id in parameter_info.parameter_id
        if _estimate_parameter(id, parameter_info) == false
            continue
        end
        if id in Iterators.flatten((xids_sys, xids_condition, xids_observable, xids_noise))
            continue
        end
        push!(xids_nondynamic, id)
    end
    return xids_nondynamic
end

function _get_xids_condition(sys, parameter_info::ParametersInfo, conditions_df::DataFrame)::Vector{Symbol}
    xids_sys = parameters(sys) .|> string
    species_sys = _get_state_ids(sys)
    xids_condition = Symbol[]
    for colname in names(conditions_df)
        colname in ["conditionName", "conditionId"] && continue
        if !(colname in Iterators.flatten((xids_sys, species_sys)))
            throw(PEtabFileError("Parameter $colname that dictates an experimental " *
                                 "condition does not appear among the model variables"))
        end
        for condition_variable in Symbol.(conditions_df[!, colname])
            is_number(condition_variable) && continue
            condition_variable == :missing && continue
            if _estimate_parameter(condition_variable, parameter_info) == false
                continue
            end
            condition_variable in xids_condition && continue
            push!(xids_condition, condition_variable)
        end
    end
    return xids_condition
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
function compute_maps_condition(sys,
                                parameter_map,
                                state_map,
                                parameter_info::ParametersInfo,
                                conditions_df::DataFrame,
                                _θ_dynamic_names::Vector{Symbol})::Dict{Symbol,
                                                                        MapConditionId}
    θ_dynamic_names = string.(_θ_dynamic_names)
    n_conditions = nrow(conditions_df)
    model_state_names = string.(states(sys))
    model_state_names = replace.(model_state_names, "(t)" => "")
    all_parameters_ode = string.(parameters(sys))

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

            # In case a condition specific ode-sys parameter is mapped to constant number
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
