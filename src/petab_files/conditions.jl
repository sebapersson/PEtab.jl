"""
        parse_conditions

Parse conditions and build parameter maps for the different parameter types.

There are four type of parameters in a PEtab problem
1. Dynamic (appear in ODE)
2. Noise (appear in noiseParameters columns of measurement table)
3. Observable (appear in observableParameters of measurement table)
4. NonDynamic (do not correspond to any above)

These parameter types need to be treated separately for computing efficient gradients.
This function extracts which parameter is what type, and builds maps for correctly mapping
the parameter during likelihood computations. It further accounts for parameters potentially
only appearing in a certain simulation condition.
"""
function parse_conditions(parameter_info::ParametersInfo, measurements_info::MeasurementsInfo, petab_model::PEtabModel)::ParameterIndices
    @unpack statemap, parametermap, system_mutated, conditions_df = petab_model
    return parse_conditions(parameter_info, measurements_info, system_mutated, parametermap, statemap, conditions_df)
end
function parse_conditions(parameter_info::ParametersInfo, measurements_info::MeasurementsInfo, sys, parametermap, statemap, conditions_df::DataFrame)::ParameterIndices
    _check_conditionids(conditions_df, measurements_info)
    xids = _get_xids(parameter_info, measurements_info, sys, conditions_df)

    # indices for mapping parameters correctly, e.g. from xest -> xdynamic etc...
    # TODO: SII is going to make this much easier (but the reverse will be harder)
    xindices = _get_xindices(xids)
    odeproblem_map = _get_odeproblem_map(xids)
    condition_maps = _get_condition_maps(sys, parametermap, statemap, parameter_info, conditions_df, xids)
    # For each time-point we must build a map that stores if i) noise/obserable parameters
    # are constants, ii) should be estimated, iii) and corresponding index in parameter
    # vector if they should be estimated
    xobservable_maps = _get_map_observable_noise(xids[:observable], measurements_info, parameter_info; observable = true)
    xnoise_maps = _get_map_observable_noise(xids[:noise], measurements_info, parameter_info; observable = false)

    # TODO: Uncertain this should live here
    θ_scale = _get_xscales(xids, parameter_info)
    return ParameterIndices(xindices, xids, θ_scale, xobservable_maps, xnoise_maps, odeproblem_map, condition_maps)
end

function _get_xids(parameter_info::ParametersInfo, measurements_info::MeasurementsInfo, sys, conditions_df::DataFrame)::Dict{Symbol, Vector{Symbol}}
    @unpack observable_parameters, noise_parameters = measurements_info

    # Non-dynamic parameters are those that only appear in the observable and noise
    # functions, but are not defined noise or observable column of the measurement file.
    # Need to be tracked separately for efficient gradient computations
    xids_observable = _get_xids_observable_noise(observable_parameters, parameter_info)
    xids_noise = _get_xids_observable_noise(noise_parameters, parameter_info)
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

function _get_map_observable_noise(xids::Vector{Symbol}, measurements_info::MeasurementsInfo, parameter_info::ParametersInfo; observable::Bool)::Vector{ObservableNoiseMap}
    if observable == true
        parameter_rows = measurements_info.observable_parameters
    else
        parameter_rows = measurements_info.noise_parameters
    end
    maps = Vector{ObservableNoiseMap}(undef, length(parameter_rows))
    for (i, parameter_row) in pairs(parameter_rows)
        # No observable/noise parameter for observation i
        if isempty(parameter_row)
            maps[i] = ObservableNoiseMap(Bool[], Int32[], Float64[], Int32(0), false)
            continue
        end
        # Multiple parameters in PEtab table are separated by ;. These multiple parameters
        # can be constant (hard-coded values), constant parameters, or parameters to
        # according to the PEtab standard
        nvalues_row = count(';', parameter_row) + 1
        estimate = fill(false, nvalues_row)
        constant_values = zeros(Float64, nvalues_row)
        xindices = zeros(Int32, nvalues_row)
        for (j, value) in pairs(split(parameter_row, ';'))
            if is_number(value)
                constant_values[j] = parse(Float64, value)
                continue
            end
            # Must be a parameter
            value = Symbol(value)
            if value in xids
                estimate[j] = true
                xindices[j] = findfirst(x -> x == value, xids)
                continue
            end
            # If a constant parameter defined in the PEtab files
            if value in parameter_info.parameter_id
                ix = findfirst(x -> x == value, parameter_info.parameter_id)
                constant_values[j] = parameter_info.nominal_value[ix]
                continue
            end
            throw(PEtabFileError("Id $value in noise or observable column in measurement " *
                                 "file does not correspond to any id in the parameters " *
                                 "table"))
        end
        single_constant = nvalues_row == 1 && estimate[1] == false
        maps[i] = ObservableNoiseMap(estimate, xindices, constant_values, nvalues_row, single_constant)
    end
    return maps
end

function _get_odeproblem_map(xids::Dict{Symbol, Vector{Symbol}})::MapODEProblem
    sys_to_dynamic = findall(x -> x in xids[:sys], xids[:dynamic]) |> Vector{Int64}
    ids = xids[:dynamic][sys_to_dynamic]
    dynamic_to_sys = Int64[findfirst(x -> x == id, xids[:sys]) for id in ids]
    return MapODEProblem(sys_to_dynamic, dynamic_to_sys)
end

function _get_condition_maps(sys, parametermap, statemap, parameter_info::ParametersInfo, conditions_df::DataFrame, xids::Dict{Symbol, Vector{Symbol}})::Dict{Symbol, ConditionMap}
    species_sys = _get_state_ids(sys)
    xids_sys, xids_dynamic = xids[:sys] .|> string, xids[:dynamic] .|> string
    xids_model = Iterators.flatten((xids_sys, species_sys))

    nconditions = nrow(conditions_df)
    maps = Dict{Symbol, ConditionMap}()
    for i in 1:nconditions
        conditionid = conditions_df[i, :conditionId] |> Symbol
        constant_values, isys_constant_values = Float64[], Int32[]
        ix_dynamic, ix_sys = Int32[], Int32[]
        for variable in names(conditions_df)
            variable in ["conditionName", "conditionId"] && continue
            value = conditions_df[i, variable]
            # Sometimes value can be parsed as string even though it is a Number due to
            # how DataFrames parses columns
            if value isa String && is_number(value)
                value = parse(Float64, value)
            end

            # When the value in the condidtion table maps to a numeric value
            if value isa Real && variable in xids_model
                push!(constant_values, value |> Float64)
                _add_ix_sys!(isys_constant_values, variable, xids_sys)
                continue
            end

            # If value is missing the default SBML values should be used. These are encoded
            # in the parameter- and state-maps
            if ismissing(value) && variable in xids_model
                default_value = _get_default_map_value(variable, parametermap, statemap)
                push!(constant_values, default_value)
                _add_ix_sys!(isys_constant_values, variable, xids_sys)
                continue
            end

            # When the value in the condtitions table maps to a parameter to estimate
            if value in xids_dynamic && variable in xids_model
                push!(ix_dynamic, findfirst(x -> x == value, xids_dynamic))
                _add_ix_sys!(ix_sys, variable, xids_sys)
                continue
            end

            # When the value in the conditions table maps to a constant parameter
            if Symbol(value) in parameter_info.parameter_id && variable in xids_model
                iconstant = findfirst(x -> x == Symbol(value), parameter_info.parameter_id)
                push!(constant_values, parameter_info.nominal_values[iconstant])
                _add_ix_sys!(isys_constant_values, variable, xids_sys)
                continue
            end

            # NaN values are relevant for pre-equilibration and denotes a controll variable
            # should use the value obtained from the forward simulation
            if value isa Real && isnan(value) && variable in species_sys
                push!(constant_values, NaN)
                _add_ix_sys!(isys_constant_values, variable, xids_sys)
                continue
            elseif value isa Real && isnan(value) && variable in xids_model
                throw(PEtabFileError("If a row in conditions file is NaN then the column " *
                                     "header must be a state"))
            end
            throw(PEtabFileError("For condition $conditionid, the value of $variable, " *
                                 "$value, does not correspond to any model parameter, " *
                                 "species, or PEtab parameter. The condition variable " *
                                 "must be a valid model id or a numeric value"))
        end
        maps[conditionid] = ConditionMap(constant_values, isys_constant_values, ix_dynamic, ix_sys)
    end
    return maps
end

function _add_ix_sys!(ix::Vector{Int32}, variable::String, xids_sys::Vector{String})::Nothing
    # When a species initial value is set in the condition table, a new parameter
    # is introduced that sets the initial value. This is required to be able to
    # correctly compute gradients. Hence special handling if variable is not in parameter
    # ids (xid_sys)
    if variable in xids_sys
        push!(ix, findfirst(x -> x == variable, xids_sys))
    else
        state_parameter = "__init__" * variable * "__"
        push!(ix, findfirst(x -> x == state_parameter, xids_sys))
    end
    return nothing
end

# Extract default parameter value from state, or parameter map
function _get_default_map_value(variable::String, parametermap, statemap)::Float64
    species_sys = first.(statemap) .|> string
    species_sys = replace.(species_sys, "(t)" => "")
    xids_sys = first.(parametermap) .|> string

    if variable in xids_sys
        ix = findfirst(x -> x == variable, xids_sys)
        return parametermap[ix].second |> Float64
    end

    # States can have 1 level of allowed recursion (map to a parameter)
    ix = findfirst(x -> x == variable, species_sys)
    value = statemap[ix].second |> string
    if value in xids_sys
        ix = findfirst(x -> x == value, xids_sys)
        return parametermap[ix].second |> Float64
    else
        return parse(Float64, value)
    end
end

function _check_conditionids(conditions_df::DataFrame, measurements_info::MeasurementsInfo)::Nothing
    ncol(conditions_df) == 1 && return nothing
    @unpack pre_equilibration_condition_id, simulation_condition_id = measurements_info
    measurementids = unique(vcat(pre_equilibration_condition_id, simulation_condition_id))
    for conditionid in (conditions_df[!, :conditionId] .|> Symbol)
        conditionid in measurementids && continue
        @warn "Simulation condition id $conditionid does not appear in the measurements " *
              "table. Therefore no measurement corresponds to this id."
    end
    return nothing
end
