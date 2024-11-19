"""
        ParameterIndices

Parse conditions and build parameter maps for the different parameter types.

There are four types of mechanistic parameters in a PEtab problem:
1. Dynamic (appear in ODE)
2. Noise (appear in noiseParameters columns of measurement table)
3. Observable (appear in observableParameters of measurement table)
4. NonDynamic (do not correspond to any above)

There are three different types of neural-net parameters:
1. inode (appear in the ODE)
2. preode (appear before ODE, setting a subset of parameters values)
3. postode (appear after ODE, and should be treated as NonDynamic, as they basically
    belong to this group).

These parameter types need to be treated separately for computing efficient gradients.
This function extracts which parameter is what type, and builds maps for correctly mapping
the parameter during likelihood computations. It further accounts for parameters potentially
only appearing in a certain simulation conditions.
"""
function ParameterIndices(petab_tables::Dict{Symbol, DataFrame}, sys, parametermap,
                          speciemap, nn::Union{Nothing, Dict})::ParameterIndices
    petab_parameters = PEtabParameters(petab_tables[:parameters])
    petab_measurements = PEtabMeasurements(petab_tables[:measurements],
                                           petab_tables[:observables])
    return ParameterIndices(petab_parameters, petab_measurements, sys, parametermap,
                            speciemap, petab_tables[:conditions], nn,
                            petab_tables[:mapping_table])
end
function ParameterIndices(petab_parameters::PEtabParameters,
                          petab_measurements::PEtabMeasurements,
                          model::PEtabModel)::ParameterIndices
    @unpack speciemap, parametermap, sys_mutated, petab_tables = model
    return ParameterIndices(petab_parameters, petab_measurements, sys_mutated, parametermap,
                            speciemap, petab_tables[:conditions], model.nn,
                            petab_tables[:mapping_table])
end
function ParameterIndices(petab_parameters::PEtabParameters,
                          petab_measurements::PEtabMeasurements, sys, parametermap,
                          speciemap, conditions_df::DataFrame, nn::Union{Nothing, Dict},
                          mapping_table::Union{Nothing, DataFrame})::ParameterIndices
    _check_conditionids(conditions_df, petab_measurements)
    mapping_table = _check_mapping_table(mapping_table, nn, petab_parameters, sys, conditions_df)

    xids = _get_xids(petab_parameters, petab_measurements, sys, conditions_df, speciemap,
                     parametermap, nn, mapping_table)

    # indices for mapping parameters correctly, e.g. from xest -> xdynamic etc...
    # TODO: SII is going to make this much easier (but the reverse will be harder)
    xindices = _get_xindices(xids, nn)
    xindices_dynamic = _get_xindices_dynamic(xids, nn)
    xindices_notsys = _get_xindices_notsys(xids, nn)
    odeproblem_map = _get_odeproblem_map(xids, nn)
    condition_maps = _get_condition_maps(sys, parametermap, speciemap, petab_parameters,
                                         conditions_df, mapping_table, xids)
    # For each time-point we must build a map that stores if i) noise/obserable parameters
    # are constants, ii) should be estimated, iii) and corresponding index in parameter
    # vector if they should be estimated
    xobservable_maps = _get_map_observable_noise(xids[:observable], petab_measurements,
                                                 petab_parameters; observable = true)
    xnoise_maps = _get_map_observable_noise(xids[:noise], petab_measurements,
                                            petab_parameters;
                                            observable = false)
    # If a neural-network sets values for a subset of model parameters, for efficent AD on
    # said network, it is neccesary to pre-compute the input, pre-allocate the output,
    # and build a map for which parameters in xdynamic the network maps to.
    nn_pre_ode_maps = _get_nn_pre_ode_maps(conditions_df, xids, petab_parameters, mapping_table, nn, sys)

    xscale = _get_xscales(xids, petab_parameters)
    _get_xnames_ps!(xids, xscale)

    return ParameterIndices(xindices, xids, xindices_notsys, xindices_dynamic, xscale,
                            xobservable_maps, xnoise_maps, odeproblem_map, condition_maps,
                            nn_pre_ode_maps)
end

function _get_xids(petab_parameters::PEtabParameters, petab_measurements::PEtabMeasurements,
                   sys::ModelSystem, conditions_df::DataFrame, speciemap, parametermap,
                   nn::Union{Nothing, Dict}, mapping_table)::Dict{Symbol, Vector{Symbol}}
    @unpack observable_parameters, noise_parameters = petab_measurements
    # xids in the ODESystem in correct order
    xids_sys = _get_xids_sys_order(sys, speciemap, parametermap)

    # Parameters related to neural networks (data-driven models)
    xids_nn = _get_xids_nn(nn)
    xids_nn_in_ode = _get_xids_nn_in_ode(xids_nn, sys)
    xids_nn_pre_ode = _get_xids_nn_pre_ode(mapping_table, sys)
    xids_nn_nondynamic = _get_xids_nn_nondynamic(xids_nn, xids_nn_in_ode, xids_nn_pre_ode)
    # Parameter which are input to a neural net, and are estimated. These must be tracked
    # for gradients
    xids_nn_input_est = _get_xids_nn_input_est(mapping_table, conditions_df, petab_parameters, sys)
    # If a Neural-Net sets the values for a parameter, in practice for gradient computations
    # the derivative of the parameter is needed to compute the network gradient following
    # the chain-rule. Therefore, these parameters must be tracked such that they can be
    # a part of xdynamic.
    xids_nn_pre_ode_output = _get_xids_nn_pre_ode_output(mapping_table, xids_sys)

    # Mechanistic (none neural-net parameters). Note non-dynamic parameters are those that
    # only appear in the observable and noise functions, but are not defined noise or
    # observable column of the measurement file.
    xids_observable = _get_xids_observable_noise(observable_parameters, petab_parameters)
    xids_noise = _get_xids_observable_noise(noise_parameters, petab_parameters)
    xids_nondynamic_mech = _get_xids_nondynamic_mech(xids_observable, xids_noise, xids_nn, xids_nn_input_est, sys, petab_parameters, conditions_df, mapping_table)
    xids_dynamic_mech = _get_xids_dynamic_mech(xids_observable, xids_noise, xids_nondynamic_mech, xids_nn, petab_parameters)
    # If a parameter is in xids_nn_input_est and does not appear in the sys, it must be
    # treated as a xdynamic parameter, since it informs the ODE (it is not a nondynamic)
    _add_xids_nn_input_est_only!(xids_dynamic_mech, xids_nn_input_est)
    xids_not_system_mech = unique(vcat(xids_observable, xids_noise, xids_nondynamic_mech))

    xids_estimate = vcat(xids_dynamic_mech, xids_not_system_mech, xids_nn)
    xids_petab = petab_parameters.parameter_id
    return Dict(:dynamic_mech => xids_dynamic_mech, :noise => xids_noise, :nn => xids_nn,
                :observable => xids_observable, :nondynamic_mech => xids_nondynamic_mech,
                :not_system_mech => xids_not_system_mech, :sys => xids_sys,
                :estimate => xids_estimate, :petab => xids_petab,
                :nn_in_ode => xids_nn_in_ode, :nn_pre_ode => xids_nn_pre_ode,
                :nn_pre_ode_outputs => xids_nn_pre_ode_output,
                :nn_nondynamic => xids_nn_nondynamic)
end

function _get_xindices(xids::Dict{Symbol, Vector{Symbol}}, nn)::Dict{Symbol, Vector{Int32}}
    xids_est = xids[:estimate]
    xi_dynamic_mech = Int32[findfirst(x -> x == id, xids_est) for id in xids[:dynamic_mech]]
    xi_noise = Int32[findfirst(x -> x == id, xids_est) for id in xids[:noise]]
    xi_observable = Int32[findfirst(x -> x == id, xids_est) for id in xids[:observable]]
    xi_nondynamic = Int32[findfirst(x -> x == id, xids_est) for id in xids[:nondynamic_mech]]
    xi_not_system_mech = Int32[findfirst(x -> x == id, xids_est) for id in xids[:not_system_mech]]
    xindices = Dict(:dynamic_mech => xi_dynamic_mech, :noise => xi_noise, :observable => xi_observable,
                    :nondynamic_mech => xi_nondynamic, :not_system_mech => xi_not_system_mech)
    # Each neural network input is in a sense its own parameter component, hence they need
    # to be treated separately.
    if !isnothing(nn)
        istart = length(xids_est) - length(xids[:nn])
        for id in keys(nn)
            pid = ("p_" * string(id)) |> Symbol
            np = _get_n_net_parameters(nn, [pid])
            xindices[pid] = (istart+1):(istart + np)
            istart = xindices[pid][end]
        end
    end
    xi_not_system_nn = Int32[]
    if !isnothing(nn)
        for pid in xids[:nn_nondynamic]
            xi_not_system_nn = vcat(xi_not_system_nn, xindices[pid])
        end
    end
    xindices[:not_system_nn] = xi_not_system_nn
    xindices[:not_system_tot] = vcat(xi_not_system_mech, xi_not_system_nn)
    return xindices
end

function _get_xindices_dynamic(xids::Dict{Symbol, Vector{Symbol}}, nn)::Dict{Symbol, Vector{Int32}}
    xindices = Dict{Symbol, Vector{Int32}}()
    # Get indices for mechanstic parameters in xdynamic
    xindices[:xdynamic_to_mech] = Int32.(1:length(xids[:dynamic_mech]))
    # Get indicies in xest for all dynamic parameters (mechanistic + neural net)
    xi_xest_to_xdynamic = Int32[findfirst(x -> x == id, xids[:estimate]) for id in xids[:dynamic_mech]]
    xi_nn_in_ode = Int32[]
    istart = length(xids[:estimate]) - length(xids[:nn])
    for pid in xids[:nn_in_ode]
        np = _get_n_net_parameters(nn, [pid])
        xi_xest_to_xdynamic = vcat(xi_xest_to_xdynamic, (istart+1):(istart + np))
        xi_nn_in_ode = vcat(xi_nn_in_ode, (istart+1):(istart + np))
        istart += np
    end
    xindices[:nn_in_ode] = xi_nn_in_ode
    # Get indices in xdynamic for neural-net output parameters (these are included in
    # xdynamic for gradient to be computed when split_over_conditions = false, otherwise
    # they are just kept as constant)
    xi_nn_pre_ode = Int32[]
    for pid in xids[:nn_pre_ode]
        np = _get_n_net_parameters(nn, [pid])
        xi_xest_to_xdynamic = vcat(xi_xest_to_xdynamic, (istart+1):(istart + np))
        xi_nn_pre_ode = vcat(xi_nn_pre_ode, (istart+1):(istart + np))
        istart += np
    end
    xindices[:nn_pre_ode] = xi_nn_pre_ode
    xindices[:xest_to_xdynamic] = xi_xest_to_xdynamic
    # Get indices in xdynamic for each neural net inside of the ODE, in order to map
    # dynamic parameters to neural nets
    istart = length(xindices[:xdynamic_to_mech])
    for pid in xids[:nn_in_ode]
        np = _get_n_net_parameters(nn, [pid])
        xindices[pid] = (istart+1):(istart + np)
        istart += np
    end
    for pid in xids[:nn_pre_ode]
        np = _get_n_net_parameters(nn, [pid])
        xindices[pid] = (istart+1):(istart + np)
        istart += np
    end
    # This only holds when xdynamic is expanded during split_over_conditions = true
    # to get the gradient of neural-net set parameters
    np = length(xindices[:xest_to_xdynamic])
    xindices[:xdynamic_to_nnout] = (np+1):(np + length(xids[:nn_pre_ode_outputs]))
    return xindices
end

function _get_xindices_notsys(xids::Dict{Symbol, Vector{Symbol}},
                              nn::Union{Nothing, Dict})::Dict{Symbol, Vector{Int32}}
    ins = xids[:not_system_mech]
    ixnoise = Int32[findfirst(x -> x == id, ins) for id in xids[:noise]]
    ixobservable = Int32[findfirst(x -> x == id, ins) for id in xids[:observable]]
    ixnondynamic_mech_mech = Int32[findfirst(x -> x == id, ins) for id in xids[:nondynamic_mech]]
    xindices_notsys = Dict(:noise => ixnoise, :observable => ixobservable, :nondynamic_mech => ixnondynamic_mech_mech)
    # Nondynamic neural-net parameters, mapping from nondynamic to neural nets
    istart = length(ins)
    for pid in xids[:nn_nondynamic]
        np = _get_n_net_parameters(nn, [pid])
        xindices_notsys[pid] = (istart+1):(istart + np)
        istart += np
    end
    return xindices_notsys
end

function _get_xscales(xids::Dict{T, Vector{T}},
                      petab_parameters::PEtabParameters)::Dict{T, T} where {T <: Symbol}
    @unpack parameter_scale, parameter_id = petab_parameters
    s = [parameter_scale[findfirst(x -> x == id, parameter_id)] for id in xids[:estimate]]
    return Dict(xids[:estimate] .=> s)
end

function _get_xids_dynamic_mech(xids_observable::T, xids_noise::T, xids_nondynamic_mech::T, xids_nn::T, petab_parameters::PEtabParameters)::T where {T <: Vector{Symbol}}
    dynamics_xids = Symbol[]
    _ids = Iterators.flatten((xids_observable, xids_noise, xids_nondynamic_mech, xids_nn))
    for id in petab_parameters.parameter_id
        _estimate_parameter(id, petab_parameters) == false && continue
        id in _ids && continue
        push!(dynamics_xids, id)
    end
    return dynamics_xids
end

function _get_xids_nn_pre_ode_output(mapping_table::DataFrame, xids_sys::Vector{Symbol})::Vector{Symbol}
    out = Symbol[]
    for i in 1:nrow(mapping_table)
        io_value = mapping_table[i, :ioValue]
        io_value ∉ xids_sys && continue
        if io_value in out
            throw(PEtabInputError("Only one neural network output can map to paramter \
                                   $(io_value)"))
        end
        push!(out, io_value)
    end
    return out
end

function _get_xids_observable_noise(values, petab_parameters::PEtabParameters)::Vector{Symbol}
    ids = Symbol[]
    for value in values
        isempty(value) && continue
        is_number(value) && continue
        # Multiple ids are split by ; in the PEtab table
        for id in Symbol.(split(value, ';'))
            is_number(id) && continue
            if !(id in petab_parameters.parameter_id)
                throw(PEtabFileError("Parameter $id in measurement file does not appear " *
                                     "in the PEtab parameters table."))
            end
            id in ids && continue
            if _estimate_parameter(id, petab_parameters) == false
                continue
            end
            push!(ids, id)
        end
    end
    return ids
end

function _get_xids_nondynamic_mech(xids_observable::T, xids_noise::T, xids_nn::T, xids_nn_input_est::T, sys::ModelSystem, petab_parameters::PEtabParameters, conditions_df::DataFrame, mapping_table::DataFrame)::T where {T <: Vector{Symbol}}
    xids_condition = _get_xids_condition(sys, petab_parameters, conditions_df, mapping_table)
    xids_sys = _get_xids_sys(sys)
    xids_nondynamic_mech = Symbol[]
    _ids = Iterators.flatten((xids_sys, xids_condition, xids_observable, xids_noise, xids_nn, xids_nn_input_est))
    for id in petab_parameters.parameter_id
        _estimate_parameter(id, petab_parameters) == false && continue
        id in _ids && continue
        push!(xids_nondynamic_mech, id)
    end
    return xids_nondynamic_mech
end

function _get_xids_condition(sys, petab_parameters::PEtabParameters, conditions_df::DataFrame, mapping_table::DataFrame)::Vector{Symbol}
    xids_sys = parameters(sys) .|> string
    species_sys = _get_state_ids(sys)
    net_inputs = String[]
    if !isempty(mapping_table)
        for netid in unique(mapping_table[!, :netId])
            _net_inputs = _get_net_values(mapping_table, netid, :inputs) .|> string
            net_inputs = vcat(net_inputs, _net_inputs)
        end
    end
    problem_variables = Iterators.flatten((xids_sys, species_sys, net_inputs))
    xids_condition = Symbol[]
    for colname in names(conditions_df)
        colname in ["conditionName", "conditionId"] && continue
        if !(colname in problem_variables)
            throw(PEtabFileError("Parameter $colname that dictates an experimental " *
                                 "condition does not appear among the model variables"))
        end
        for condition_variable in Symbol.(conditions_df[!, colname])
            is_number(condition_variable) && continue
            condition_variable == :missing && continue
            should_est = _estimate_parameter(condition_variable, petab_parameters)
            if should_est == false
                continue
            end
            if string(colname) in net_inputs && should_est == true
                throw(PEtabFileError("Neural net input variable $(condition_variable) for \
                                      condition variable $colname is not allowed to be a \
                                      parameter that is estimated"))
            end
            condition_variable in xids_condition && continue
            push!(xids_condition, condition_variable)
        end
    end
    return xids_condition
end

function _get_map_observable_noise(xids::Vector{Symbol},
                                   petab_measurements::PEtabMeasurements,
                                   petab_parameters::PEtabParameters;
                                   observable::Bool)::Vector{ObservableNoiseMap}
    if observable == true
        parameter_rows = petab_measurements.observable_parameters
    else
        parameter_rows = petab_measurements.noise_parameters
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
            if value in petab_parameters.parameter_id
                ix = findfirst(x -> x == value, petab_parameters.parameter_id)
                constant_values[j] = petab_parameters.nominal_value[ix]
                continue
            end
            throw(PEtabFileError("Id $value in noise or observable column in measurement " *
                                 "file does not correspond to any id in the parameters " *
                                 "table"))
        end
        single_constant = nvalues_row == 1 && estimate[1] == false
        maps[i] = ObservableNoiseMap(estimate, xindices, constant_values, nvalues_row,
                                     single_constant)
    end
    return maps
end

function _get_odeproblem_map(xids::Dict{Symbol, Vector{Symbol}}, nn::Union{Dict, Nothing})::MapODEProblem
    dynamic_to_sys, sys_to_dynamic = Int32[], Int32[]
    sys_to_dynamic_nn, sys_to_nn_pre_ode_output = Int32[], Int32[]
    isys = 1
    for (i, id_xdynmaic) in pairs(xids[:dynamic_mech])
        for id_sys in xids[:sys]
            if id_sys in xids[:nn]
                isys += _get_n_net_parameters(nn, [id_sys])
                continue
            end
            if id_sys == id_xdynmaic
                push!(dynamic_to_sys, i)
                push!(sys_to_dynamic, isys)
                break
            end
            isys += 1
        end
        isys = 1
    end
    isys = 1
    for id_nn_output in xids[:nn_pre_ode_outputs]
        for id_sys in xids[:sys]
            if id_sys in xids[:nn]
                isys += _get_n_net_parameters(nn, [id_sys])
                continue
            end
            if id_sys == id_nn_output
                push!(sys_to_nn_pre_ode_output, isys)
                break
            end
            isys += 1
        end
        isys = 1
    end
    isys = 1
    for id_sys in xids[:sys]
        if id_sys in xids[:nn]
            np = _get_n_net_parameters(nn, [id_sys]) - 1
            sys_to_dynamic_nn = vcat(sys_to_dynamic_nn, collect(isys:(isys+np)))
            isys += np
        end
        isys += 1
    end
    dynamic_to_sys = findall(x -> x in xids[:sys], xids[:dynamic_mech]) |> Vector{Int64}
    ids = xids[:dynamic_mech][dynamic_to_sys]
    sys_to_dynamic = Int64[findfirst(x -> x == id, xids[:sys]) for id in ids]
    return MapODEProblem(sys_to_dynamic, dynamic_to_sys, sys_to_dynamic_nn, sys_to_nn_pre_ode_output)
end

function _get_condition_maps(sys, parametermap, speciemap, petab_parameters::PEtabParameters, conditions_df::DataFrame, mapping_table::DataFrame, xids::Dict{Symbol, Vector{Symbol}})::Dict{Symbol, ConditionMap}
    species_sys = _get_state_ids(sys)
    xids_sys, xids_dynamic = xids[:sys] .|> string, xids[:dynamic_mech] .|> string
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

            if value isa Real && Symbol(variable) in mapping_table[!, :ioValue]
                continue
            end

            # If value is missing the default SBML values should be used. These are encoded
            # in the parameter- and state-maps
            if ismissing(value) && variable in xids_model
                default_value = _get_default_map_value(variable, parametermap, speciemap)
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
            if Symbol(value) in petab_parameters.parameter_id && variable in xids_model
                iconstant = findfirst(x -> x == Symbol(value),
                                      petab_parameters.parameter_id)
                push!(constant_values, petab_parameters.nominal_value[iconstant])
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

            # A variable can be one of the neural net inputs. The map is then built later
            # when building the input function for the NN
            if Symbol(variable) in mapping_table[!, :ioValue]
                continue
            end

            throw(PEtabFileError("For condition $conditionid, the value of $variable, " *
                                 "$value, does not correspond to any model parameter, " *
                                 "species, or PEtab parameter. The condition variable " *
                                 "must be a valid model id or a numeric value"))
        end
        maps[conditionid] = ConditionMap(constant_values, isys_constant_values, ix_dynamic,
                                         ix_sys)
    end
    return maps
end

function _add_ix_sys!(ix::Vector{Int32}, variable::String,
                      xids_sys::Vector{String})::Nothing
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
function _get_default_map_value(variable::String, parametermap, speciemap)::Float64
    species_sys = first.(speciemap) .|> string
    species_sys = replace.(species_sys, "(t)" => "")
    xids_sys = first.(parametermap) .|> string

    if variable in xids_sys
        ix = findfirst(x -> x == variable, xids_sys)
        return parametermap[ix].second |> Float64
    end

    # States can have 1 level of allowed recursion (map to a parameter)
    ix = findfirst(x -> x == variable, species_sys)
    value = speciemap[ix].second |> string
    if value in xids_sys
        ix = findfirst(x -> x == value, xids_sys)
        return parametermap[ix].second |> Float64
    else
        return parse(Float64, value)
    end
end

function _check_conditionids(conditions_df::DataFrame,
                             petab_measurements::PEtabMeasurements)::Nothing
    ncol(conditions_df) == 1 && return nothing
    @unpack pre_equilibration_condition_id, simulation_condition_id = petab_measurements
    measurementids = unique(vcat(pre_equilibration_condition_id, simulation_condition_id))
    for conditionid in (conditions_df[!, :conditionId] .|> Symbol)
        conditionid in measurementids && continue
        @warn "Simulation condition id $conditionid does not appear in the measurements " *
              "table. Therefore no measurement corresponds to this id."
    end
    return nothing
end

function _get_xids_sys_order(sys::ModelSystem, speciemap, parametermap)::Vector{Symbol}
    if sys isa ODEProblem
        return collect(keys(sys.p))
    end
    # This is a hack untill SciMLSensitivity integrates with the SciMLStructures interface.
    # Basically allows the parameters in the system to be retreived in the order they
    # appear in the ODESystem later on
    _p = parameters(sys)
    out = similar(_p)
    if sys isa SDESystem
        prob = SDEProblem(sys, speciemap, [0.0, 5e3], parametermap)
    else
        prob = ODEProblem(sys, speciemap, [0.0, 5e3], parametermap; jac = true)
    end
    maps = ModelingToolkit.getp(prob, _p)
    for (i, map) in pairs(maps.getters)
        out[map.idx.idx] = _p[i]
    end
    return Symbol.(out)
end

function _get_xids_sys(sys::ModelSystem)::Vector{Symbol}
    return sys isa ODEProblem ? collect(keys(sys.p)) : Symbol.(parameters(sys))
end

function _get_xids_nn(nn::Union{Nothing, Dict})::Vector{Symbol}
    isnothing(nn) && return Symbol[]
    return ("p_" .* string.(collect(keys(nn)))) .|> Symbol
end

function _get_xids_nn_in_ode(xids_nn::Vector{Symbol}, sys)::Vector{Symbol}
    !(sys isa ODEProblem) && return Symbol[]
    xids_nn_in_ode = Symbol[]
    for id in xids_nn
        !haskey(sys.p, id) && continue
        push!(xids_nn_in_ode, id)
    end
    return xids_nn_in_ode
end

function _get_xids_nn_pre_ode(mapping_table::DataFrame, sys::ModelSystem)::Vector{Symbol}
    isempty(mapping_table) && return Symbol[]
    xids_sys = _get_xids_sys(sys)
    out = Symbol[]
    for netid in Symbol.(unique(mapping_table[!, :netId]))
        outputs = _get_net_values(mapping_table, Symbol(netid), :outputs) .|> Symbol
        if all([output in xids_sys for output in outputs])
            push!(out, Symbol("p_$netid"))
        end
    end
    return out
end

function _get_xids_nn_nondynamic(xids_nn::T, xids_nn_in_ode::T, xids_nn_pre_ode::T)::T where T <: Vector{Symbol}
    out = Symbol[]
    for id in xids_nn
        id in Iterators.flatten((xids_nn_in_ode, xids_nn_pre_ode)) && continue
        push!(out, id)
    end
    return out
end

function _get_xids_nn_input_est(mapping_table::DataFrame, conditions_df::DataFrame, petab_parameters::PEtabParameters, sys)::Vector{Symbol}
    isempty(mapping_table) && return Symbol[]
    out = Symbol[]
    for netid in Symbol.(unique(mapping_table[!, :netId]))
        inputs = _get_net_values(mapping_table, netid, :inputs) .|> Symbol
        input_variables = _get_nn_input_variables(inputs, conditions_df, petab_parameters, sys)
        for input_variable in input_variables
            !(input_variable in petab_parameters.parameter_id) && continue
            ip = findfirst(x -> x == input_variable, petab_parameters.parameter_id)
            if petab_parameters.estimate[ip] == true
                push!(out, input_variable)
            end
        end
    end
    return out
end

function _add_xids_nn_input_est_only!(xids_dynamic_mech::Vector{Symbol}, xids_nn_input_est::Vector{Symbol})::Nothing
    for id in xids_nn_input_est
        id in xids_dynamic_mech && continue
        push!(xids_dynamic_mech, id)
    end
    return nothing
end

function _xdynamic_in_event_cond(model_SBML::SBMLImporter.ModelSBML,
                                 xindices::ParameterIndices,
                                 petab_tables::Dict{Symbol, DataFrame})::Bool
    xids_sys_in_xdynamic = _get_xids_sys_order_in_xdynamic(xindices, petab_tables[:conditions])
    for event in values(model_SBML.events)
        for xid in xids_sys_in_xdynamic
            trigger_alt = SBMLImporter._replace_variable(event.trigger, xid, "")
            if trigger_alt != event.trigger
                return true
            end
        end
    end
    return false
end

function _get_xids_sys_order_in_xdynamic(xindices::ParameterIndices,
                                   conditions_df::DataFrame)::Vector{String}
    xids_sys = xindices.xids[:sys]
    xids_sys_in_xdynamic = filter(x -> x in xids_sys, xindices.xids[:dynamic_mech])
    # Extract sys parameters where an xdynamic via the condition table maps to a parameter
    # in the ODE
    xids_condition = filter(x -> !(x in xids_sys), xindices.xids[:dynamic_mech])
    for variable in propertynames(conditions_df)
        !(variable in xids_sys) && continue
        for xid_condition in string.(xids_condition)
            if xid_condition in conditions_df[!, variable]
                push!(xids_sys_in_xdynamic, variable)
            end
        end
    end
    unique!(xids_sys_in_xdynamic)
    return xids_sys_in_xdynamic .|> string
end

function _get_xnames_ps!(xids::Dict{Symbol, Vector{Symbol}}, xscale)::Nothing
    out = similar(xids[:estimate])
    for (i, id) in pairs(xids[:estimate])
        scale = xscale[id]
        if scale == :lin
            out[i] = id
            continue
        end
        out[i] = "$(scale)_$id" |> Symbol
    end
    xids[:estimate_ps] = out
    return nothing
end

function _check_mapping_table(mapping_table::Union{DataFrame, Nothing}, nn::Union{Nothing, Dict}, petab_parameters::PEtabParameters, sys, conditions_df::DataFrame)::DataFrame
    if isempty(mapping_table) || isnothing(nn)
        return DataFrame()
    end
    state_ids = _get_state_ids(sys) .|> Symbol
    xids_sys = _get_xids_sys(sys)
    model_variables = Iterators.flatten((state_ids, xids_sys, petab_parameters.parameter_id))

    # Sanity check ioId column
    pattern = r"^(input\d|output\d)$"
    for io_id in string.(mapping_table[!, :ioId])
        if !occursin(pattern, io_id)
            throw(PEtabInputError("In mapping table, in ioId column allowed values are \
                                   only input{:digit} or output{:digit} where digit is \
                                   the number of the input/output to the network. Not \
                                   $io_id"))
        end
    end

    # Sanity check ioValue column (input and outputs to neural-net)
    for netid in Symbol.(unique(mapping_table[!, :netId]))
        if !haskey(nn, netid)
            throw(PEtabInputError("Neural network id $netid provided in the mapping table \
                                   does not correspond to any Neural Network id provided \
                                   in the PEtab problem"))
        end
        outputs = _get_net_values(mapping_table, netid, :outputs) .|> Symbol
        inputs = _get_net_values(mapping_table, netid, :inputs) .|> Symbol
        input_variables = _get_nn_input_variables(inputs, conditions_df, petab_parameters, sys)

        # If input_variables is empty all inputs are numbers which can always be handled
        if isempty(input_variables)
            continue
        # If all outputs maps to ODEProblem parameters, it is a pre-nn ODE case. In this
        # case all input variables must be PEtab parameters (or numbers which are already
        # filtered out from input_variables)
        elseif all([output in xids_sys for output in outputs])
            if !all([ipv in petab_parameters.parameter_id for ipv in input_variables])
                throw(PEtabInputError("If mapping table output is ODEProblem parameters \
                                       input must be a PEtabParameter, this does not hold \
                                       for $inputs"))
            end
            continue
        # If all inputs maps to model variables, the nn must be in the observable formula,
        # else something is wrong
        elseif !all([ipv in model_variables for ipv in input_variables])
            throw(PEtabInputError("If mapping table output is a parameter in the \
                                   observable/sd formula, input must be a PEtabParameter, \
                                   model specie, or model parameter. Does not hold for \
                                   all inputs in $inputs"))
            continue
        end
    end
    return DataFrame(netId = Symbol.(mapping_table[!, :netId]),
                     ioId = Symbol.(mapping_table[!, :ioId]),
                     ioValue = Symbol.(mapping_table[!, :ioValue]))
end

function _get_nn_pre_ode_maps(conditions_df::DataFrame, xids::Dict{Symbol, Vector{Symbol}}, petab_parameters::PEtabParameters, mapping_table::DataFrame, nn, sys)::Dict{Symbol, Dict{Symbol, NNPreODEMap}}
    nconditions = nrow(conditions_df)
    maps = Dict{Symbol, Dict{Symbol, NNPreODEMap}}()
    isempty(xids[:nn_pre_ode]) && return maps
    for i in 1:nconditions
        conditionid = conditions_df[i, :conditionId] |> Symbol
        maps_nn = Dict{Symbol, NNPreODEMap}()
        for pnnid in xids[:nn_pre_ode]
            netid = string(pnnid)[3:end] |> Symbol
            outputs = _get_net_values(mapping_table, netid, :outputs) .|> Symbol
            inputs = _get_net_values(mapping_table, netid, :inputs) .|> Symbol
            input_variables = _get_nn_input_variables(inputs, DataFrame(conditions_df[i, :]), petab_parameters, sys; keep_numbers = true)
            ninputs = length(input_variables)

            # Get values for the inputs. For the pre-ODE inputs can either be constant
            # values (numeric) or a parameter, which is treated as a xdynamic parameter
            constant_inputs, iconstant_inputs = zeros(Float64, 0), zeros(Int32, 0)
            ixdynamic_mech_inputs, ixdynamic_inputs = zeros(Int32, 0), zeros(Int32, 0)
            for (i, input_variable) in pairs(input_variables)
                if is_number(input_variable)
                    val = parse(Float64, string(input_variable))
                    push!(constant_inputs, val)
                    push!(iconstant_inputs, i)
                    continue
                end
                # Has to now be a PEtabParameter, that can either be constant or be
                # estimated
                ip = findfirst(x -> x == input_variable, petab_parameters.parameter_id)
                if petab_parameters.estimate[ip] == false
                    push!(constant_inputs, petab_parameters.nominal_value[ip])
                    push!(iconstant_inputs, i)
                    continue
                end
                # Parameter that is estimated (and part of ixdynamic)
                ixmech = findfirst(x -> x == input_variable, xids[:dynamic_mech])
                push!(ixdynamic_mech_inputs, ixmech)
                push!(ixdynamic_inputs, i)
            end
            nxdynamic_inputs = length(ixdynamic_mech_inputs)

            # Indicies for correctly mapping the output. The outputs are stored in a
            # separate vector of order xids[:nn_pre_ode_outputs], which ix_nn_outputs
            # stores the index for. The outputs maps to parameters in sys, which
            # ioutput_sys stores. Lastly, for split_over_conditions = true the
            # gradient of the output variables is needed, ix_nn_outputs_grad stores
            # the indices in xdynamic_grad for the outputs (the indices depend on the
            # condition so the Vector is only pre-allocated here).
            noutputs = length(outputs)
            ix_nn_outputs = zeros(Int32, noutputs)
            ioutput_sys = zeros(Int32, noutputs)
            ix_nn_outputs_grad = zeros(Int32, noutputs)
            for (i, output_variable) in pairs(outputs)
                io = findfirst(x -> x == output_variable, xids[:nn_pre_ode_outputs])
                ix_nn_outputs[i] = io
                isys = 1
                # TODO: To common pattern, refactor into a function
                for id_sys in xids[:sys]
                    if id_sys in xids[:nn]
                        np = _get_n_net_parameters(nn, [id_sys]) - 1
                        isys += np
                    end
                    if id_sys == output_variable
                        ioutput_sys[i] = isys
                        break
                    end
                    isys += 1
                end
            end
            maps_nn[pnnid] = NNPreODEMap(constant_inputs, iconstant_inputs, ixdynamic_mech_inputs, ixdynamic_inputs, ninputs, nxdynamic_inputs, noutputs, ix_nn_outputs, ix_nn_outputs_grad, ioutput_sys)
        end
        maps[conditionid] = maps_nn
    end
    return maps
end

function _get_nn_input_variables(inputs::Vector{Symbol}, conditions_df::DataFrame, petab_parameters::PEtabParameters, sys; keep_numbers::Bool = false)::Vector{Symbol}
    state_ids = _get_state_ids(sys) .|> Symbol
    xids_sys = _get_xids_sys(sys)
    input_variables = Symbol[]
    for input in inputs
        if is_number(input)
            if keep_numbers == true
                push!(input_variables, input)
            end
            continue
        end
        if input in petab_parameters.parameter_id
            push!(input_variables, input)
            continue
        end
        if input in Iterators.flatten((state_ids, xids_sys))
            push!(input_variables, input)
            continue
        end
        if input in propertynames(conditions_df)
            for condition_value in Symbol.(conditions_df[!, input])
                _input_variables = _get_nn_input_variables([condition_value], conditions_df, petab_parameters, sys; keep_numbers = keep_numbers)
                input_variables = vcat(input_variables, _input_variables)
            end
            continue
        end
        throw(PEtabInputError("Input $input to neural-network cannot be found among ODE \
                               variables, PEtab parameters, or in the conditions table"))
    end
    return input_variables
end
