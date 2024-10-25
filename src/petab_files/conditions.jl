"""
        ParameterIndices

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
function ParameterIndices(petab_tables::Dict{Symbol, DataFrame}, sys, parametermap,
                          speciemap)::ParameterIndices
    petab_parameters = PEtabParameters(petab_tables[:parameters])
    petab_measurements = PEtabMeasurements(petab_tables[:measurements],
                                           petab_tables[:observables])
    return ParameterIndices(petab_parameters, petab_measurements, sys, parametermap,
                            speciemap, petab_tables[:conditions], nothing,
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
    mapping_table = _check_mapping_table(mapping_table, nn, petab_parameters, sys)
    xids = _get_xids(petab_parameters, petab_measurements, sys, conditions_df, speciemap,
                     parametermap, nn, mapping_table)

    # indices for mapping parameters correctly, e.g. from xest -> xdynamic etc...
    # TODO: SII is going to make this much easier (but the reverse will be harder)
    xindices = _get_xindices(xids, nn)
    xindices_dynamic = _get_xindices_dynamic(xids, nn)
    xindices_notsys = _get_xindices_notsys(xids)
    odeproblem_map = _get_odeproblem_map(xids, nn)
    condition_maps = _get_condition_maps(sys, parametermap, speciemap, petab_parameters,
                                         conditions_df, xids)
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
    nn_pre_ode_maps = _get_nn_pre_ode_maps(conditions_df, odeproblem_map, condition_maps,
                                           xids, petab_parameters, xindices_dynamic,
                                           mapping_table, nn)

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

    xids_sys = _get_sys_parameters(sys, speciemap, parametermap)
    # Non-dynamic parameters are those that only appear in the observable and noise
    # functions, but are not defined noise or observable column of the measurement file.
    # Need to be tracked separately for efficient gradient computations
    if isnothing(nn)
        xids_nn = Symbol[]
    else
        xids_nn = ("p_" .* string.(collect(keys(nn)))) .|> Symbol
    end
    xids_nn_in_ode = _get_xids_nn_in_ode(xids_nn, sys)
    xids_nn_pre_ode = _get_xids_nn_pre_ode(mapping_table, sys)
    xids_observable = _get_xids_observable_noise(observable_parameters, petab_parameters)
    xids_noise = _get_xids_observable_noise(noise_parameters, petab_parameters)
    xids_nondynamic = _get_xids_nondynamic(xids_observable, xids_noise, xids_nn_pre_ode, sys, petab_parameters, conditions_df)
    xids_dynamic_mech = _get_xids_dynamic_mech(xids_observable, xids_noise, xids_nondynamic, xids_nn, petab_parameters)

    xids_not_system = unique(vcat(xids_observable, xids_noise, xids_nondynamic))
    xids_estimate = vcat(xids_dynamic_mech, xids_not_system, xids_nn)
    xids_petab = petab_parameters.parameter_id

    # If a Neural-Net sets the values for a parameter, in practice for gradient computations
    # the derivative of the parameter is needed to compute the network gradient following
    # the chain-rule. Therefore, these parameters must be tracked such that they can be
    # a part of xdynamic. As xdynamic is never exposed to the user, this is not noticed
    # by the user.
    xids_nn_pre_ode_output = _get_xids_nn_pre_ode_output(mapping_table, xids_sys)

    return Dict(:dynamic_mech => xids_dynamic_mech, :noise => xids_noise, :nn => xids_nn,
                :observable => xids_observable, :nondynamic => xids_nondynamic,
                :not_system => xids_not_system, :sys => xids_sys,
                :estimate => xids_estimate, :petab => xids_petab,
                :nn_in_ode => xids_nn_in_ode, :nn_pre_ode => xids_nn_pre_ode,
                :nn_pre_ode_output => xids_nn_pre_ode_output)
end

function _get_xindices(xids::Dict{Symbol, Vector{Symbol}}, nn)::Dict{Symbol, Vector{Int32}}
    xids_est = xids[:estimate]
    xi_dynamic_mech = Int32[findfirst(x -> x == id, xids_est) for id in xids[:dynamic_mech]]
    xi_noise = Int32[findfirst(x -> x == id, xids_est) for id in xids[:noise]]
    xi_observable = Int32[findfirst(x -> x == id, xids_est) for id in xids[:observable]]
    xi_nondynamic = Int32[findfirst(x -> x == id, xids_est) for id in xids[:nondynamic]]
    xi_not_system = Int32[findfirst(x -> x == id, xids_est) for id in xids[:not_system]]
    xindices = Dict(:dynamic_mech => xi_dynamic_mech, :noise => xi_noise, :observable => xi_observable,
                    :nondynamic => xi_nondynamic, :not_system => xi_not_system)
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
    return xindices
end

function _get_xindices_dynamic(xids::Dict{Symbol, Vector{Symbol}}, nn)::Dict{Symbol, Vector{Int32}}
    xindices = Dict{Symbol, Vector{Int32}}()
    # Get indices for mechanstic parameters in xdynamic
    xindices[:xdynamic_to_mech] = Int32.(1:length(xids[:dynamic_mech]))
    # Get indicies in xest for all dynamic parameters (mechanistic + neural net)
    xi_xest_to_xdynamic = Int32[findfirst(x -> x == id, xids[:estimate]) for id in xids[:dynamic_mech]]
    istart = length(xids[:estimate]) - length(xids[:nn])
    for pid in xids[:nn_in_ode]
        np = _get_n_net_parameters(nn, [pid])
        xi_xest_to_xdynamic = vcat(xi_xest_to_xdynamic, (istart+1):(istart + np))
        istart += np
    end
    # Get indices in xdynamic for neural-net output parameters (these are included in
    # xdynamic for gradient to be computed when split_over_conditions = false, otherwise
    # they are just kept as constant)
    for pid in xids[:nn_pre_ode]
        np = _get_n_net_parameters(nn, [pid])
        xi_xest_to_xdynamic = vcat(xi_xest_to_xdynamic, (istart+1):(istart + np))
        istart += np
    end
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
    xindices[:xdynamic_to_nnout] = (np+1):(np + length(xids[:nn_pre_ode_output]))
    return xindices
end

function _get_xindices_notsys(xids::Dict{Symbol, Vector{Symbol}})::Dict{Symbol,
                                                                        Vector{Int32}}
    ins = xids[:not_system]
    ixnoise = Int32[findfirst(x -> x == id, ins) for id in xids[:noise]]
    ixobservable = Int32[findfirst(x -> x == id, ins) for id in xids[:observable]]
    ixnondynamic = Int32[findfirst(x -> x == id, ins) for id in xids[:nondynamic]]
    return Dict(:noise => ixnoise, :observable => ixobservable,
                :nondynamic => ixnondynamic)
end

function _get_xscales(xids::Dict{T, Vector{T}},
                      petab_parameters::PEtabParameters)::Dict{T, T} where {T <: Symbol}
    @unpack parameter_scale, parameter_id = petab_parameters
    s = [parameter_scale[findfirst(x -> x == id, parameter_id)] for id in xids[:estimate]]
    return Dict(xids[:estimate] .=> s)
end

function _get_xids_dynamic_mech(xids_observable::T, xids_noise::T, xids_nondynamic::T, xids_nn::T,
                           petab_parameters::PEtabParameters)::T where {T <: Vector{Symbol}}
    dynamics_xids = Symbol[]
    for id in petab_parameters.parameter_id
        if _estimate_parameter(id, petab_parameters) == false
            continue
        end
        if id in Iterators.flatten((xids_observable, xids_noise, xids_nondynamic, xids_nn))
            continue
        end
        push!(dynamics_xids, id)
    end
    return dynamics_xids
end

function _get_xids_nn_in_ode(xids_nn::Vector{Symbol}, sys)::Vector{Symbol}
    if !(sys isa ODEProblem)
        return Symbol[]
    end
    xids_nn_in_ode = Symbol[]
    for id in xids_nn
        !haskey(sys.p, id) && continue
        push!(xids_nn_in_ode, id)
    end
    return xids_nn_in_ode
end

function _get_xids_nn_pre_ode(mapping_table, sys)::Vector{Symbol}
    !(sys isa ODEProblem) && return Symbol[]
    isempty(mapping_table) && return Symbol[]
    netids_mapping = mapping_table[!, :netId] |> unique
    return [Symbol("p_" * string(id)) for id in netids_mapping]
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

function _get_xids_nondynamic(xids_observable::T, xids_noise::T, xids_nn_pre_ode::T, sys,
                              petab_parameters::PEtabParameters,
                              conditions_df::DataFrame)::T where {T <: Vector{Symbol}}
    xids_condition = _get_xids_condition(sys, petab_parameters, conditions_df)
    if sys isa ODEProblem
        xids_sys = keys(sys.p) |> collect
    else
        xids_sys = parameters(sys) .|> Symbol
    end
    xids_nondynamic = Symbol[]
    for id in petab_parameters.parameter_id
        _estimate_parameter(id, petab_parameters) == false && continue
        _ids = Iterators.flatten((xids_sys, xids_condition, xids_observable, xids_noise, xids_nn_pre_ode))
        id in _ids && continue
        push!(xids_nondynamic, id)
    end
    return xids_nondynamic
end

function _get_xids_condition(sys, petab_parameters::PEtabParameters,
                             conditions_df::DataFrame)::Vector{Symbol}
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
            if _estimate_parameter(condition_variable, petab_parameters) == false
                continue
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
    dynamic_to_sys, sys_to_dynamic, sys_to_dynamic_nn = Int32[], Int32[], Int32[]
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
    return MapODEProblem(sys_to_dynamic, dynamic_to_sys, sys_to_dynamic_nn)
end

function _get_condition_maps(sys, parametermap, speciemap,
                             petab_parameters::PEtabParameters,
                             conditions_df::DataFrame,
                             xids::Dict{Symbol, Vector{Symbol}})::Dict{Symbol, ConditionMap}
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

function _get_sys_parameters(sys::ModelSystem, speciemap, parametermap)::Vector{Symbol}
    if sys isa ODEProblem
        return keys(sys.p) |> collect
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
    return out .|> Symbol
end

function _xdynamic_in_event_cond(model_SBML::SBMLImporter.ModelSBML,
                                 xindices::ParameterIndices,
                                 petab_tables::Dict{Symbol, DataFrame})::Bool
    xids_sys_in_xdynamic = _get_xids_sys_in_xdynamic(xindices, petab_tables[:conditions])
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

function _get_xids_sys_in_xdynamic(xindices::ParameterIndices,
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

function _check_mapping_table(mapping_table::Union{DataFrame, Nothing}, nn::Union{Nothing, Dict}, petab_parameters::PEtabParameters, sys)::DataFrame
    isnothing(mapping_table) || isnothing(nn) && return DataFrame()
    for i in 1:nrow(mapping_table)
        netid = Symbol(mapping_table[i, :netId])
        if !haskey(nn, netid)
            throw(PEtabInputError("Neural network id $netid provided in the mapping table \
                                   does not correspond to any Neural Network id provided \
                                   in the PEtab problem"))
        end
        io_id = string(mapping_table[i, :ioId])
        pattern = r"^(input\d|output\d)$"
        if !occursin(pattern, io_id)
            throw(PEtabInputError("In mapping table, in ioId column allowed values are \
                                   only input{:digit} or output{:digit} where digit is \
                                   the number of the input/output to the network. Not \
                                   $io_id"))
        end
        # TODO: Add more stringent here (e.g. not allowed to estimate parameter)
        io_value = Symbol(mapping_table[i, :ioValue])
        if occursin("input", io_id) && !(io_value in petab_parameters.parameter_id)
            throw(PEtabInputError("Input to neural net must be a PEtab parameter"))
        elseif occursin("output", io_id) && !haskey(sys.p, io_value)
            throw(PEtabInputError("Output to neural net must be an ODEModel parameter"))
        end
    end
    return DataFrame(netId = Symbol.(mapping_table[!, :netId]),
                     ioId = Symbol.(mapping_table[!, :ioId]),
                     ioValue = Symbol.(mapping_table[!, :ioValue]))
end

function _get_nn_pre_ode_maps(conditions_df::DataFrame, map_odeproblem::MapODEProblem,
                              condition_maps::Dict{Symbol, ConditionMap},
                              xids::Dict{Symbol, Vector{Symbol}}, petab_parameters::PEtabParameters,
                              xindices_dynamic, mapping_table::DataFrame, nn)::Dict{Symbol, Dict{Symbol, NNPreODEMap}}
    nconditions = nrow(conditions_df)
    maps = Dict{Symbol, Dict{Symbol, NNPreODEMap}}()
    isempty(mapping_table) && return maps
    for i in 1:nconditions
        conditionid = conditions_df[i, :conditionId] |> Symbol
        xmap_simid = condition_maps[conditionid]
        nxdynamic = vcat(map_odeproblem.dynamic_to_sys, xmap_simid.ix_dynamic,
                         xindices_dynamic[:xest_to_xdynamic]) |>
                    unique |>
                    length
        maps_nn = Dict{Symbol, NNPreODEMap}()
        for pnnid in xids[:nn_pre_ode]
            nnid = string(pnnid)[3:end] |> Symbol
            dfnn = mapping_table[mapping_table[!, :netId] .== nnid, :]
            ninputs = sum(occursin.("input", string.(dfnn[!, :ioId])))
            noutputs = sum(occursin.("output", string.(dfnn[!, :ioId])))
            np = _get_n_net_parameters(nn, [pnnid])
            inputs = zeros(Float64, ninputs)
            outputs = DiffCache(zeros(Float64, noutputs), levels = 2)
            jac_nn = zeros(Float64, noutputs, np)
            grad_output = zeros(Float64, noutputs)
            # Index for the output parameter in xdynamic
            xindices_output_xdynamic = zeros(Int32, noutputs)
            # Index for the output in sys
            xindices_output_sys = zeros(Int32, noutputs)
            for (i, io_id) in pairs(string.(dfnn[!, :ioId]))
                if occursin("input", io_id)
                    iinput = parse(Int64, io_id[6:end])
                    ip = findfirst(x -> x == dfnn[i, :ioValue], petab_parameters.parameter_id)
                    inputs[iinput] = petab_parameters.nominal_value[ip]
                end
                if occursin("output", io_id)
                    ioutput = parse(Int64, io_id[7:end])
                    ip = findfirst(x -> x == dfnn[i, :ioValue], xids[:nn_pre_ode_output])
                    xindices_output_xdynamic[ioutput] = ip + nxdynamic
                    io_value = Symbol(dfnn[i, :ioValue])
                    # TODO: Funciton to query sys
                    isys = 1
                    for id_sys in xids[:sys]
                        if id_sys in xids[:nn]
                            np = _get_n_net_parameters(nn, [id_sys]) - 1
                            isys += np
                        end
                        if id_sys == io_value
                            xindices_output_sys[ioutput] = isys
                            break
                        end
                        isys += 1
                    end
                end
            end
            maps_nn[pnnid] = NNPreODEMap(inputs, outputs, xindices_output_xdynamic, jac_nn, grad_output, xindices_output_sys)
        end
        maps[conditionid] = maps_nn
    end
    return maps
end
