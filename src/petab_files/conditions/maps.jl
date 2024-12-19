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

function _get_odeproblem_map(xids::Dict{Symbol, Vector{Symbol}}, nnmodels::Union{Dict{Symbol, <:NNModel}, Nothing})::MapODEProblem
    dynamic_to_sys, sys_to_dynamic = Int32[], Int32[]
    sys_to_dynamic_nn, sys_to_nn_preode_output = Int32[], Int32[]
    for (i, id_xdynmaic) in pairs(xids[:dynamic_mech])
        isys = 1
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
    end
    for id_nn_output in xids[:nn_preode_outputs]
        isys = 1
        for id_sys in xids[:sys]
            if id_sys in xids[:nn]
                isys += _get_n_net_parameters(nn, [id_sys])
                continue
            end
            if id_sys == id_nn_output
                push!(sys_to_nn_preode_output, isys)
                break
            end
            isys += 1
        end
    end
    isys = 0
    for id_sys in xids[:sys]
        if id_sys in xids[:nn]
            sys_to_dynamic_nn = vcat(sys_to_dynamic_nn, _get_xindices_net(id_sys, isys, nnmodels))
            isys = sys_to_dynamic_nn[end]
            continue
        end
        isys += 1
    end
    dynamic_to_sys = findall(x -> x in xids[:sys], xids[:dynamic_mech]) |> Vector{Int64}
    ids = xids[:dynamic_mech][dynamic_to_sys]
    sys_to_dynamic = Int64[findfirst(x -> x == id, xids[:sys]) for id in ids]
    return MapODEProblem(sys_to_dynamic, dynamic_to_sys, sys_to_dynamic_nn, sys_to_nn_preode_output)
end

function _get_condition_maps(sys::ModelSystem, parametermap, speciemap, petab_parameters::PEtabParameters, conditions_df::DataFrame, mapping_table::DataFrame, xids::Dict{Symbol, Vector{Symbol}})::Dict{Symbol, ConditionMap}
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

            if value isa Real && Symbol(variable) in mapping_table[!, "petabEntityId"]
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
            if !isempty(mapping_table) && Symbol(variable) in mapping_table[!, "petabEntityId"]
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

function _get_nn_preode_maps(conditions_df::DataFrame, xids::Dict{Symbol, Vector{Symbol}}, petab_parameters::PEtabParameters, mapping_table::DataFrame, nnmodels::Dict{Symbol, <:NNModel}, sys::ModelSystem)::Dict{Symbol, Dict{Symbol, NNPreODEMap}}
    nconditions = nrow(conditions_df)
    maps = Dict{Symbol, Dict{Symbol, NNPreODEMap}}()
    isempty(xids[:nn_preode]) && return maps
    for i in 1:nconditions
        conditionid = conditions_df[i, :conditionId] |> Symbol
        maps_nn = Dict{Symbol, NNPreODEMap}()
        for netid in xids[:nn_preode]
            outputs = _get_net_values(mapping_table, netid, :outputs) .|> Symbol
            inputs = _get_net_values(mapping_table, netid, :inputs) .|> Symbol
            input_variables = _get_nn_input_variables(inputs, netid, nnmodels[netid], DataFrame(conditions_df[i, :]), petab_parameters, sys; keep_numbers = true)
            ninputs = length(input_variables)
            file_input = false

            # Get values for the inputs. For the pre-ODE inputs can either be constant
            # values (numeric) or a parameter, which is treated as a xdynamic parameter
            constant_inputs, iconstant_inputs = zeros(Float64, 0), zeros(Int32, 0)
            ixdynamic_mech_inputs, ixdynamic_inputs = zeros(Int32, 0), zeros(Int32, 0)
            for (i, input_variable) in pairs(input_variables)
                if isfile(string(input_variable))
                    if length(input_variables) > 1
                        throw(PEtabInputError("If input to neural net is a file, only one \
                            input can be provided in the mapping table. This does not \
                            hold for $netid"))
                    end
                    # TODO: Add support for multiple inputs here
                    input_data = h5read(string(input_variable), "input1") .|> Float64
                    # hdf5 files are in row-major
                    input_data = permutedims(input_data, reverse(1:ndims(input_data)))
                    constant_inputs = _reshape_io_data(input_data)
                    # Given an input file, we need to have a batch
                    constant_inputs = reshape(constant_inputs, (size(constant_inputs)..., 1))
                    file_input = true
                    continue
                end
                if is_number(input_variable)
                    val = parse(Float64, string(input_variable))
                    push!(constant_inputs, val)
                    push!(iconstant_inputs, i)
                    continue
                end
                # At this point the value must be a parameter
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
            # separate vector of order xids[:nn_preode_outputs], which ix_nn_outputs
            # stores the index for. The outputs maps to parameters in sys, which
            # ix_output_sys stores. Lastly, for split_over_conditions = true the
            # gradient of the output variables is needed, ix_outputs_grad stores
            # the indices in xdynamic_grad for the outputs (the indices depend on the
            # condition so the Vector is only pre-allocated here).
            noutputs = length(outputs)
            ix_nn_outputs = zeros(Int32, noutputs)
            ix_output_sys = zeros(Int32, noutputs)
            ix_outputs_grad = zeros(Int32, noutputs)
            for (i, output_variable) in pairs(outputs)
                io = findfirst(x -> x == output_variable, xids[:nn_preode_outputs])
                ix_nn_outputs[i] = io
                isys = 1
                for id_sys in xids[:sys]
                    if id_sys in xids[:nn]
                        isys += (_get_n_net_parameters(nn, [id_sys]) - 1)
                    end
                    if id_sys == output_variable
                        ix_output_sys[i] = isys
                        break
                    end
                    isys += 1
                end
            end
            maps_nn[netid] = NNPreODEMap(constant_inputs, iconstant_inputs, ixdynamic_mech_inputs, ixdynamic_inputs, ninputs, nxdynamic_inputs, noutputs, ix_nn_outputs, ix_outputs_grad, ix_output_sys, file_input)
        end
        maps[conditionid] = maps_nn
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
