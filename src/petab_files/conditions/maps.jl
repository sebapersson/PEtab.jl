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

function _get_odeproblem_map(xids::Dict{Symbol, Vector{Symbol}}, ml_models::MLModels)::MapODEProblem
    dynamic_to_sys, sys_to_dynamic = Int32[], Int32[]
    sys_to_dynamic_nn, sys_to_nn_preode_output = Int32[], Int32[]
    for (i, id_xdynmaic) in pairs(xids[:dynamic_mech])
        isys = 1
        for id_sys in xids[:sys]
            if id_sys in xids[:ml_est]
                isys += _get_n_ml_model_parameters(nn, [id_sys])
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
    for id_nn_output in xids[:ml_preode_outputs]
        isys = 1
        for id_sys in xids[:sys]
            if id_sys in xids[:ml_est]
                isys += _get_n_ml_model_parameters(nn, [id_sys])
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
        if id_sys in xids[:ml_est]
            sys_to_dynamic_nn = vcat(sys_to_dynamic_nn, _get_xindices_net(id_sys, isys, ml_models))
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

function _get_condition_maps(sys::ModelSystem, parametermap, speciemap, petab_parameters::PEtabParameters, petab_tables::PEtabTables, xids::Dict{Symbol, Vector{Symbol}})::Dict{Symbol, ConditionMap}
    mappings_df = petab_tables[:mapping]
    conditions_df = petab_tables[:conditions]
    # TODO: xids_model should just a be functions
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

            # When the value in the condition table maps to a numeric value
            if value isa Real && variable in xids_model
                push!(constant_values, value |> Float64)
                _add_ix_sys!(isys_constant_values, variable, xids_sys)
                continue
            end

            if value isa Real && Symbol(variable) in mappings_df[!, "petabEntityId"]
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

            # When the value in the conditions table maps to a parameter to estimate
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

            # NaN values are relevant for pre-equilibration and denotes a control variable
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
            if !isempty(mappings_df) && variable in mappings_df.petabEntityId
                continue
            end

            throw(PEtabFileError("For condition $conditionid, the value of $variable, \
                $value, does not correspond to any model parameter, species, or PEtab \
                parameter. The condition variable must be a valid model id or a numeric \
                 value"))
        end
        maps[conditionid] = ConditionMap(constant_values, isys_constant_values, ix_dynamic,
                                         ix_sys)
    end
    return maps
end

function _get_nn_preode_maps(xids::Dict{Symbol, Vector{Symbol}}, petab_parameters::PEtabParameters, petab_tables::PEtabTables, paths, ml_models::MLModels, sys::ModelSystem)::Dict{Symbol, Dict{Symbol, MLModelPreODEMap}}
    maps = Dict{Symbol, Dict{Symbol, MLModelPreODEMap}}()
    isempty(xids[:ml_preode]) && return maps

    mappings_df = petab_tables[:mapping]
    conditions_df = petab_tables[:conditions]
    hybridization_df = petab_tables[:hybridization]
    nconditions = nrow(conditions_df)
    for i in 1:nconditions
        conditionid = conditions_df[i, :conditionId] |> Symbol
        maps_nn = Dict{Symbol, MLModelPreODEMap}()
        for ml_model_id in xids[:ml_preode]
            # Get values for the inputs. For the pre-ODE inputs can either be constant
            # values (numeric) or a parameter, which is treated as a xdynamic parameter
            file_input = false
            inputs = _get_net_petab_variables(mappings_df, ml_model_id, :inputs) .|> Symbol
            input_values = _get_net_input_values(inputs, ml_model_id, ml_models[ml_model_id], DataFrame(conditions_df[i, :]), petab_tables, paths, petab_parameters, sys; keep_numbers = true)
            ninputs = length(input_values)
            constant_inputs, iconstant_inputs = zeros(Float64, 0), zeros(Int32, 0)
            ixdynamic_mech_inputs, ixdynamic_inputs = zeros(Int32, 0), zeros(Int32, 0)
            for (i, input_variable) in pairs(input_values)
                if isfile(string(input_variable))
                    if length(input_values) > 1
                        throw(PEtabInputError("If input to neural net is a file, only one \
                            input can be provided in the mapping table. This does not \
                            hold for $ml_model_id"))
                    end
                    input_data = _get_input_file_values(inputs[i], input_variable, conditionid)
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

                # At this point the value must be a parameter, that is either constant
                # or estimates
                ip = findfirst(x -> x == input_variable, petab_parameters.parameter_id)
                if petab_parameters.estimate[ip] == false
                    push!(constant_inputs, petab_parameters.nominal_value[ip])
                    push!(iconstant_inputs, i)
                    continue
                end
                ixmech = findfirst(x -> x == input_variable, xids[:dynamic_mech])
                push!(ixdynamic_mech_inputs, ixmech)
                push!(ixdynamic_inputs, i)
            end
            nxdynamic_inputs = length(ixdynamic_mech_inputs)

            # Indicies for correctly mapping the output. The outputs are stored in a
            # separate vector of order xids[:ml_preode_outputs], which ix_nn_outputs
            # stores the index for. The outputs maps to parameters in sys, which
            # ix_output_sys stores. Lastly, for split_over_conditions = true the
            # gradient of the output variables is needed, ix_outputs_grad stores
            # the indices in xdynamic_grad for the outputs (the indices depend on the
            # condition so the Vector is only pre-allocated here).
            output_variables = _get_net_petab_variables(mappings_df, ml_model_id, :outputs)
            outputs_df = filter(row -> row.targetValue in output_variables, hybridization_df)
            output_targets = Symbol.(outputs_df.targetId)
            noutputs = length(output_targets)
            ix_nn_outputs = zeros(Int32, noutputs)
            ix_output_sys = zeros(Int32, noutputs)
            ix_outputs_grad = zeros(Int32, noutputs)
            for (i, output_target) in pairs(output_targets)
                io = findfirst(x -> x == output_target, xids[:ml_preode_outputs])
                ix_nn_outputs[i] = io
                isys = 1
                for id_sys in xids[:sys]
                    if id_sys in xids[:ml_est]
                        isys += (_get_n_ml_model_parameters(nn, [id_sys]) - 1)
                    end
                    if id_sys == output_target
                        ix_output_sys[i] = isys
                        break
                    end
                    isys += 1
                end
            end
            maps_nn[ml_model_id] = MLModelPreODEMap(constant_inputs, iconstant_inputs, ixdynamic_mech_inputs, ixdynamic_inputs, ninputs, nxdynamic_inputs, noutputs, ix_nn_outputs, ix_outputs_grad, ix_output_sys, file_input)
        end
        maps[conditionid] = maps_nn
    end
    return maps
end

function _get_input_file_values(input_id::Symbol, file_path::Symbol, conditionid::Symbol)::Array{Float64}
    input_file = HDF5.h5open(string(file_path), "r")
    # Find the correct dataset associated with the provided simulation condition
    for i in keys(input_file["inputs"]["$(input_id)"])
        group_i = input_file["inputs"]["$(input_id)"][i]

        if !haskey(group_i, "conditionIds")
            input_values = HDF5.read_dataset(group_i, "data")
            close(input_file)
            return input_values
        end

        conditionids = HDF5.read_dataset(group_i, "conditionIds")
        !(string(conditionid) in conditionids) && continue
        input_values = HDF5.read_dataset(group_i, "data")
        close(input_file)
        return input_values
    end
    throw(PEtabInputError("The file $(file_path) which contains initial values for \
        neural network input ID $(input_id) does not provide any data for condition \
        $(condition_id)"))
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
