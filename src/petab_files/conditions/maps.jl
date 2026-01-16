function _get_map_observable_noise(
        xids::Vector{Symbol}, petab_measurements::PEtabMeasurements,
        petab_parameters::PEtabParameters; observable::Bool
    )::Vector{ObservableNoiseMap}
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
            throw(PEtabFileError("Id $value in noise or observable column in measurement \
                file does not correspond to any id in the parameters table"))
        end
        single_constant = nvalues_row == 1 && estimate[1] == false
        maps[i] = ObservableNoiseMap(estimate, xindices, constant_values, nvalues_row,
                                     single_constant)
    end
    return maps
end

function _get_condition_maps(
        sys::ModelSystem, parametermap, speciemap, petab_parameters::PEtabParameters,
        petab_tables::PEtabTables, xids::Dict{Symbol, Vector{Symbol}}, ml_models::MLModels
    )::Dict{Symbol, ConditionMap}
    conditions_df, mappings_df = _get_petab_tables(petab_tables, [:conditions, :mapping])

    xids_sys = string.(xids[:sys])
    xids_dynamic_mech = string.(xids[:est_to_dynamic_mech])
    state_ids = _get_state_ids(sys)
    model_ids = Iterators.flatten((xids_sys, state_ids))

    ml_inputs = String[]
    for (ml_id, ml_model) in ml_models
        ml_model.static == false && continue
        _ml_inputs = _get_ml_model_io_petab_ids(mappings_df, ml_id, :inputs) |>
            Iterators.flatten .|>
            string
        ml_inputs = vcat(ml_inputs, _ml_inputs)
    end

    ix_all_conditions, isys_all_conditions = _get_all_conditions_map(xids)
    condition_maps = Dict{Symbol, ConditionMap}()
    for (i, conditionid) in pairs(Symbol.(conditions_df[!, :conditionId]))
        target_ids = String[]
        target_value_formulas = String[]
        isys_condition = Int32[]
        for target_id in names(conditions_df)
            target_id in ["conditionName", "conditionId"] && continue
            target_id in ml_inputs && continue

            target_value = conditions_df[i, target_id]
            push!(target_ids, target_id)

            # If target_value is missing the default model value should be used, which
            # are encoded in the parameter- and specie-maps
            # in the parameter- and speciemaps
            if ismissing(target_value) && target_id in model_ids
                default_value = _get_default_map_value(target_id, parametermap, speciemap)
                push!(target_value_formulas, string(default_value))
                _add_ix_sys!(isys_condition, target_id, xids_sys)
                continue
            end

            # NaN values apply for pre-equilibration and implies a specie should use the
            # value obtained from the forward simulation
            is_nan = (target_value isa Real && isnan(target_value)) || target_value == "NaN"
            if is_nan && target_id in state_ids
                push!(target_value_formulas, "NaN")
                _add_ix_sys!(isys_condition, target_id, xids_sys)
                continue
            elseif is_nan && target_id in model_ids
                throw(PEtabFileError("If a row in conditions file is NaN then the column \
                                      header must be a specie"))
            end

            # At this point a valid PEtab formula is assumed
            push!(target_value_formulas, string(target_value))
            _add_ix_sys!(isys_condition, target_id, xids_sys)
        end

        # In the formulas replace any xdynamic parameters. Tracking which parameters
        # occur is important for runtime performance with split gradient methods.
        ix_condition = Int32[]
        for i in eachindex(target_value_formulas)
            for (j, xid) in pairs(xids_dynamic_mech)
                _formula = SBMLImporter._replace_variable(target_value_formulas[i], xid, "xdynamic[$(j)]")
                target_value_formulas[i] == _formula && continue
                push!(ix_condition, j)
                target_value_formulas[i] = _formula
            end
            # Hard-code potential parameters which are not estimated
            for j in eachindex(petab_parameters.estimate)
                petab_parameters.estimate[j] == true && continue
                xid = string(petab_parameters.parameter_id[j])
                value = petab_parameters.nominal_value[j]
                _formula = SBMLImporter._replace_variable(target_value_formulas[i], xid, "$(value)")
                target_value_formulas[i] = _formula
            end
        end

        target_value_functions = Vector{Function}(undef, length(target_value_formulas))
        for i in eachindex(target_value_functions)
            formula = _template_target_value(target_value_formulas[i], conditionid, target_ids[i])
            target_value_functions[i] = @RuntimeGeneratedFunction(Meta.parse(formula))
        end

        condition_maps[conditionid] = ConditionMap(isys_condition, ix_condition, target_value_functions, ix_all_conditions, isys_all_conditions, zeros(0, 0))
    end
    return condition_maps
end

function _get_all_conditions_map(xids::Dict{Symbol, Vector{Symbol}})::NTuple{2, Vector{Int32}}
    isys_all_conditions = findall(x -> x in xids[:sys], xids[:est_to_dynamic_mech]) |> Vector{Int32}
    ids = xids[:est_to_dynamic_mech][isys_all_conditions]
    ix_all_conditions = Int32[findfirst(x -> x == id, xids[:sys]) for id in ids]
    return ix_all_conditions, isys_all_conditions
end

function _get_nn_pre_simulate_maps(xids::Dict{Symbol, Vector{Symbol}}, petab_parameters::PEtabParameters, petab_tables::PEtabTables, paths, ml_models::MLModels, sys::ModelSystem)::Dict{Symbol, Dict{Symbol, MLModelPreODEMap}}
    maps = Dict{Symbol, Dict{Symbol, MLModelPreODEMap}}()
    isempty(xids[:ml_pre_simulate]) && return maps

    mappings_df = petab_tables[:mapping]
    conditions_df = petab_tables[:conditions]
    hybridization_df = petab_tables[:hybridization]
    nconditions = nrow(conditions_df)
    for i in 1:nconditions
        conditionid = conditions_df[i, :conditionId] |> Symbol
        maps_nn = Dict{Symbol, MLModelPreODEMap}()
        for ml_id in xids[:ml_pre_simulate]
            input_info = _get_ml_pre_simulate_inputs(ml_id, conditionid, xids, petab_parameters, petab_tables, paths, ml_models, sys)

            # Indices for correctly mapping the output. The outputs are stored in a
            # separate vector of order xids[:sys_ml_pre_simulate_outputs], which ix_nn_outputs
            # stores the index for. The outputs maps to parameters in sys, which
            # ix_output_sys stores. Lastly, for split_over_conditions = true the
            # gradient of the output variables is needed, ix_outputs_grad stores
            # the indices in xdynamic_grad for the outputs (the indices depend on the
            # condition so the Vector is only pre-allocated here).
            output_variables = _get_ml_model_io_petab_ids(mappings_df, ml_id, :outputs) |>
                Iterators.flatten
            outputs_df = filter(row -> row.targetValue in output_variables, hybridization_df)
            output_targets = Symbol.(outputs_df.targetId)
            noutputs = length(output_targets)
            ix_nn_outputs = zeros(Int32, noutputs)
            ix_output_sys = zeros(Int32, noutputs)
            ix_outputs_grad = zeros(Int32, noutputs)
            for (i, output_target) in pairs(output_targets)
                io = findfirst(x -> x == output_target, xids[:sys_ml_pre_simulate_outputs])
                ix_nn_outputs[i] = io
                isys = 1
                for id_sys in xids[:sys]
                    if id_sys in xids[:ml_est]
                        isys += (_get_n_ml_parameters(nn, [id_sys]) - 1)
                    end
                    if id_sys == output_target
                        ix_output_sys[i] = isys
                        break
                    end
                    isys += 1
                end
            end
            n_input_arguments = length(_get_ml_model_io_petab_ids(mappings_df, ml_id, :inputs))
            maps_nn[ml_id] = MLModelPreODEMap(n_input_arguments, input_info[:constant_inputs], input_info[:iconstant_inputs], input_info[:ixdynamic_mech_inputs], input_info[:ixdynamic_inputs], input_info[:ninputs], input_info[:nxdynamic_inputs], noutputs, ix_nn_outputs, ix_outputs_grad, ix_output_sys, input_info[:file_input])
        end
        maps[conditionid] = maps_nn
    end
    return maps
end

function _get_ml_pre_simulate_inputs(ml_id::Symbol, conditionid::Symbol, xids::Dict{Symbol, Vector{Symbol}}, petab_parameters::PEtabParameters, petab_tables::PEtabTables, paths, ml_models::MLModels, sys::ModelSystem)::Dict
    mappings_df = petab_tables[:mapping]
    conditions_df = petab_tables[:conditions]
    input_arguments = _get_ml_model_io_petab_ids(mappings_df, ml_id, :inputs)
    n_input_arguments = length(input_arguments)

    out = Dict()
    out[:constant_inputs] = Vector{Array{Float64}}(undef, n_input_arguments)
    out[:iconstant_inputs] = Vector{Vector{Int32}}(undef, n_input_arguments)
    out[:ixdynamic_mech_inputs] = Vector{Vector{Int32}}(undef, n_input_arguments)
    out[:ixdynamic_inputs] = Vector{Vector{Int32}}(undef, n_input_arguments)
    out[:ninputs] = zeros(Int64, n_input_arguments)
    out[:nxdynamic_inputs] = zeros(Int64, n_input_arguments)
    out[:file_input] = zeros(Bool, n_input_arguments)
    for (i, _input_argument) in pairs(input_arguments)
        input_argument = Symbol.(_input_argument)
        i_condition = findfirst(x -> x == string(conditionid), conditions_df.conditionId)
        input_values = _get_ml_model_input_values(input_argument, ml_id, ml_models[ml_id], DataFrame(conditions_df[i_condition, :]), petab_tables, paths, petab_parameters, sys; keep_numbers = true)

        out[:constant_inputs][i] = zeros(Float64, 0)
        out[:iconstant_inputs][i] = zeros(Int32, 0)
        out[:ixdynamic_mech_inputs][i] = zeros(Int32, 0)
        out[:ixdynamic_inputs][i] = zeros(Int32, 0)
        for (j, input_variable) in pairs(input_values)
            if isfile(string(input_variable))
                if length(input_values) > 1
                    throw(PEtabInputError("If input to neural net is a file, only one \
                        input can be provided in the mapping table. This does not \
                        hold for $ml_id"))
                end
                input_data = _get_input_file_values(input_argument[j], input_variable, conditionid)
                # hdf5 files are in row-major
                input_data = permutedims(input_data, reverse(1:ndims(input_data)))
                _constant_inputs = _reshape_io_data(input_data)
                # Given an input file, we need to have a batch
                out[:constant_inputs][i] = reshape(_constant_inputs, (size(_constant_inputs)..., 1))
                out[:file_input][i] = true
                continue
            end

            if is_number(input_variable)
                val = parse(Float64, string(input_variable))
                push!(out[:constant_inputs][i], val)
                push!(out[:iconstant_inputs][i], j)
                continue
            end

            # At this point the value must be a parameter, that is either constant
            # or estimated
            ip = findfirst(x -> x == input_variable, petab_parameters.parameter_id)
            if petab_parameters.estimate[ip] == false
                push!(out[:constant_inputs][i], petab_parameters.nominal_value[ip])
                push!(out[:iconstant_inputs][i], j)
                continue
            end
            ixmech = findfirst(x -> x == input_variable, xids[:est_to_dynamic_mech])
            push!(out[:ixdynamic_mech_inputs][i], ixmech)
            push!(out[:ixdynamic_inputs][i], j)
        end
        out[:nxdynamic_inputs][i] = length(out[:ixdynamic_mech_inputs][i])
        out[:ninputs][i] = length(input_values)
    end
    return out
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

function _template_target_value(formula::String, condition_id::Symbol, target_id::String)::String
    out = "function _map_$(target_id)_$(condition_id)(xdynamic)\n"
    out *= "\treturn $(formula)\nend"
    return out
end
