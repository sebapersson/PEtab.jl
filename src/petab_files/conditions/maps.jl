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

    isys_all_conditions = findall(x -> x in xids[:sys], xids[:est_to_dynamic_mech]) |>
        Vector{Int32}
    ids = xids[:est_to_dynamic_mech][isys_all_conditions]
    ix_all_conditions = Int32[findfirst(x -> x == id, xids[:sys]) for id in ids]

    condition_maps = Dict{Symbol, ConditionMap}()
    for (i, condition_id) in pairs(Symbol.(conditions_df[!, :conditionId]))
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
            formula = _template_target_value(target_value_formulas[i], condition_id, target_ids[i])
            target_value_functions[i] = @RuntimeGeneratedFunction(Meta.parse(formula))
        end

        condition_maps[condition_id] = ConditionMap(isys_condition, ix_condition, target_value_functions, ix_all_conditions, isys_all_conditions, zeros(0, 0))
    end
    return condition_maps
end

function _get_ml_pre_simulate_maps(xids::Dict{Symbol, Vector{Symbol}}, petab_parameters::PEtabParameters, petab_tables::PEtabTables, paths, ml_models::MLModels, sys::ModelSystem)::Dict{Symbol, Dict{Symbol, MLModelPreSimulateMap}}
    maps = Dict{Symbol, Dict{Symbol, MLModelPreSimulateMap}}()
    isempty(xids[:ml_pre_simulate]) && return maps

    mappings_df, conditions_df, hybridization_df = _get_petab_tables(
        petab_tables, [:mapping, :conditions, :hybridization]
    )

    for i in 1:nrow(conditions_df)
        condition_id = Symbol(conditions_df[i, :conditionId])

        maps_ml = Dict{Symbol, MLModelPreSimulateMap}()
        for ml_id in xids[:ml_pre_simulate]

            _f_input, constant_inputs, i_dynamic_mech = _get_ml_pre_simulate_inputs(
                ml_id, condition_id, xids, petab_parameters, petab_tables, paths,
                ml_models, sys
            )
            f_input = @RuntimeGeneratedFunction(Meta.parse(_f_input))
            n_input_args = length(constant_inputs)

            # Indices for correctly mapping the output. The outputs are stored in a
            # separate vector of order xids[:sys_ml_pre_simulate_outputs]. The outputs
            # maps to parameters in sys, which ix_output_sys stores. Lastly, for
            # split_over_conditions = true the gradient of the output variables is needed,
            # ix_outputs_grad stores the indices in xdynamic_grad for the outputs (the
            # indices depend on the condition so the Vector is only pre-allocated here).
            output_ids = Iterators.flatten(
                _get_ml_model_io_petab_ids(mappings_df, ml_id, :outputs)
            )
            outputs_df = filter(r -> r.targetValue in output_ids, hybridization_df)
            output_targets = Symbol.(outputs_df.targetId)
            n_outputs = length(output_targets)

            ix_ml_outputs = zeros(Int32, n_outputs)
            ix_sys_outputs = zeros(Int32, n_outputs)
            for (i, output_target) in pairs(output_targets)
                ix_ml_outputs[i] = findfirst(
                    x -> x == output_target, xids[:sys_ml_pre_simulate_outputs]
                )

                isys = 1
                for id_sys in xids[:sys]
                    if id_sys in xids[:ml_est]
                        isys += (_get_n_ml_parameters(nn, [id_sys]) - 1)
                    end
                    if id_sys == output_target
                        ix_sys_outputs[i] = isys
                        break
                    end
                    isys += 1
                end
            end

            maps_ml[ml_id] = MLModelPreSimulateMap(
                n_input_args, f_input, constant_inputs, i_dynamic_mech,  n_outputs,
                ix_ml_outputs, ix_sys_outputs
            )
        end
        maps[condition_id] = maps_ml
    end
    return maps
end

function _get_ml_pre_simulate_inputs(
        ml_id::Symbol, condition_id::Symbol, xids::Dict{Symbol, Vector{Symbol}},
        petab_parameters::PEtabParameters, petab_tables::PEtabTables, paths,
        ml_models::MLModels, sys::ModelSystem
    )
    mappings_df, conditions_df, experiments_df = _get_petab_tables(
        petab_tables, [:mapping, :conditions, :experiments]
    )

    # Arrays used to track input mappings
    input_ids = _get_ml_model_io_petab_ids(mappings_df, ml_id, :inputs)
    n_input_args = length(input_ids)
    constant_inputs = Vector{Array{Float64}}(undef, n_input_args)
    input_formulas = Vector{Vector{String}}(undef, n_input_args)
    file_input = fill(false, n_input_args)
    i_dynamic_mech = Int32[]

    for (i, input_id) in pairs(input_ids)
        condition_row = filter(r -> r.conditionId == string(condition_id), conditions_df) |>
            DataFrame
        input_values = _get_ml_model_input_values(
            Symbol.(input_id), ml_id, ml_models[ml_id], condition_row, petab_tables, paths,
            petab_parameters, sys; keep_numbers = true
        )

        input_formulas[i] = fill("", length(input_values))
        constant_inputs[i] = zeros(Float64, 0)
        for (j, input_value) in pairs(input_values)
            # If an argument has a file-input, it is the only allowed input for said arg
            if isfile(string(input_value))
                if length(input_values) > 1
                    throw(PEtabInputError("If input to neural net is a file, only one \
                        input can be provided in the mapping table. This does not \
                        hold for $ml_id"))
                end

                # hdf5 files are in row-major, requiring re-shaping
                v2_condition_id = _get_petab_v2_condition_id(condition_id, experiments_df)
                input_data = _get_input_file_values(input_id[1], input_value, v2_condition_id)
                input_data = permutedims(input_data, reverse(1:ndims(input_data)))
                input_data = _reshape_io_data(input_data)
                input_data = reshape(input_data, (size(input_data)..., 1))
                constant_inputs[i] = input_data
                input_formulas[i][j] = "map_pre_simulate.constant_inputs[$(i)]"
                file_input[i] = true
                break
            end

            # In case an array value is provided to apply for all simulation conditions
            if input_ids[i][j] == "_ARRAY_INPUT"
                constant_inputs[i] = ml_models[ml_id].array_inputs[Symbol("__arg$(i)")]
                input_formulas[i][j] = "map_pre_simulate.constant_inputs[$(i)]"
                file_input[i] = true
                break
            end

            # Condition specific array input
            if input_value == :_ARRAY_INPUT
                constant_inputs[i] = ml_models[ml_id].array_inputs[Symbol("$(condition_id)_$(i)")]
                input_formulas[i][j] = "map_pre_simulate.constant_inputs[$(i)]"
                file_input[i] = true
                break
            end

            # At this point, a valid PEtab-formula is assumed, for which constant values
            # will be inserted
            input_formula = string(input_value)
            for (k, parameter_id) in pairs(string.(petab_parameters.parameter_id))
                petab_parameters.estimate[k] == true && continue
                value = petab_parameters.nominal_value[k]
                input_formula = SBMLImporter._replace_variable(
                    input_formula, parameter_id, "$(value)"
                )
            end
            # Potential parameters from x_dynamic
            for (k, id) in pairs(string.(xids[:est_to_dynamic_mech]))
                ix = length(i_dynamic_mech) + 1
                _formula = SBMLImporter._replace_variable(input_formula, id, "xdynamic[$ix]")
                input_formula == _formula && continue
                push!(i_dynamic_mech, k)
                input_formula = _formula
            end

            input_formulas[i][j] = input_formula
        end
    end

    unique!(i_dynamic_mech)
    f_input = _template_ml_input(input_formulas, file_input, condition_id, ml_id, i_dynamic_mech)
    return f_input, constant_inputs, i_dynamic_mech
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

function _template_ml_input(input_formulas, file_input::Vector{Bool}, condition_id, ml_id, i_dynamic_mech::Vector{Int32})
    out = "function _map_input_$(condition_id)_$(ml_id)(xdynamic, map_pre_simulate)\n"
    for i in eachindex(input_formulas)
        if file_input[i] == true
            out *= "\tout_$(i) = $(input_formulas[i][1])\n"
            continue
        end

        if isempty(i_dynamic_mech)
            out *= "\tout_$(i) = zeros(Float64, $(length(input_formulas[i])))\n"
        else
            out *= "\tout_$(i) = zeros(eltype(xdynamic), $(length(input_formulas[i])))\n"
        end

        for (j, formula) in pairs(input_formulas[i])
            out *= "\tout_$(i)[$(j)] = $(formula)\n"
        end
    end

    if length(input_formulas) == 1
        out *= "\treturn out_1\n"
    else
        out_args = prod("out_" .* string.(1:length(input_formulas)) .* ", ")
        out *= "\treturn ($(out_args))\n"
    end
    out *= "end\n"
    return out
end

function _get_petab_v2_condition_id(condition_id::Symbol, experiments_df::DataFrame)::Symbol
    isempty(experiments_df) && return condition_id

    condition_id = string(condition_id)
    for experiment_id in experiments_df.experimentId
        if startswith(condition_id, "$(experiment_id)_")
            condition_id_v2 = replace(condition_id, "$(experiment_id)_" => "")
            return Symbol(condition_id_v2)
        end
    end
    return Symbol(condition_id)
end
