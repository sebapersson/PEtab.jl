"""
    _get_ids(
        petab_parameters, petab_ml_parameters, petab_measurements, sys, petab_tables,
        speciemap, parametermap, ml_models
    )

Categorize all models parameters.

This is a **very** important function, as parameter type dictates how the gradient should
be computed for any parameter. The assigned parameter categories are later used to
build all the indices used to correctly map parameters.
"""
function _get_ids(
        petab_parameters::PEtabParameters, petab_ml_parameters::PEtabMLParameters,
        petab_measurements::PEtabMeasurements, sys::ModelSystem, petab_tables::PEtabTables,
        paths::Dict{Symbol, String}, speciemap, parametermap, ml_models::MLModels
    )::Dict{Symbol, Vector{Symbol}}
    @unpack observable_parameters, noise_parameters = petab_measurements

    # ids in the ODESystem in correct order
    ids_sys = _get_ids_sys_order(sys, speciemap, parametermap)

    # ML parameters
    # In case of multiple neural networks, to get correct indexing for adjoint gradient
    # methods, nets that appears in the ODE RHS must appear first, and ids_sys must follow
    # the order of ids_ml_in_ode for adjoint indexing to be correct.
    ids_ml_in_ode = _get_ids_ml_in_ode(ml_models, sys)
    ids_ml_pre_simulate = _get_ids_ml_pre_simulate(ml_models)
    ids_ml_nondynamic = _get_ids_ml_nondynamic(ml_models, ids_ml_in_ode, ids_ml_pre_simulate)
    ids_ml = _get_ids_ml(ids_ml_in_ode, ids_ml_pre_simulate, ids_ml_nondynamic)

    # Parameter which are input to a neural net, and are estimated. These must be
    # tracked in order to compute correct gradients
    ids_ml_input_est = _get_ids_ml_input_est(
        petab_tables, petab_parameters, sys, paths, ml_models
    )

    # Parameters set by a static neural net. Needed to be tracked for gradient computations
    ids_ml_pre_simulate_output = _get_ids_ml_pre_simulate_output(petab_tables, ml_models)

    # Ensure sys has correct order for adjoint sensitivities
    _order_id_sys!(ids_sys, ids_ml_in_ode)

    # Mechanistic (none neural-net parameters). Note non-dynamic parameters are those that
    # only appear in the observable and noise functions, but are not defined noise or
    # observable column of the measurement file.
    ids_observable = _get_ids_observable_noise(observable_parameters, petab_parameters)
    ids_noise = _get_ids_observable_noise(noise_parameters, petab_parameters)
    ids_nondynamic_mech = _get_ids_nondynamic_mech(
        ids_observable, ids_noise, ids_ml, ids_ml_input_est, sys, petab_parameters,
        petab_tables, ml_models
    )
    ids_dynamic_mech = _get_ids_dynamic_mech(
        ids_observable, ids_noise, ids_nondynamic_mech, ids_ml, petab_parameters
    )

    # Edge case. If a parameter is in ids_ml_input_est and does not appear in the sys,
    # it must be treated as a dynamic parameter for gradient purposes since it informs
    # the ODE, so in a sense, it is a form of dynamic parameter.
    _add_ids_ml_input_est!(ids_dynamic_mech, ids_ml_input_est)
    ids_not_system_mech = unique(vcat(ids_observable, ids_noise, ids_nondynamic_mech))

    # Parameters to estimate
    ids_ml_est = _get_ids_ml_est(ids_ml, petab_ml_parameters)
    ids_estimate = vcat(ids_dynamic_mech, ids_not_system_mech, ids_ml_est)
    ids_petab = petab_parameters.parameter_id

    return Dict(
        :est_to_dynamic_mech => ids_dynamic_mech, :noise => ids_noise,
        :ml => ids_ml, :ml_est => ids_ml_est, :observable => ids_observable,
        :nondynamic_mech => ids_nondynamic_mech, :not_system_mech => ids_not_system_mech,
        :sys => ids_sys, :estimate => ids_estimate, :petab => ids_petab,
        :ml_in_ode => ids_ml_in_ode, :ml_pre_simulate => ids_ml_pre_simulate,
        :sys_ml_pre_simulate_outputs => ids_ml_pre_simulate_output,
        :ml_nondynamic => ids_ml_nondynamic
    )
end

function _get_ids_dynamic_mech(
        ids_observable::T, ids_noise::T, ids_nondynamic_mech::T, ids_ml::T,
        petab_parameters::PEtabParameters
    )::T where {T <: Vector{Symbol}}
    other_ids = Iterators.flatten((ids_observable, ids_noise, ids_nondynamic_mech, ids_ml))

    dynamics_ids = Symbol[]
    for id in petab_parameters.parameter_id
        if _estimate_parameter(id, petab_parameters) == false
            continue
        end
        if id in other_ids
            continue
        end
        push!(dynamics_ids, id)
    end
    return dynamics_ids
end

_get_ids_sys_order(sys::ODEProblem, ::Any, ::Any)::Vector{Symbol} = collect(keys(sys.p))
function _get_ids_sys_order(sys::ModelSystem, speciemap, parametermap)::Vector{Symbol}
    # This is a hack until SciMLSensitivity integrates with the SciMLStructures interface.
    # Basically allows the parameters in the system to be retrieved in the order they
    # appear in the ODESystem later on
    _p = parameters(sys)
    out = similar(_p)
    if sys isa SDESystem
        prob = SDEProblem(sys, speciemap, [0.0, 5.0e3], parametermap)
    else
        prob = ODEProblem(sys, speciemap, [0.0, 5.0e3], parametermap; jac = true)
    end
    maps = ModelingToolkit.getp(prob, _p)
    for (i, map) in pairs(maps.getters)
        out[map.idx.idx] = _p[i]
    end
    return Symbol.(out)
end

function _get_ids_sys(sys::ODEProblem)::Vector{Symbol}
    return collect(keys(sys.p))
end
function _get_ids_sys(sys::ModelSystem)::Vector{Symbol}
    return Symbol.(parameters(sys))
end

function _get_ids_ml_pre_simulate_output(
        petab_tables::PEtabTables, ml_models::MLModels
    )::Vector{Symbol}
    mappings_df, hybridization_df = _get_petab_tables(
        petab_tables, [:mapping, :hybridization]
    )

    out = Symbol[]
    isempty(hybridization_df) && return out

    for ml_model in ml_models.ml_models
        ml_model.static == false && continue

        ml_id = ml_model.ml_id
        output_variables = Iterators.flatten(
            _get_ml_model_io_petab_ids(mappings_df, ml_id, :outputs)
        )
        outputs_df = filter(row -> row.targetValue in output_variables, hybridization_df)
        out = vcat(out, Symbol.(outputs_df.targetId))
    end
    return out
end

function _get_ids_observable_noise(
        values, petab_parameters::PEtabParameters
    )::Vector{Symbol}
    ids = Symbol[]
    for value in values
        isempty(value) && continue
        is_number(value) && continue
        # Multiple ids are split by ; in the PEtab table
        for id in Symbol.(split(value, ';'))
            is_number(id) && continue
            if !(id in petab_parameters.parameter_id)
                throw(
                    PEtabFileError(
                        "Parameter $id in measurement file does not appear \
                        in the PEtab parameters table."
                    )
                )
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

function _get_ids_nondynamic_mech(
        ids_observable::T, ids_noise::T, ids_ml::T, ids_ml_input_est::T, sys::ModelSystem,
        petab_parameters::PEtabParameters, petab_tables::PEtabTables, ml_models::MLModels
    )::T where {T <: Vector{Symbol}}
    ids_condition = _get_ids_condition(sys, petab_parameters, petab_tables, ml_models)
    ids_sys = _get_ids_sys(sys)
    other_ids = Iterators.flatten(
        (ids_sys, ids_condition, ids_observable, ids_noise, ids_ml, ids_ml_input_est)
    )

    ids_nondynamic_mech = Symbol[]
    for id in petab_parameters.parameter_id
        _estimate_parameter(id, petab_parameters) == false && continue
        id in other_ids && continue
        push!(ids_nondynamic_mech, id)
    end
    return ids_nondynamic_mech
end

function _get_ids_condition(
        sys::ModelSystem, petab_parameters::PEtabParameters, petab_tables::PEtabTables,
        ml_models::MLModels
    )::Vector{Symbol}
    mappings_df, conditions_df = _get_petab_tables(petab_tables, [:mapping, :conditions])
    ids_sys = parameters(sys) .|> string
    species_sys = _get_state_ids(sys)

    ml_inputs = String[]
    for ml_model in ml_models.ml_models
        ml_model.static == false && continue

        ml_id = ml_model.ml_id
        ml_model_inputs = _get_ml_model_io_petab_ids(mappings_df, ml_id, :inputs)
        ml_inputs = vcat(ml_inputs, reduce(vcat, ml_model_inputs))
    end

    ids_condition = Symbol[]
    for id in names(conditions_df)
        id in ["conditionName", "conditionId"] && continue

        if !(id in Iterators.flatten((ids_sys, species_sys, ml_inputs)))
            throw(PEtabFileError("Parameter/state $id that dictates an experimental \
                                  condition does not appear among the model variables"))
        end

        for condition_value in conditions_df[!, id]
            ismissing(condition_value) && continue
            condition_value isa Real && continue
            is_number(condition_value) && continue

            for parameter_id in petab_parameters.parameter_id
                estimate = _estimate_parameter(parameter_id, petab_parameters)

                # Sanity check input
                for ml_input in ml_inputs
                    _formula = SBMLImporter._replace_variable(
                        condition_value, "$(ml_input)", ""
                    )
                    _formula == condition_value && continue
                    throw(PEtabFileError("ML model input variable $(condition_variable) \
                        setting value of $id in a simulation condition is not \
                        allowed to be estimated"))
                end

                estimate == false && continue
                _formula = SBMLImporter._replace_variable(
                    condition_value, "$(parameter_id)", ""
                )
                _formula == condition_value && continue
                parameter_id in ids_condition && continue
                push!(ids_condition, parameter_id)
            end
        end
    end
    return ids_condition
end

function _get_ids_ml_in_ode(ml_models::MLModels, sys::ModelSystem)::Vector{Symbol}
    !(sys isa ODEProblem) && return Symbol[]
    ids_ml_in_ode = Symbol[]
    for id in ml_models.ml_ids
        !haskey(sys.p, id) && continue
        push!(ids_ml_in_ode, id)
    end
    return ids_ml_in_ode
end

function _get_ids_ml_pre_simulate(ml_models::MLModels)::Vector{Symbol}
    out = Symbol[]
    for ml_model in ml_models.ml_models
        ml_id = ml_model.ml_id
        ml_model.static == false && continue
        push!(out, ml_id)
    end
    return out
end

function _get_ids_ml_nondynamic(
        ml_models::MLModels, ids_ml_in_ode::T, ids_ml_pre_simulate::T
    )::T where {T <: Vector{Symbol}}
    out = Symbol[]
    for id in ml_models.ml_ids
        id in Iterators.flatten((ids_ml_in_ode, ids_ml_pre_simulate)) && continue
        push!(out, id)
    end
    return out
end

function _get_ids_ml_input_est(
        petab_tables::PEtabTables, petab_parameters::PEtabParameters, sys::ModelSystem,
        paths::Dict{Symbol, String}, ml_models::MLModels
    )::Vector{Symbol}
    mappings_df, conditions_df = _get_petab_tables(
        petab_tables, [:mapping, :conditions]
    )
    isempty(mappings_df) && return Symbol[]

    out = Symbol[]
    for ml_model in ml_models.ml_models
        ml_model.static == false && continue

        ml_id = ml_model.ml_id
        input_variables = Iterators.flatten(
            _get_ml_model_io_petab_ids(mappings_df, ml_id, :inputs)
        ) .|> Symbol
        input_values = _get_ml_model_input_values(
            input_variables, ml_id, ml_model, conditions_df, petab_tables, paths,
            petab_parameters, sys
        )

        for input_value in input_values
            !(input_value in petab_parameters.parameter_id) && continue
            ip = findfirst(x -> x == input_value, petab_parameters.parameter_id)
            if petab_parameters.estimate[ip] == true
                push!(out, input_value)
            end
        end
    end
    return out
end

function _add_ids_ml_input_est!(
        ids_dynamic_mech::Vector{Symbol}, ids_ml_input_est::Vector{Symbol}
    )::Nothing
    for id in ids_ml_input_est
        id in ids_dynamic_mech && continue
        push!(ids_dynamic_mech, id)
    end
    return nothing
end

function _get_xnames_ps!(ids::Dict{Symbol, Vector{Symbol}}, xscale)::Nothing
    out = similar(ids[:estimate])
    for (i, id) in pairs(ids[:estimate])
        scale = xscale[id]
        if scale == :lin
            out[i] = id
            continue
        end
        out[i] = "$(scale)_$id" |> Symbol
    end
    ids[:estimate_ps] = out
    return nothing
end

function _get_xscales(
        ids::Dict{T, Vector{T}}, petab_parameters::PEtabParameters
    )::Dict{T, T} where {T <: Symbol}
    @unpack parameter_scale, parameter_id = petab_parameters
    s = Symbol[]
    for id in ids[:estimate]
        if id in petab_parameters.parameter_id
            ip = findfirst(x -> x == id, petab_parameters.parameter_id)
            push!(s, petab_parameters.parameter_scale[ip])
        else
            push!(s, :lin)
        end
    end
    return Dict(ids[:estimate] .=> s)
end

function _get_ids_ml_est(
        ids_ml::Vector{Symbol}, petab_ml_parameters::PEtabMLParameters
    )::Vector{Symbol}
    out = Symbol[]
    for ml_id in ids_ml
        ip = findall(x -> x == ml_id, petab_ml_parameters.ml_id)
        for estimate in petab_ml_parameters.estimate[ip]
            estimate == false && continue
            push!(out, ml_id)
            break
        end
    end
    return out
end

function _get_ids_ml(
        ids_ml_in_ode::T, ids_ml_pre_simulate::T, ids_ml_nondynamic::T
    )::T where {T <: Vector{Symbol}}
    return unique(vcat(ids_ml_in_ode, ids_ml_pre_simulate, ids_ml_nondynamic))
end

function _order_id_sys!(ids_sys::T, ids_ml_in_ode::T)::Nothing where {T <: Vector{Symbol}}
    ix = [findfirst(x -> x == id, ids_sys) for id in ids_ml_in_ode]
    ids_sys[sort(ix)] .= ids_ml_in_ode
    return nothing
end
