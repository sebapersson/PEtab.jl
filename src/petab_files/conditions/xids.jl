"""
    _get_xids(petab_parameters, petab_ml_parameters, petab_measurements, sys, petab_tables,
        speciemap, parametermap, ml_models)

Categorize all models parameters.

This is a **very** important function, as parameter type dictates how the gradient should
be computed for any parameter. The assigned parameter categories are later used to
build all the indices used by PEtab.jl to correctly map parameters.
"""
function _get_xids(petab_parameters::PEtabParameters, petab_ml_parameters::PEtabMLParameters, petab_measurements::PEtabMeasurements, sys::ModelSystem, petab_tables::PEtabTables, paths::Dict{Symbol, String}, speciemap, parametermap, ml_models::MLModels)::Dict{Symbol, Vector{Symbol}}
    @unpack observable_parameters, noise_parameters = petab_measurements

    # xids in the ODESystem in correct order
    xids_sys = _get_xids_sys_order(sys, speciemap, parametermap)

    # Neural network parameters.
    # In case of multiple neural networks, to get correct indexing for adjoint gradient
    # methods, nets that appears in the ODE RHS must appear first, and xids_sys must follow
    # the order of xids_ml_in_ode for adjoint indexing to be correct.
    _xids_ml = _get_xids_ml(ml_models)
    xids_ml_in_ode = _get_xids_ml_in_ode(_xids_ml, sys)
    xids_ml_pre_simulate = _get_xids_ml_pre_simulate(ml_models)
    xids_ml_nondynamic = _get_xids_ml_nondynamic(_xids_ml, xids_ml_in_ode, xids_ml_pre_simulate)
    xids_ml = unique(vcat(xids_ml_in_ode, xids_ml_pre_simulate, xids_ml_nondynamic))
    ix = [findfirst(x -> x == id, xids_sys) for id in xids_ml_in_ode]
    xids_sys[sort(ix)] .= xids_ml_in_ode

    # Parameter which are input to a neural net, and are estimated. These must be
    # separately tracked in order to compute correct gradients
    xids_ml_input_est = _get_xids_ml_input_est(petab_tables, petab_parameters, sys, paths, ml_models)

    # Parameters set by a static neural net. Needed to be tracked for gradient computations
    # (as PEtab.jl computes neural net and ODE gradients separately in this case)
    xids_ml_pre_simulate_output = _get_xids_ml_pre_simulate_output(petab_tables, ml_models)

    # Mechanistic (none neural-net parameters). Note non-dynamic parameters are those that
    # only appear in the observable and noise functions, but are not defined noise or
    # observable column of the measurement file.
    xids_observable = _get_xids_observable_noise(observable_parameters, petab_parameters)
    xids_noise = _get_xids_observable_noise(noise_parameters, petab_parameters)
    xids_nondynamic_mech = _get_xids_nondynamic_mech(xids_observable, xids_noise, xids_ml, xids_ml_input_est, sys, petab_parameters, petab_tables, ml_models)
    xids_dynamic_mech = _get_xids_dynamic_mech(xids_observable, xids_noise, xids_nondynamic_mech, xids_ml, petab_parameters)

    # Edge case. If a parameter is in xids_ml_input_est and does not appear in the sys,
    # it must be treated as a xdynamic parameter for gradient purposes since it informs
    # the ODE, so in a sense, it is a form of dynamic parameter.
    _add_xids_ml_input_est_only!(xids_dynamic_mech, xids_ml_input_est)
    xids_not_system_mech = unique(vcat(xids_observable, xids_noise, xids_nondynamic_mech))

    # Bookeep parameters to estimate
    xids_ml_est = _get_xids_ml_est(xids_ml, petab_ml_parameters)
    xids_estimate = vcat(xids_dynamic_mech, xids_not_system_mech, xids_ml_est)
    xids_petab = petab_parameters.parameter_id
    return Dict(:est_to_dynamic_mech => xids_dynamic_mech, :noise => xids_noise, :ml => xids_ml, :ml_est => xids_ml_est, :observable => xids_observable, :nondynamic_mech => xids_nondynamic_mech, :not_system_mech => xids_not_system_mech, :sys => xids_sys, :estimate => xids_estimate, :petab => xids_petab, :ml_in_ode => xids_ml_in_ode, :ml_pre_simulate => xids_ml_pre_simulate, :sys_ml_pre_simulate_outputs => xids_ml_pre_simulate_output, :ml_nondynamic => xids_ml_nondynamic)
end

function _get_xids_dynamic_mech(xids_observable::T, xids_noise::T, xids_nondynamic_mech::T, xids_ml::T, petab_parameters::PEtabParameters)::T where {T <: Vector{Symbol}}
    dynamics_xids = Symbol[]
    other_ids = Iterators.flatten((xids_observable, xids_noise, xids_nondynamic_mech, xids_ml))
    for id in petab_parameters.parameter_id
        if _estimate_parameter(id, petab_parameters) == false
            continue
        end
        if id in other_ids
            continue
        end
        push!(dynamics_xids, id)
    end
    return dynamics_xids
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

function _get_xids_ml_pre_simulate_output(petab_tables::PEtabTables, ml_models::MLModels)::Vector{Symbol}
    out = Symbol[]
    mappings_df = petab_tables[:mapping]
    hybridization_df = petab_tables[:hybridization]
    for (ml_id, ml_model) in ml_models
        ml_model.static == false && continue
        isempty(hybridization_df) && continue
        output_variables = _get_ml_model_io_petab_ids(mappings_df, ml_id, :outputs) |>
            Iterators.flatten
        outputs_df = filter(row -> row.targetValue in output_variables, hybridization_df)
        out = vcat(out, Symbol.(outputs_df.targetId))
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

function _get_xids_nondynamic_mech(xids_observable::T, xids_noise::T, xids_ml::T, xids_ml_input_est::T, sys::ModelSystem, petab_parameters::PEtabParameters, petab_tables::PEtabTables, ml_models::MLModels)::T where {T <: Vector{Symbol}}
    xids_condition = _get_xids_condition(sys, petab_parameters, petab_tables, ml_models)
    xids_sys = _get_xids_sys(sys)
    xids_nondynamic_mech = Symbol[]
    _ids = Iterators.flatten((xids_sys, xids_condition, xids_observable, xids_noise, xids_ml, xids_ml_input_est))
    for id in petab_parameters.parameter_id
        _estimate_parameter(id, petab_parameters) == false && continue
        id in _ids && continue
        push!(xids_nondynamic_mech, id)
    end
    return xids_nondynamic_mech
end

function _get_xids_condition(sys, petab_parameters::PEtabParameters, petab_tables::PEtabTables, ml_models::MLModels)::Vector{Symbol}
    mappings_df = petab_tables[:mapping]
    conditions_df = petab_tables[:conditions]
    xids_sys = parameters(sys) .|> string
    species_sys = _get_state_ids(sys)

    # TODO: Make this a function
    net_inputs = String[]
    for (ml_id, ml_model) in ml_models
        ml_model.static == false && continue
        _net_inputs = _get_ml_model_io_petab_ids(mappings_df, ml_id, :inputs) |>
            Iterators.flatten .|>
            string
        net_inputs = vcat(net_inputs, _net_inputs)
    end

    xids_condition = Symbol[]
    for colname in names(conditions_df)
        colname in ["conditionName", "conditionId"] && continue
        if !(colname in Iterators.flatten((xids_sys, species_sys, net_inputs)))
            throw(PEtabFileError("Parameter $colname that dictates an experimental \
                                  condition does not appear among the model variables"))
        end
        for condition_value in conditions_df[!, colname]
            ismissing(condition_value) && continue
            condition_value isa Real && continue
            is_number(condition_value) && continue

            for parameter_id in petab_parameters.parameter_id
                estimate = _estimate_parameter(parameter_id, petab_parameters)
                for net_input in net_inputs
                    _formula = SBMLImporter._replace_variable(condition_value, "$(net_input)", "")
                    _formula == condition_value && continue
                    throw(PEtabFileError("Neural net input variable $(condition_variable) \
                        setting value of $colname is a simulation condition is not \
                        allowed to be estimated"))
                end

                estimate == false && continue
                _formula = SBMLImporter._replace_variable(condition_value, "$(parameter_id)", "")
                _formula == condition_value && continue
                parameter_id in xids_condition && continue
                push!(xids_condition, parameter_id)
            end
        end
    end
    return xids_condition
end

function _get_xids_ml(ml_models::MLModels)::Vector{Symbol}
    isnothing(ml_models) && return Symbol[]
    return collect(keys(ml_models)) .|> Symbol
end

function _get_xids_ml_in_ode(xids_ml::Vector{Symbol}, sys)::Vector{Symbol}
    !(sys isa ODEProblem) && return Symbol[]
    xids_ml_in_ode = Symbol[]
    for id in xids_ml
        !haskey(sys.p, id) && continue
        push!(xids_ml_in_ode, id)
    end
    return xids_ml_in_ode
end

function _get_xids_ml_pre_simulate(ml_models::MLModels)::Vector{Symbol}
    out = Symbol[]
    for (ml_id, ml_model) in ml_models
        ml_model.static == false && continue
        push!(out, ml_id)
    end
    return out
end

function _get_xids_ml_nondynamic(xids_ml::T, xids_ml_in_ode::T, xids_ml_pre_simulate::T)::T where T <: Vector{Symbol}
    out = Symbol[]
    for id in xids_ml
        id in Iterators.flatten((xids_ml_in_ode, xids_ml_pre_simulate)) && continue
        push!(out, id)
    end
    return out
end

function _get_xids_ml_input_est(petab_tables::PEtabTables, petab_parameters::PEtabParameters, sys::ModelSystem, paths::Dict{Symbol, String}, ml_models::MLModels)::Vector{Symbol}
    mappings_df = petab_tables[:mapping]
    conditions_df = petab_tables[:conditions]
    isempty(mappings_df) && return Symbol[]

    out = Symbol[]
    for (ml_id, ml_model) in ml_models
        ml_model.static == false && continue
        input_variables = _get_ml_model_io_petab_ids(mappings_df, ml_id, :inputs) |>
            Iterators.flatten .|>
            Symbol
        input_values = _get_ml_model_input_values(input_variables, ml_id, ml_model, conditions_df, petab_tables, paths, petab_parameters, sys)
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

function _add_xids_ml_input_est_only!(xids_dynamic_mech::Vector{Symbol}, xids_ml_input_est::Vector{Symbol})::Nothing
    for id in xids_ml_input_est
        id in xids_dynamic_mech && continue
        push!(xids_dynamic_mech, id)
    end
    return nothing
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

function _get_xscales(xids::Dict{T, Vector{T}}, petab_parameters::PEtabParameters)::Dict{T, T} where {T <: Symbol}
    @unpack parameter_scale, parameter_id = petab_parameters
    s = Symbol[]
    for id in xids[:estimate]
        if id in petab_parameters.parameter_id
            ip = findfirst(x -> x == id, petab_parameters.parameter_id)
            push!(s, petab_parameters.parameter_scale[ip])
        else
            push!(s, :lin)
        end
    end
    return Dict(xids[:estimate] .=> s)
end

function _get_xids_ml_est(xids_ml::Vector{Symbol}, petab_ml_parameters::PEtabMLParameters)::Vector{Symbol}
    out = Symbol[]
    for ml_id in xids_ml
        ip = findall(x -> x == ml_id, petab_ml_parameters.ml_id)
        for estimate in petab_ml_parameters.estimate[ip]
            estimate == false && continue
            push!(out, ml_id)
            break
        end
    end
    return out
end
