"""
    _get_xids(petab_parameters, petab_net_parameters, petab_measurements, sys, petab_tables,
        speciemap, parametermap, nnmodels)

Categorize all models parameters.

This is a **very** important function, as parameter type dictates how the gradient should
be computed for any parameter. The assigned parameter categories are later used to
build all the indices used by PEtab.jl to correctly map parameters.
"""
function _get_xids(petab_parameters::PEtabParameters, petab_net_parameters::PEtabNetParameters, petab_measurements::PEtabMeasurements, sys::ModelSystem, petab_tables::PEtabTables, speciemap, parametermap, nnmodels::Dict{Symbol, <:NNModel})::Dict{Symbol, Vector{Symbol}}
    @unpack observable_parameters, noise_parameters = petab_measurements

    # xids in the ODESystem in correct order
    xids_sys = _get_xids_sys_order(sys, speciemap, parametermap)

    # Neural network parameters.
    # In case of multiple neural networks, to get correct indexing for adjoint gradient
    # methods, nets that appears in the ODE RHS must appear first, and xids_sys must follow
    # the order of xids_nn_in_ode for adjoint indexing to be correct.
    _xids_nn = _get_xids_nn(nnmodels)
    xids_nn_in_ode = _get_xids_nn_in_ode(_xids_nn, sys)
    xids_nn_preode = _get_xids_nn_preode(nnmodels)
    xids_nn_nondynamic = _get_xids_nn_nondynamic(_xids_nn, xids_nn_in_ode, xids_nn_preode)
    xids_nn = unique(vcat(xids_nn_in_ode, xids_nn_preode, xids_nn_nondynamic))
    ix = [findfirst(x -> x == id, xids_sys) for id in xids_nn_in_ode]
    xids_sys[sort(ix)] .= xids_nn_in_ode

    # Parameter which are input to a neural net, and are estimated. These must be
    # separately tracked in order to compute correct gradients
    xids_nn_input_est = _get_xids_nn_input_est(petab_tables, petab_parameters, sys, nnmodels)

    # Parameters set by a static neural net. Needed to be tracked for gradient computations
    # (as PEtab.jl computes neural net and ODE gradients separately in this case)
    xids_nn_preode_output = _get_xids_nn_preode_output(petab_tables, nnmodels)

    # Mechanistic (none neural-net parameters). Note non-dynamic parameters are those that
    # only appear in the observable and noise functions, but are not defined noise or
    # observable column of the measurement file.
    xids_observable = _get_xids_observable_noise(observable_parameters, petab_parameters)
    xids_noise = _get_xids_observable_noise(noise_parameters, petab_parameters)
    xids_nondynamic_mech = _get_xids_nondynamic_mech(xids_observable, xids_noise, xids_nn, xids_nn_input_est, sys, petab_parameters, petab_tables, nnmodels)
    xids_dynamic_mech = _get_xids_dynamic_mech(xids_observable, xids_noise, xids_nondynamic_mech, xids_nn, petab_parameters)

    # Edge case. If a parameter is in xids_nn_input_est and does not appear in the sys,
    # it must be treated as a xdynamic parameter for gradient purposes since it informs
    # the ODE, so in a sense, it is a form of dynamic parameter.
    _add_xids_nn_input_est_only!(xids_dynamic_mech, xids_nn_input_est)
    xids_not_system_mech = unique(vcat(xids_observable, xids_noise, xids_nondynamic_mech))

    # Bookeep parameters to estimate
    xids_nn_est = _get_xids_nn_est(xids_nn, petab_net_parameters)
    xids_estimate = vcat(xids_dynamic_mech, xids_not_system_mech, xids_nn_est)
    xids_petab = petab_parameters.parameter_id
    return Dict(:dynamic_mech => xids_dynamic_mech, :noise => xids_noise, :nn => xids_nn, :nn_est => xids_nn_est, :observable => xids_observable, :nondynamic_mech => xids_nondynamic_mech, :not_system_mech => xids_not_system_mech, :sys => xids_sys, :estimate => xids_estimate, :petab => xids_petab, :nn_in_ode => xids_nn_in_ode, :nn_preode => xids_nn_preode, :nn_preode_outputs => xids_nn_preode_output, :nn_nondynamic => xids_nn_nondynamic)
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

function _get_xids_nn_preode_output(petab_tables::PEtabTables, nnmodels::Dict{Symbol, <:NNModel})::Vector{Symbol}
    out = Symbol[]
    mappings_df = petab_tables[:mapping_table]
    hybridization_df = petab_tables[:hybridization]
    for (netid, nnmodel) in nnmodels
        nnmodel.static == false && continue
        output_variables = _get_net_petab_variables(mappings_df, netid, :outputs)
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

function _get_xids_nondynamic_mech(xids_observable::T, xids_noise::T, xids_nn::T, xids_nn_input_est::T, sys::ModelSystem, petab_parameters::PEtabParameters, petab_tables::PEtabTables, nnmodels::Dict{Symbol, <:NNModel})::T where {T <: Vector{Symbol}}
    xids_condition = _get_xids_condition(sys, petab_parameters, petab_tables, nnmodels)
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

function _get_xids_condition(sys, petab_parameters::PEtabParameters, petab_tables::PEtabTables, nnmodels::Dict{Symbol, <:NNModel})::Vector{Symbol}
    mappings_df = petab_tables[:mapping_table]
    conditions_df = petab_tables[:conditions]
    xids_sys = parameters(sys) .|> string
    species_sys = _get_state_ids(sys)
    net_inputs = String[]
    for (netid, nnmodel) in nnmodels
        nnmodel.static == false && continue
        input_variables = _get_net_petab_variables(mappings_df, netid, :inputs) .|> string
        net_inputs = vcat(net_inputs, input_variables)
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

function _get_xids_nn(nnmodels::Union{Nothing, Dict{Symbol, <:NNModel}})::Vector{Symbol}
    isnothing(nnmodels) && return Symbol[]
    return collect(keys(nnmodels)) .|> Symbol
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

function _get_xids_nn_preode(nnmodels::Dict{Symbol, <:NNModel})::Vector{Symbol}
    out = Symbol[]
    for (netid, nnmodel) in nnmodels
        nnmodel.static == false && continue
        push!(out, netid)
    end
    return out
end

function _get_xids_nn_nondynamic(xids_nn::T, xids_nn_in_ode::T, xids_nn_preode::T)::T where T <: Vector{Symbol}
    out = Symbol[]
    for id in xids_nn
        id in Iterators.flatten((xids_nn_in_ode, xids_nn_preode)) && continue
        push!(out, id)
    end
    return out
end

function _get_xids_nn_input_est(petab_tables::PEtabTables, petab_parameters::PEtabParameters, sys::ModelSystem, nnmodels::Dict{Symbol, <:NNModel})::Vector{Symbol}
    mappings_df = petab_tables[:mapping_table]
    conditions_df = petab_tables[:conditions]
    isempty(mappings_df) && return Symbol[]

    out = Symbol[]
    for (netid, nnmodel) in nnmodels
        nnmodel.static == false && continue
        input_variables = _get_net_petab_variables(mappings_df, netid, :inputs) .|> Symbol
        input_values = _get_net_input_values(input_variables, netid, nnmodel, conditions_df, petab_tables, petab_parameters, sys)
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

function _add_xids_nn_input_est_only!(xids_dynamic_mech::Vector{Symbol}, xids_nn_input_est::Vector{Symbol})::Nothing
    for id in xids_nn_input_est
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

function _get_xids_nn_est(xids_nn::Vector{Symbol}, petab_net_parameters::PEtabNetParameters)::Vector{Symbol}
    out = Symbol[]
    for netid in xids_nn
        ip = findfirst(x -> x == netid, petab_net_parameters.netid)
        petab_net_parameters.estimate[ip] == false && continue
        push!(out, netid)
    end
    return out
end
