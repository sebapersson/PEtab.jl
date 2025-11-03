function _get_speciemap(sys::ModelSystem, conditions_df::DataFrame, hybridization_df::DataFrame, ml_models::MLModels , speciemap_input)
    specie_ids = _get_state_ids(sys)
    speciemap_ids = _get_speciemap_ids(sys)
    default_values = _get_default_values(sys)
    speciemap = Vector{Pair}(undef, 0)
    for (i, specieid) in pairs(specie_ids)
        value = haskey(default_values, specieid) ? default_values[specieid] : 0.0
        push!(speciemap, speciemap_ids[i] => value)
    end

    # Default values as speciemap_input might only set values for subset of species
    if !isnothing(speciemap_input)
        for (specie_id, value) in speciemap_input
            id = replace(string(specie_id), "(t)" => "")
            !(id in specie_ids) && continue
            is = findfirst(x -> x == id, specie_ids)
            speciemap[is] = first(speciemap[is]) => value
        end
    end

    # Add extra parameter in case any of the conditions map to a model specie (just as must
    # be done for SBML models). An extra parameter must also be added if a neural network
    # maps to a specie
    condition_variables = names(conditions_df)
    net_outputs = String[]
    for ml_model in values(ml_models)
        ml_model.static == false && continue
        for output_id in string.(ml_model.outputs)
            if output_id in specie_ids
                push!(net_outputs, output_id)
            end
        end
    end

    # Add extra parameter in case any of the conditions map to a model specie (just as must
    # be done for SBML models). As NaN is allowed value in the conditions table, need to
    # save model map. See comment in petab_model file for standard format import.
    speciemap_model = deepcopy(speciemap)
    for variable in Iterators.flatten((condition_variables, net_outputs))
        !(variable in specie_ids) && continue
        pid = "__init__" * string(variable) * "__"
        sys = _add_parameter(sys, pid)
        is = findfirst(x -> x == variable, specie_ids)
        speciemap[is] = first(speciemap[is]) => eval(Meta.parse("@parameters $pid"))[1]
        # Rename output in the mapping table, to have the neural-net map to the
        # initial-value parameter instead
        if variable in net_outputs
            ix = findall(x -> x == variable, hybridization_df[!, :targetId])
            hybridization_df[ix, :targetId] .= pid
        end
    end
    return sys, speciemap_model, speciemap
end

function _get_parametermap(sys::ODEProblem, ::Any, conditions_df::DataFrame, parameters_df::DataFrame, ml_models::MLModels)
    specie_ids = string.(keys(sys.u0))
    for specie_id in specie_ids
        !(specie_id in names(conditions_df)) && continue
        for condition_value in conditions_df[!, specie_id]
            !(condition_value in parameters_df.parameterId) && continue
            condition_value in string.(keys(sys.p)) && continue
            sys = _add_parameter(sys, condition_value)
        end
    end
    # Need to re-order sys.p such that ML-model are last for correct indexing
    if isempty(ml_models) || !any([ml_id in keys(ml_models) for ml_id in keys(ml_models)])
        return sys, nothing
    end
    pkeys = keys(sys.p)
    ml_ids = collect(keys(ml_models))
    ml_ode_ids = filter(in(pkeys), ml_ids)
    mech_ids = (k for k in pkeys if k ∉ ml_ode_ids)
    p_ode = (; (k => sys.p[k] for k in mech_ids)...,
            (k => sys.p[k] for k in ml_ode_ids)...)
    sys = remake(sys, p = ComponentArray(p_ode))
    return sys, nothing
end
function _get_parametermap(sys::ModelSystem, parametermap_input, conditions_df::DataFrame, parameters_df::DataFrame, ::MLModels)
    parametermap = [Num(p) => 0.0 for p in parameters(sys)]
    # User are allowed to specify default numerical values in the system
    default_values = ModelingToolkit.get_defaults(sys)
    for (i, pid) in pairs(first.(parametermap))
        !haskey(default_values, pid) && continue
        value = default_values[pid]
        if !(value isa Real)
            throw(PEtabInuptError("When setting a parameter to a fixed value in the \
                                   model system it must be set to a constant \
                                   numberic value. This does not hold for $pid \
                                   which is set to $value"))
        end
        parametermap[i] = first(parametermap[i]) => value
    end

    if !isnothing(parametermap_input)
        parameterids = first.(parametermap) .|> string
        for (i, inputid) in pairs(string.(first.(parametermap_input)))
            if !(inputid in parameterids)
                throw(PEtab.PEtabFormatError("Parameter $inputid does not appear among the \
                                            model parameters in the dynamic model"))
            end
            ip = findfirst(x -> x == inputid, parameterids)
            parametermap[ip] = parametermap[ip].first => parametermap_input[i].second
        end
    end

    # Check if for any of condition, initial values are given by a parameter set to
    # be estimated. In that case, the parameter must be added to the system as it is
    # a dynamic parameter
    specie_ids = _get_state_ids(sys)
    for specie_id in specie_ids
        !(specie_id in names(conditions_df)) && continue
        for condition_value in conditions_df[!, specie_id]
            !(condition_value in parameters_df.parameterId) && continue
            condition_value in string.(first.(parametermap)) && continue
            push!(parametermap, eval(Meta.parse("@parameters $(condition_value)"))[1] => 0.0)
            sys = _add_parameter(sys, condition_value)
        end
    end
    return sys, parametermap
end

function _check_unassigned_variables(sys::ModelSystem, variablemap, mapinput, whichmap::Symbol, parameters_df::DataFrame, conditions_df::DataFrame)::Nothing
    if whichmap == :parameter && sys isa ODEProblem
        return nothing
    end
    default_values = _get_default_values(sys)
    if !isnothing(mapinput)
        ids_input = replace.(first.(mapinput) .|> string, "(t)" => "")
    else
        ids_input = nothing
    end
    for (variableid, value) in variablemap
        value = value |> string
        value != "0.0" && continue
        haskey(default_values, variableid) && continue

        # As usual specie ids can be on the form S(t) ...
        id = replace(variableid |> string, "(t)" => "")
        !isnothing(ids_input) && id in ids_input && continue
        # Parameter for initial value
        if length(id) > 8 && id[1:8] == "__init__"
            continue
        end
        id in parameters_df[!, :parameterId] && continue
        id in names(conditions_df) && continue
        @warn "The $(whichmap) $id has not been assigned a value among PEtabParameters, " *
              "simulation conditions, or in the $(whichmap) map. It default to 0."
    end
end

# TODO: Add for SDEProblem
function _add_parameter(sys::ReactionSystem, parameter)
    _p = Symbol(parameter)
    addparam!(sys, only(@parameters($_p)))
    return sys
end
function _add_parameter(sys::ODESystem, parameter)
    # Mutating an ODESystem is ill-advised, therefore a new system is built with additional
    # parameter
    p = parameter |> Symbol
    pnew = vcat(parameters(sys), only(@parameters($p)))
    @named de = ODESystem(equations(sys), Catalyst.default_t(), unknowns(sys), pnew)
    return complete(de)
end
function _add_parameter(sys::ODEProblem, parameter)
    # For an ODEProblem p can be arbitrary struct (we enfroce ComponentArray though for
    # parameter mapping)
    @assert sys.p isa ComponentArray "p for ODEProblem must be a ComponentArray"
    _p = sys.p |> NamedTuple
    _padd = NamedTuple{(Symbol(parameter), )}((0.0, ))
    p = ComponentArray(merge(_p, _padd))
    return remake(sys, p = p)
end

function _keys_to_string(d::Union{Dict, NamedTuple})::Dict
    keysnew = replace.(keys(d) |> collect .|> string, "(t)" => "")
    return Dict(keysnew .=> values(d))
end
function _keys_to_string(d::ComponentArray)::Dict
    keysnew = replace.(keys(d) |> collect .|> string, "(t)" => "")
    return Dict(keysnew .=> collect(d))
end

# They removed this function from Catalyst, so I copied it here
function addparam!(rn::ReactionSystem, p; disablechecks = false)
    Catalyst.reset_networkproperties!(rn)
    curidx = disablechecks ? nothing :
             findfirst(S -> isequal(S, p), ModelingToolkit.get_ps(rn))
    if curidx === nothing
        push!(ModelingToolkit.get_ps(rn), p)
        ModelingToolkit.process_variables!(ModelingToolkit.get_var_to_name(rn),
                                           ModelingToolkit.get_defaults(rn), [p])
        return length(ModelingToolkit.get_ps(rn))
    else
        return curidx
    end
end

function _get_speciemap_ids(sys::ODEProblem)
    return keys(sys.u0) |> collect
end
function _get_speciemap_ids(sys)
    return unknowns(sys)
end

function _get_default_values(sys)
    return ModelingToolkit.get_defaults(sys) |> _keys_to_string
end
function _get_default_values(sys::ODEProblem)
    return _keys_to_string(sys.u0)
end
