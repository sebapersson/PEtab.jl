function _get_speciemap(sys::ModelSystem, conditions_df::DataFrame, mapping_table::DataFrame, speciemap_input)
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
    # be done for SBML models). Also add extra parameter if a Neural-Net maps to a specie
    condition_variables = names(conditions_df)
    net_outputs = String[]
    if !isempty(mapping_table)
        for netid in Symbol.(unique(mapping_table[!, :netId]))
            net_outputs = vcat(net_outputs, _get_net_values(mapping_table, netid, :outputs))
        end
        for net_output in net_outputs
            !(net_output in condition_variables) && continue
            throw(PEtabInputError("Output $(net_output) for a neural-net in the mapping \
                                   table points to a parameter which is set by an
                                   experimental condition. This is not allowed."))
        end
    end
    for variable in Iterators.flatten((condition_variables, net_outputs))
        !(variable in specie_ids) && continue
        pid = "__init__" * string(variable) * "__"
        sys = _add_parameter(sys, pid)
        is = findfirst(x -> x == variable, specie_ids)
        speciemap[is] = first(speciemap[is]) => eval(Meta.parse("@parameters $pid"))[1]
        # Rename output in the mapping table, to have the neural-net map to the
        # initial-value parameter instead
        if variable in net_outputs
            ix = findall(x -> x == variable, mapping_table[!, :ioValue])
            mapping_table[ix, :ioValue] .= pid
        end
    end

    return sys, speciemap
end

function _get_parametermap(sys::ModelSystem, parametermap_input)
    sys isa ODEProblem && return nothing

    parametermap = [Num(p) => 0.0 for p in parameters(sys)]
    # User are allowed to specify default numerical values in the system
    default_values = ModelingToolkit.get_defaults(sys)
    for (i, pid) in pairs(first.(parametermap))
        !haskey(default_values, pid) && continue
        value = default_values[pid]
        if !(value isa Real)
            throw(PEtabInuptError("When setting a parameter to a fixed value in the " *
                                  "model system it must be set to a constant " *
                                  "numberic value. This does not hold for $pid " *
                                  "which is set to $value"))
        end
        parametermap[i] = first(parametermap[i]) => value
    end

    if isnothing(parametermap_input)
        return parametermap
    end
    parameterids = first.(parametermap) .|> string
    for (i, inputid) in pairs(string.(first.(parametermap_input)))
        if !(inputid in parameterids)
            throw(PEtab.PEtabFormatError("Parameter $inputid does not appear among the  " *
                                         "model parameters in the dynamic model"))
        end
        ip = findfirst(x -> x == inputid, parameterids)
        parametermap[ip] = parametermap[ip].first => parametermap_input[i].second
    end
    return parametermap
end

function _check_unassigned_variables(sys::ModelSystem, variablemap, mapinput,
                                     whichmap::Symbol, parameters_df::DataFrame,
                                     conditions_df::DataFrame)::Nothing
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

# TODO: Add for SDEProblem and ODEProblem
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
