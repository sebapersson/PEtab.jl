# TODO: Need to make sure consistent type in the statemap (use same as in unknowns)
function _get_statemap(sys::Union{ODESystem, ReactionSystem}, conditions_df::DataFrame,
                       statemap_input)
    specie_ids = _get_state_ids(sys)
    sys_unknowns = unknowns(sys)
    default_values = ModelingToolkit.get_defaults(sys) |> _keys_to_string
    statemap = Vector{Pair}(undef, 0)
    for (i, specieid) in pairs(specie_ids)
        value = haskey(default_values, specieid) ? default_values[specieid] : 0.0
        push!(statemap, sys_unknowns[i] => value)
    end

    # Default values as statemap_input might only set values for subset of species
    if !isnothing(statemap_input)
        for (specie_id, value) in statemap_input
            id = replace(string(specie_id), "(t)" => "")
            !(id in specie_ids) && continue
            is = findfirst(x -> x == id, specie_ids)
            statemap[is] = first(statemap[is]) => value
        end
    end

    # Add extra parameter in case any of the conditions map to a model specie (just as must
    # be done for SBML models)
    for condition_variable in names(conditions_df)
        !(condition_variable in specie_ids) && continue
        pid = "__init__" * string(condition_variable) * "__"
        sys = _add_parameter(sys, pid)
        is = findfirst(x -> x == condition_variable, specie_ids)
        statemap[is] = first(statemap[is]) => eval(Meta.parse("@parameters $pid"))[1]
    end

    # For an ODESystem parameter only appearing in the initial conditions are not a part of
    # the ODESystem parameters. For correct gradient they must be added to the system
    sys = add_u0_parameters(sys, statemap)
    return sys, statemap
end

function _get_parametermap(sys::Union{ODESystem, ReactionSystem}, parametermap_input)
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

function _check_unassigned_variables(variablemap, whichmap::Symbol,
                                     parameters_df::DataFrame,
                                     conditions_df::DataFrame)::Nothing
    for (variableid, value) in variablemap
        value = value |> string
        value != "0.0" && continue

        # As usual specie ids can be on the form S(t) ...
        id = replace(variableid |> string, "(t)" => "")
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

function add_u0_parameters(sys::ReactionSystem, statemap)::ReactionSystem
    return sys
end
function add_u0_parameters(sys::ODESystem, statemap)::ODESystem
    parameter_ids = parameters(sys) .|> string
    specie_ids = _get_state_ids(sys)
    for (id, value) in statemap
        u0exp = replace(string(value), " " => "")
        istart, iend = 1, 1
        char_terminate = ['(', ')', '+', '-', '/', '*', '^']
        while iend < length(u0exp)
            variable, iend = get_word(u0exp, istart, char_terminate)
            istart = istart == iend ? iend + 1 : iend
            isempty(variable) && continue
            is_number(variable) && continue
            variable in parameter_ids && continue
            if variable in specie_ids
                throw(PEtabFormatError("Initial value for specie $id cannot depend on " *
                                       "another specie (in this case $variable"))
            end
            sys = _add_parameter(sys, variable)
        end
    end
    return sys
end

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

function _keys_to_string(d::Dict)::Dict
    keysnew = replace.(keys(d) |> collect .|> string, "(t)" => "")
    return Dict(keysnew .=> values(d))
end

# They removed this function from Catalyst, so I copied it here
function addparam!(rn::ReactionSystem, p; disablechecks = false)
    Catalyst.reset_networkproperties!(rn)
    curidx = disablechecks ? nothing : findfirst(S -> isequal(S, p), ModelingToolkit.get_ps(rn))
    if curidx === nothing
        push!(ModelingToolkit.get_ps(rn), p)
        ModelingToolkit.process_variables!(ModelingToolkit.get_var_to_name(rn),
                                           ModelingToolkit.get_defaults(rn), [p])
        return length(ModelingToolkit.get_ps(rn))
    else
        return curidx
    end
end
