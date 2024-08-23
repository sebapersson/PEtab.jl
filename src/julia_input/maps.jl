function _get_statemap(sys::Union{ODESystem, ReactionSystem}, conditions_df::DataFrame, statemap_input)
    specie_ids = _get_state_ids(sys)
    default_values = ModelingToolkit.get_defaults(sys) |> _keys_to_string
    statemap = Vector{Pair}(undef, 0)
    for specieid in specie_ids
        value = haskey(default_values, specieid) ? default_values[specieid] : "0.0"
        push!(statemap, Symbol(specieid) => value)
    end

    # Default values as statemap_input might only set values for subset of species
    if !isnothing(statemap_input)
        for (specie_id, value) in statemap_input
            id = replace(string(specie_id), "(t)" => "") |> Symbol
            !(id in first.(statemap)) && continue
            is = findfirst(x -> x == id, first.(statemap))
            statemap[is] = id => string(value)
        end
    end

    # Add extra parameter in case any of the conditions map to a model specie (just as must
    # be done for SBML models)
    for condition_variable in names(conditions_df)
        !(condition_variable in specie_ids) && continue
        parameterid = "__init__" * string(condition_variable) * "__"
        _add_parameter!(sys, parameterid)
        condition_variable = Symbol(condition_variable)
        is = findfirst(x -> x == condition_variable, first.(statemap))
        statemap[is] = condition_variable => parameterid
    end

    # For an ODESystem parameter only appearing in the initial conditions are not a part of
    # the ODESystem parameters. For correct gradient they must be added to the system
    add_u0_parameters!(sys, statemap)
    return statemap
end

function _get_parametermap(sys::Union{ODESystem, ReactionSystem}, parametermap_input)
    parametermap = [Num(p) => 0.0 for p in parameters(sys)]
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

function _check_unassigned_variables(variablemap, whichmap::Symbol, parameters_df::DataFrame, conditions_df::DataFrame)::Nothing
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

function add_u0_parameters!(sys::ReactionSystem, statemap)::Nothing
    return nothing
end
function add_u0_parameters!(sys::ODESystem, statemap)::Nothing
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
            _add_parameter!(sys, variable)
        end
    end
    return nothing
end

function _add_parameter!(sys::ReactionSystem, parameter)
    eval(Meta.parse("@parameters " * parameter))
    Catalyst.addparam!(sys, eval(Meta.parse(parameter)))
end
function _add_parameter!(sys::ODESystem, parameter)
    eval(Meta.parse("@parameters $parameter"))
    push!(ModelingToolkit.get_ps(sys),
    eval(Meta.parse("ModelingToolkit.value($parameter)")))
end

function _keys_to_string(d::Dict)::Dict
    keysnew = replace.(keys(d) |> collect .|> string, "(t)" => "")
    return Dict(keysnew .=> values(d))
end
