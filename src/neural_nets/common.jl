function _get_f_nn_preode_x(nnpre::NNPreODE, xdynamic_mech::AbstractVector, pnn::ComponentArray, map_nn::NNPreODEMap)::AbstractVector
    x = get_tmp(nnpre.x, xdynamic_mech)
    x[1:map_nn.nxdynamic_inputs] = xdynamic_mech[map_nn.ixdynamic_mech_inputs]
    @views x[(map_nn.nxdynamic_inputs+1):end] .= pnn
    return x
end

function _get_net_values(mapping_table::DataFrame, netid::Symbol, type::Symbol)::Vector{String}
    dfnet = mapping_table[Symbol.(mapping_table[!, :netId]) .== netid, :]
    if type == :outputs
        dfvals = dfnet[startswith.(string.(dfnet[!, :ioId]), "output"), :]
    elseif type == :inputs
        dfvals = dfnet[startswith.(string.(dfnet[!, :ioId]), "input"), :]
    end
    # Sort to get inputs in order output1, output2, ...
    is = sortperm(string.(dfvals[!, :ioId]), by = x -> parse(Int, match(r"\d+$", x).match))
    return dfvals[is, :ioValue] .|> string
end

function _get_nn_input_variables(inputs::Vector{Symbol}, conditions_df::DataFrame, petab_parameters::PEtabParameters, sys::ModelSystem; keep_numbers::Bool = false, paths::Union{Nothing, Dict{Symbol, String}} = nothing)::Vector{Symbol}
    state_ids = _get_state_ids(sys) .|> Symbol
    xids_sys = _get_xids_sys(sys)
    input_variables = Symbol[]
    for input in inputs
        if is_number(input)
            if keep_numbers == true
                push!(input_variables, input)
            end
            continue
        end
        if input in petab_parameters.parameter_id
            push!(input_variables, input)
            continue
        end
        if input in Iterators.flatten((state_ids, xids_sys))
            push!(input_variables, input)
            continue
        end
        # When building ParameterIndices somtimes only the relative input is provided for
        # the path of a potential input file. To ease downstream processing the complete p
        # ath is provided for downstream processing
        if isfile(string(input))
            push!(input_variables, input)
            continue
        end
        if haskey(paths, :dirmodel) && isfile(joinpath(paths[:dirmodel], string(input)))
            push!(input_variables, Symbol(joinpath(paths[:dirmodel], string(input))))
            continue
        end
        if input in propertynames(conditions_df)
            for condition_value in Symbol.(conditions_df[!, input])
                _input_variables = _get_nn_input_variables([condition_value], conditions_df, petab_parameters, sys; keep_numbers = keep_numbers, paths = paths)
                input_variables = vcat(input_variables, _input_variables)
            end
            continue
        end
        throw(PEtabInputError("Input $input to neural-network cannot be found among ODE \
                               variables, PEtab parameters, or in the conditions table"))
    end
    return input_variables
end

function _get_nns_in_ode(nnmodels::Union{Dict, Nothing})::Dict
    out = Dict()
    isnothing(nnmodels) && return out
    for (id, nnmodel) in nnmodels
        if nnmodel[:input] == "ode" && nnmodel[:output] == "ode"
            out[id] = nnmodel
        end
    end
    return out
end

function _get_n_net_parameters(nn::Union{Dict, Nothing}, xids::Vector{Symbol})::Int64
    isnothing(nn) && return 0
    nparameters = 0
    for xid in xids
        netid = string(xid)[3:end] |> Symbol
        !haskey(nn, netid) && continue
        nparameters += _get_n_net_parameters(nn[netid][2])
    end
    return nparameters
end
