function PEtab.MLModels(path_yaml::String)::PEtab.MLModels
    problem_yaml = YAML.load_file(path_yaml)
    dir_model = dirname(path_yaml)
    yaml_models = problem_yaml["extensions"]["sciml"]["neural_nets"]

    ml_models = PEtab.MLModel[]
    for (ml_id, model_info) in yaml_models
        ml_id = Symbol(ml_id)
        path_model = joinpath(dirname(path_yaml), model_info["location"])
        static = model_info["static"]
        lux_model, _ = PEtab.parse_to_lux(path_model)

        # With @compact Lux.jl does not allow freezing post model definition, hence the
        # layers to be frozen are extracted here, as well as their parameter values.
        # Given this information, the model is redefined.
        _ml_model = PEtab.MLModel(ml_id, lux_model, static)
        freeze_info = _parse_freeze(_ml_model, path_yaml)

        lux_model, _ = PEtab.parse_to_lux(path_model; freeze_info = freeze_info)
        ml_model = PEtab.MLModel(
            ml_id, lux_model, static; dir_data = dir_model,
            freeze_info = freeze_info
        )
        push!(ml_models, ml_model)
    end
    return PEtab.MLModels(ml_models)
end

function PEtab.MLModel(
        ml_id::Symbol, lux_model::Union{Lux.Chain, Lux.CompactLuxLayer}, static::Bool = true;
        st = nothing, dir_data = nothing, outputs::Vector{T} = Symbol[],
        inputs::Union{Vector{T}, Tuple, Array{<:Real}} = Symbol[],
        freeze_info::Union{Nothing, Dict} = nothing
    )::MLModel where T <: Union{String, Symbol}

    rng = Random.default_rng()
    st = (isnothing(st) ? Lux.initialstates(rng, lux_model) : st) |> f64
    ps = ComponentArray(Lux.initialparameters(rng, lux_model)) |> f64

    # Potentially set values for frozen layers
    if !isnothing(freeze_info)
        for (layer_id, layer_info) in freeze_info
            for (array_id, array_value) in layer_info
                st[layer_id][:frozen_params][array_id] .= array_value
            end
        end
    end

    if isnothing(dir_data)
        dir_data = ""
    elseif !isdir(dir_data)
        throw(PEtab.PEtabInputError("For a MLModel dir_data keyword argument must be a \
            valid directory. This does not hold for $dir_data"))
    end

    if !isempty(inputs) && isempty(outputs)
        throw(PEtab.PEtabInputError("If inputs are provided to an ML model, then outputs \
            must be provided. This does not hold for MLModel with id $(ml_id)"))
    end
    if isempty(inputs) && !isempty(outputs)
        throw(PEtab.PEtabInputError("If outputs are provided to an ML model, then inputs \
            must be provided. This does not hold for MLModel with id $(ml_id)"))
    end

    array_inputs = Dict{Symbol, Array{<:Real}}()
    if inputs isa Array{<:Real}
        _inputs = [_parse_input!(array_inputs, inputs, 1)]
    else
        _inputs = [_parse_input!(array_inputs, input, i) for (i, input) in pairs(inputs)]
    end
    _outputs = [Symbol.(output) for output in outputs]

    return PEtab.MLModel(
        ml_id, lux_model, _inputs, _outputs, static, st, ps, dir_data, array_inputs
    )
end
