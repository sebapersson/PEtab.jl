function PEtabParameters(_parameters_df::DataFrame, mappings_df::DataFrame, ml_models::Union{MLModels, Nothing}; custom_values::Union{Nothing, Dict} = nothing)::PEtabParameters
    # Neural-net parameters are parsed in different function, as they have different
    # intialisation, etc...
    imech = _get_parameters_ix(_parameters_df, mappings_df, ml_models, :mechanistic)
    parameters_df = _parameters_df[imech, 1:end]

    _check_values_column(parameters_df, VALID_SCALES, :parameterScale, "parameters")
    _check_values_column(parameters_df, [0, 1], :estimate, "parameters")

    # nominalValue parsed as Vector{String} in case neural-network file appears in the
    # column
    if parameters_df[1, :nominalValue] isa String
        parameters_df[!, :nominalValue] .= parse.(Float64, parameters_df[!, :nominalValue])
    end
    nparameters = nrow(parameters_df)
    parameter_ids = fill(Symbol(), nrow(parameters_df))
    lower_bounds = fill(-Inf, nparameters)
    upper_bounds = fill(Inf, nparameters)
    nominal_values = zeros(Float64, nparameters)
    paramter_scales = fill(Symbol(), nparameters)
    estimate = fill(false, nparameters)

    _parse_table_column!(nominal_values, parameters_df[!, :nominalValue], Float64)
    _parse_table_column!(parameter_ids, parameters_df[!, :parameterId], Symbol)
    _parse_table_column!(paramter_scales, parameters_df[!, :parameterScale], Symbol)
    _parse_table_column!(estimate, parameters_df[!, :estimate], Bool)
    _parse_bound_column!(lower_bounds, parameters_df[!, :lowerBound], estimate)
    _parse_bound_column!(upper_bounds, parameters_df[!, :upperBound], estimate)
    nparameters_estimate = sum(estimate) |> Int64

    # When doing model selection it can be necessary to change the parameter values
    # without changing in the PEtab files. To get all subsequent parameter running
    # correct it must be done here.
    if !isnothing(custom_values)
        for (id, value) in custom_values
            ip = findfirst(x -> x == id, parameter_ids)
            if value == "estimate"
                estimate[ip] = true
            elseif value isa Real
                estimate[ip] = false
                nominal_values[ip] = value
            elseif is_number(value)
                estimate[ip] = false
                nominal_values[ip] = parse(Float64, value)
            end
        end
    end

    return PEtabParameters(nominal_values, lower_bounds, upper_bounds, parameter_ids,
                           paramter_scales, estimate, nparameters_estimate)
end

function PEtabMLParameters(_parameters_df::DataFrame, mappings_df::DataFrame, ml_models::Union{MLModels, Nothing})::PEtabMLParameters
    inet = _get_parameters_ix(_parameters_df, mappings_df, ml_models, :net)
    parameters_df = _parameters_df[inet, 1:end]

    _check_values_column(parameters_df, [0, 1], :estimate, "parameters")

    nparameters = nrow(parameters_df)
    parameter_ids = fill(Symbol(), nrow(parameters_df))
    ml_model_ids = fill(Symbol(), nrow(parameters_df))
    lower_bounds = fill(-Inf, nparameters)
    upper_bounds = fill(Inf, nparameters)
    estimate = fill(false, nparameters)
    mapping_table_ids = fill("", nparameters)

    _parse_table_column!(parameter_ids, parameters_df[!, :parameterId], Symbol)
    _parse_table_column!(estimate, parameters_df[!, :estimate], Bool)
    _parse_bound_column!(lower_bounds, parameters_df[!, :lowerBound], estimate)
    _parse_bound_column!(upper_bounds, parameters_df[!, :upperBound], estimate)
    _get_ml_model_ids!(ml_model_ids, parameter_ids, mappings_df, ml_models)
    _get_mapping_table_ids!(mapping_table_ids, parameter_ids, mappings_df)

    # Nominal-value for net parameters can be either a file name, of a numerical value
    nominal_values = Vector{Union{String, Float64}}(undef, nparameters)
    for (i, nominal_value) in pairs(parameters_df[!, :nominalValue])
        if nominal_value isa Real
            nominal_values[i] = nominal_value
        elseif ismissing(nominal_value)
            nominal_values[i] = ""
        elseif SBMLImporter.is_number(nominal_value)
            nominal_values[i] = parse(Float64, nominal_value)
        else
            nominal_values[i] = nominal_value
        end
    end

    return PEtabMLParameters(nominal_values, lower_bounds, upper_bounds, parameter_ids, estimate, ml_model_ids, mapping_table_ids, Vector{Function}(undef, 0))
end

function Priors(xindices::ParameterIndices, parameters_df::DataFrame)::Priors
    # In case there are no model priors
    if !(:objectivePriorType in propertynames(parameters_df))
        return Priors()
    end

    priors = Dict{Symbol, Distribution{Univariate, Continuous}}()
    on_parameter_scale = Dict{Symbol, Bool}()
    prior_logpdfs = Dict{Symbol, Function}()
    initialisation_dists = Dict{Symbol, Distribution{Univariate, Continuous}}()

    for id in xindices.xids[:estimate]
        irow = findfirst(x -> x == string(id), string.(parameters_df[!, :parameterId]))
        _prior = parameters_df[irow, :objectivePriorType]
        if ismissing(_prior) || isempty(_prior)
            continue
        end

        # Prior provided via the Julia interface
        if occursin("__Julia__", _prior)
            on_parameter_scale[id] = !parameters_df[irow, :priorOnLinearScale]
            priors[id] = _parse_julia_prior(_prior)

            # Prior via the PEtab tables
        else
            prior_parameters = split(parameters_df[irow, :objectivePriorParameters], ";")
            prior_parameters = parse.(Float64, prior_parameters)
            # Different PEtab priros
            if _prior == "parameterScaleNormal"
                on_parameter_scale[id] = true
                μ, σ = prior_parameters
                priors[id] = Normal(μ, σ)
            elseif _prior == "parameterScaleLaplace"
                on_parameter_scale[id] = true
                μ, θ = prior_parameters
                priors[id] = Laplace(μ, θ)
            elseif _prior == "normal"
                on_parameter_scale[id] = false
                μ, σ = prior_parameters
                priors[id] = Normal(μ, σ)
            elseif _prior == "laplace"
                on_parameter_scale[id] = false
                μ, θ = prior_parameters
                priors[id] = Laplace(μ, θ)
            elseif _prior == "logNormal"
                on_parameter_scale[id] = false
                μ, σ = prior_parameters
                priors[id] = LogNormal(μ, σ)
            elseif _prior == "logLaplace"
                throw(PEtabFileError("Julia does not yet support log-laplace distribution"))
            else
                throw(PEtabFileError("$(_prior) is not a valid PEtab prior"))
            end
        end

        prior_logpdfs[id] = let dist = priors[id]
            (x) -> logpdf(dist, x)
        end
        # Check if start-guesses should be sampled from prior as well
        _add_initialisation_prior!(initialisation_dists, priors, id, parameters_df)
    end

    # For remake it is useful if we can flag certain parameters to be skipped when computing
    # the prior
    skip = Symbol[]
    return Priors(prior_logpdfs, priors, initialisation_dists, on_parameter_scale, true,
                  skip)
end

function _add_initialisation_prior!(initialisation_dists::T, priors::T, id::Symbol,
                                    parameters_df::DataFrame)::Nothing where {T <:
                                                                              Dict{Symbol,
                                                                                   Distribution{Univariate,
                                                                                                Continuous}}}
    if !(:initializationPriorType in propertynames(parameters_df))
        return nothing
    end

    irow = findfirst(x -> x == string(id), string.(parameters_df[!, :parameterId]))
    initialisation_prior = parameters_df[irow, :initializationPriorType]
    if ismissing(initialisation_prior) || isempty(initialisation_prior)
        return nothing
    end
    initialisation_dists[id] = priors[id]
    return nothing
end

function _parse_julia_prior(_prior::String)::Distribution{Univariate, Continuous}
    _prior = replace(_prior, "__Julia__" => "")
    # In expressions like Normal{Float64}(μ=0.3, σ=3.0) remove variables to obtain
    # Normal{Float64}(0.3, 3.0)
    _prior = replace(_prior, r"\b\w+\s*=\s*" => "")
    _prior = replace(_prior, "Truncated" => "truncated")
    _prior = replace(_prior, ";" => ",")
    return eval(Meta.parse(_prior))
end

function _get_parameters_ix(parameters_df::DataFrame, mappings_df::DataFrame, ml_models::MLModels, which_ps::Symbol)::Vector{Int64}
    @assert which_ps in [:mechanistic, :net] "Error in PEtabParameters parsing"
    net_ps_variables = String[]
    for ml_model_id in keys(ml_models)
        _net_ps_variables = get_ml_model_petab_variables(mappings_df, ml_model_id, :parameters)
        net_ps_variables = vcat(net_ps_variables, _net_ps_variables)
    end

    out = Int64[]
    for (i, parameter_id) in pairs(parameters_df.parameterId)
        if which_ps == :net && parameter_id in net_ps_variables
            push!(out, i)
        elseif which_ps == :mechanistic && !(parameter_id in net_ps_variables)
            push!(out, i)
        end
    end
    return out
end

function _get_ml_model_ids!(ml_model_ids::Vector{Symbol}, parameter_ids::Vector{Symbol}, mappings_df::DataFrame, ml_models::Dict)::Nothing
    for (i, parameter_id) in pairs(string.(parameter_ids))
        for ml_model_id in keys(ml_models)
            if !(parameter_id in get_ml_model_petab_variables(mappings_df, ml_model_id, :parameters))
                continue
            end
            ml_model_ids[i] = ml_model_id
            break
        end
    end
    return nothing
end

function _get_mapping_table_ids!(mapping_table_ids::Vector{String}, parameter_ids::Vector{Symbol}, mappings_df::DataFrame)::Nothing
    for (i, parameter_id) in pairs(string.(parameter_ids))
        ix = findfirst(x -> x == parameter_id, mappings_df.petabEntityId)
        mapping_table_ids[i] = mappings_df.modelEntityId[ix]
    end
    return nothing
end
