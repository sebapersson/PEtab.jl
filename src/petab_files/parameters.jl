function PEtabParameters(petab_tables::PEtabTables, ml_models::MLModels)
    parameters_df, mappings_df = _get_petab_tables(petab_tables, [:parameters, :mapping])
    return PEtabParameters(parameters_df, mappings_df, ml_models)
end
function PEtabParameters(
        _parameters_df::DataFrame, mappings_df::DataFrame, ml_models::MLModels;
        custom_values::Union{Nothing, Dict} = nothing
    )::PEtabParameters
    # Neural-net parameters are parsed in different function, as they have different
    # initialization, etc...
    idx_mech = _get_parameters_ix(_parameters_df, mappings_df, ml_models, :mechanistic)
    parameters_df = _parameters_df[idx_mech, 1:end]

    _check_values_column(parameters_df, VALID_SCALES, :parameterScale, "parameters")
    _check_values_column(parameters_df, [0, 1], :estimate, "parameters")

    # nominalValue parsed as Vector{String} in case neural-network file appears in the
    # column
    if parameters_df[1, :nominalValue] isa String
        parameters_df[!, :nominalValue] .= parse.(Float64, parameters_df[!, :nominalValue])
    end
    nps = nrow(parameters_df)
    parameter_ids = fill(Symbol(), nrow(parameters_df))
    lower_bounds = fill(-Inf, nps)
    upper_bounds = fill(Inf, nps)
    nominal_values = zeros(Float64, nps)
    parameter_scales = fill(Symbol(), nps)
    estimate = fill(false, nps)

    _parse_table_column!(nominal_values, parameters_df[!, :nominalValue], Float64)
    _parse_table_column!(parameter_ids, parameters_df[!, :parameterId], Symbol)
    _parse_table_column!(parameter_scales, parameters_df[!, :parameterScale], Symbol)
    _parse_table_column!(estimate, parameters_df[!, :estimate], Bool)
    _parse_bound_column!(lower_bounds, parameters_df[!, :lowerBound], estimate)
    _parse_bound_column!(upper_bounds, parameters_df[!, :upperBound], estimate)
    nps_estimate = sum(estimate) |> Int64

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

    return PEtabParameters(
        nominal_values, lower_bounds, upper_bounds, parameter_ids, parameter_scales,
        estimate, nps_estimate
    )
end

function PEtabMLParameters(petab_tables::PEtabTables, ml_models::MLModels)
    parameters_df, mappings_df = _get_petab_tables( petab_tables, [:parameters, :mapping])
    return PEtabMLParameters(parameters_df, mappings_df, ml_models)
end
function PEtabMLParameters(
        _parameters_df::DataFrame, mappings_df::DataFrame, ml_models::MLModels
    )::PEtabMLParameters
    idx_ml = _get_parameters_ix(_parameters_df, mappings_df, ml_models, :ml)
    parameters_df = _parameters_df[idx_ml, 1:end]

    _check_values_column(parameters_df, [0, 1], :estimate, "parameters")

    nps = nrow(parameters_df)
    parameter_ids = fill(Symbol(), nrow(parameters_df))
    ml_ids = fill(Symbol(), nrow(parameters_df))
    lower_bounds = fill(-Inf, nps)
    upper_bounds = fill(Inf, nps)
    estimate = fill(false, nps)
    mapping_table_ids = fill("", nps)

    _parse_table_column!(parameter_ids, parameters_df[!, :parameterId], Symbol)
    _parse_table_column!(estimate, parameters_df[!, :estimate], Bool)
    _parse_bound_column!(lower_bounds, parameters_df[!, :lowerBound], estimate)
    _parse_bound_column!(upper_bounds, parameters_df[!, :upperBound], estimate)
    _get_mls_ps_petab_ids!(ml_ids, parameter_ids, mappings_df, ml_models)
    _get_mapping_table_ids!(mapping_table_ids, parameter_ids, mappings_df)

    # Nominal-value for net parameters can be either a file name, of a numerical value
    nominal_values = Vector{Union{String, Float64}}(undef, nps)
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

    return PEtabMLParameters(
        nominal_values, lower_bounds, upper_bounds, parameter_ids, estimate, ml_ids,
        mapping_table_ids
    )
end

function Priors(xindices::ParameterIndices, model::PEtabModel)::Priors
    @unpack petab_tables, ml_models = model

    if model.defined_in_julia == false
        petab_version = _get_version(model.paths[:yaml])
    else
        petab_version = "1.0.0"
    end

    # In case there are no model priors
    mappings_df, parameters_df = _get_petab_tables(petab_tables, [:mapping, :parameters])
    if !(:objectivePriorType in propertynames(parameters_df))
        return Priors()
    end

    ix_prior = Int32[]
    priors = ContDistribution[]
    priors_on_parameter_scale = Bool[]
    logpdfs = Function[]

    # Mechanistic parameters
    for (ix, id) in pairs(xindices.ids[:estimate])
        id in xindices.ids[:ml_est] && continue

        row_idx = findfirst(x -> x == string(id), string.(parameters_df[!, :parameterId]))
        prior_id = parameters_df[row_idx, :objectivePriorType]
        if ismissing(prior_id) || isempty(prior_id)
            continue
        end

        push!(ix_prior, ix)

        # Prior provided via the Julia interface
        if occursin("__Julia__", prior_id)
            prior = _parse_julia_prior(prior_id)
            push!(priors, _parse_julia_prior(prior_id))
            push!(priors_on_parameter_scale, false)

        # Prior via the PEtab tables
        else
            prior = _parse_petab_prior(row_idx, parameters_df)
            lb = parameters_df[row_idx, :lowerBound]
            ub = parameters_df[row_idx, :upperBound]
            prior_support = Distributions.support(prior)
            if petab_version == "2.0.0" && (lb > prior_support.lb || ub < prior_support.ub)
                prior = truncated(prior, lb, ub)
            end
            push!(priors, prior)
            push!(priors_on_parameter_scale, PETAB_PRIORS[prior_id].x_scale)
        end
        push!(logpdfs, _get_logpdf(prior))
    end

    # ML parameters
    petab_ml_parameters = PEtabMLParameters(parameters_df, mappings_df, ml_models)
    for ml_id in xindices.ids[:ml_est]
        i_parameters = _get_ml_model_indices(ml_id, petab_ml_parameters.mapping_table_id)
        for ip in i_parameters
            petab_parameter_id = string(petab_ml_parameters.parameter_id[ip])

            row_idx = findfirst(x -> x == petab_parameter_id, parameters_df.parameterId)
            ismissing(parameters_df.objectivePriorType[row_idx]) && continue
            parameters_df.estimate[row_idx] == false && continue

            model_parameter_id = petab_ml_parameters.mapping_table_id[ip]
            if endswith(model_parameter_id, ".parameters")
                ix_ml = xindices.indices_est[ml_id]

            else
                i_start = xindices.indices_est[ml_id][1]
                x_ml = _get_ml_model_initialparameters(model.ml_models[ml_id])
                layer_id = _get_layer_id(model_parameter_id)
                array_id = _get_array_id(model_parameter_id)
                label = isempty(array_id) ? "$(layer_id)" : "$(layer_id).$(array_id)"
                ix_ml = ComponentArrays.label2index(x_ml, label) .+ (i_start - 1)
            end

            prior_id = parameters_df.objectivePriorType[row_idx]
            if occursin("__Julia__", prior_id)
                prior = _parse_julia_prior(prior_id)
            else
                prior = _parse_petab_prior(row_idx, parameters_df)
            end
            _logpdf = _get_logpdf(prior)

            # Need to replace in case of nested priors
            for ix in ix_ml
                if ix in ix_prior
                    jx = findfirst(x -> x == ix, ix_prior)
                    logpdfs[jx] = _logpdf
                    priors[jx] = prior
                else
                    push!(ix_prior, ix)
                    push!(logpdfs, _logpdf)
                    push!(priors_on_parameter_scale, false)
                    push!(priors, prior)
                end
            end
        end
    end

    skip = fill(false, length(ix_prior))
    return Priors(ix_prior, logpdfs, priors, priors_on_parameter_scale, skip)
end

function _parse_petab_prior(
        row_idx::Int64, parameters_df::DataFrame
    )::ContDistribution

    prior_id = parameters_df[row_idx, :objectivePriorType]
    if !haskey(PETAB_PRIORS, prior_id)
        supported_priors = join(collect(keys(PETAB_PRIORS)), ", ")
        throw(PEtabFileError("Unsupported prior $( prior_id ) for parameter $(id) in \
            the PEtab parameter table. Supported priors are: $(supported_priors). \
            See the PEtab standard documentation for details."))
    end

    prior_parameters = split(parameters_df[row_idx, :objectivePriorParameters], ";")
    prior_parameters = parse.(Float64, prior_parameters)
    nps = PETAB_PRIORS[prior_id].n_parameters
    if length(prior_parameters) != nps
        throw(PEtabFileError("Prior $( prior_id) for parameter $(id) expects \
            $(nps) parameter(s), but $(length(prior_parameters)) were provided in \
            the  PEtab parameter table. Provide the expected number of values \
            separated by ; in the parameter table."))
    end
    return PETAB_PRIORS[prior_id].dist(prior_parameters...)
end

function _parse_julia_prior(_prior::String)::ContDistribution
    _prior = replace(_prior, "__Julia__" => "")
    # In expressions like Normal{Float64}(μ=0.3, σ=3.0) remove variables to obtain
    # Normal{Float64}(0.3, 3.0). TODO: Update to support PEtab export
    _prior = replace(_prior, r"\b\w+\s*=\s*" => "")
    _prior = replace(_prior, "Truncated" => "truncated")
    _prior = replace(_prior, ";" => ",")
    return eval(Meta.parse(_prior))
end

function _get_parameters_ix(
        parameters_df::DataFrame, mappings_df::DataFrame, ml_models::MLModels,
        which_ps::Symbol
    )::Vector{Int64}

    @assert which_ps in [:mechanistic, :ml] "Error in PEtabParameters parsing"
    ml_models_ps_ids = String[]
    for ml_id in ml_models.ml_ids
        idx = startswith.(mappings_df.modelEntityId, "$(ml_id).$(parameters)")
        ml_models_ps_ids = vcat(
            ml_models_ps_ids, mappings_df[idx, :petabEntityId]
        )
    end

    out = Int64[]
    for (i, parameter_id) in pairs(parameters_df.parameterId)
        if which_ps == :ml && parameter_id in ml_models_ps_ids
            push!(out, i)
        elseif which_ps == :mechanistic && !(parameter_id in ml_models_ps_ids)
            push!(out, i)
        end
    end
    return out
end

function _get_mls_ps_petab_ids!(
        ml_ids::Vector{Symbol}, parameter_ids::Vector{Symbol}, mappings_df::DataFrame,
        ml_models::MLModels
    )::Nothing
    for (i, parameter_id) in pairs(string.(parameter_ids))
        for ml_id in ml_models.ml_ids
            idx = startswith.(mappings_df.modelEntityId, "$(ml_id).$(parameters)")
            ml_parameters = mappings_df[idx, :petabEntityId]
            if !(parameter_id in ml_parameters)
                continue
            end
            ml_ids[i] = ml_id
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

function _get_logpdf(prior::ContDistribution)::Function
   _logpdf = let dist = prior
        (x) -> logpdf(dist, x)
    end
    return _logpdf
end
