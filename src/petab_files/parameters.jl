function PEtabParameters(parameters_df::DataFrame;
                         custom_values::Union{Nothing, Dict} = nothing)::PEtabParameters
    _check_values_column(parameters_df, VALID_SCALES, :parameterScale, "parameters")
    _check_values_column(parameters_df, [0, 1], :estimate, "parameters")

    nparameters = nrow(parameters_df)
    lower_bounds = fill(-Inf, nparameters)
    upper_bounds = fill(Inf, nparameters)
    nominal_values = zeros(Float64, nparameters) # Vector with Nominal value in PeTab-file
    parameter_ids = fill(Symbol(), nparameters)
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

function Priors(xindices::ParameterIndices, model::PEtabModel)::Priors
    if model.defined_in_julia == false
        petab_version = _get_version(model.paths[:yaml])
    else
        petab_version = "1.0.0"
    end
    parameters_df = model.petab_tables[:parameters]

    # In case there are no model priors
    if !(:objectivePriorType in propertynames(parameters_df))
        return Priors()
    end

    priors = Dict{Symbol, Distribution{Univariate, Continuous}}()
    on_parameter_scale = Dict{Symbol, Bool}()
    prior_logpdfs = Dict{Symbol, Function}()
    initialisation_dists = Dict{Symbol, Distribution{Univariate, Continuous}}()

    for id in xindices.xids[:estimate]
        row_idx = findfirst(x -> x == string(id), string.(parameters_df[!, :parameterId]))
        prior_id = parameters_df[row_idx, :objectivePriorType]
        if ismissing(prior_id) || isempty(prior_id)
            continue
        end

        # Prior provided via the Julia interface
        if occursin("__Julia__", prior_id)
            on_parameter_scale[id] = !parameters_df[row_idx, :priorOnLinearScale]
            priors[id] = _parse_julia_prior(prior_id)

        # Prior via the PEtab tables
        else
            if !haskey(PETAB_PRIORS, prior_id)
                supported_priors = join(collect(keys(PETAB_PRIORS)), ", ")
                throw(PEtabFileError("Unsupported prior $( prior_id ) for parameter $(id) in \
                    the PEtab parameter table. Supported priors are: $(supported_priors). \
                    See the PEtab standard documentation for details."))
            end

            prior_parameters = split(parameters_df[row_idx, :objectivePriorParameters], ";")
            prior_parameters = parse.(Float64, prior_parameters)
            if length(prior_parameters) != PETAB_PRIORS[prior_id].n_parameters
                nps = PETAB_PRIORS[prior_id].n_parameters
                throw(PEtabFileError("Prior $( prior_id) for parameter $(id) expects \
                    $(nps) parameter(s), but $(length(prior_parameters)) were provided in \
                    the  PEtab parameter table. Provide the expected number of values \
                    separated by ; in the parameter table."))
            end

            on_parameter_scale[id] = PETAB_PRIORS[prior_id].x_scale
            priors[id] = PETAB_PRIORS[prior_id].dist(prior_parameters...)

            # PEtab v2 priors are truncated by the parameters upper and lower bound
            lb = parameters_df[row_idx, :lowerBound]
            ub = parameters_df[row_idx, :upperBound]
            prior_support = Distributions.support(priors[id])
            if petab_version == "2.0.0" && (lb > prior_support.lb || ub < prior_support.ub)
                priors[id] = truncated(priors[id], lb, ub)
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

    row_idx = findfirst(x -> x == string(id), string.(parameters_df[!, :parameterId]))
    initialisation_prior = parameters_df[row_idx, :initializationPriorType]
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
