# Evaluate prior contribution. Note, the prior can be on parameter-scale (θ) or on the transformed parameters
# scale (θT)
function compute_priors(θ_parameter_scale::Vector{T},
                        θ_linear_scale::Vector{T},
                        θ_names::Vector{Symbol},
                        prior_info::PriorInfo)::T where {T <: Real}
    prior_value = 0.0
    for (θ_name, logpdf) in prior_info.logpdf
        iθ = findfirst(x -> x == θ_name, θ_names)

        if prior_info.prior_on_parameter_scale[θ_name] == true
            prior_value += logpdf(θ_parameter_scale[iθ])
        else
            prior_value += logpdf(θ_linear_scale[iθ])
        end
    end
    return prior_value
end

function process_priors(θ_indices::ParameterIndices, parameters_df::DataFrame)::PriorInfo

    # In case there are no model priors
    if :objectivePriorType ∉ propertynames(parameters_df)
        return PriorInfo(Dict{Symbol, Function}(),
                         Dict{Symbol, Distribution{Univariate, Continuous}}(),
                         Dict{Symbol, Distribution{Univariate, Continuous}}(),
                         Dict{Symbol, Bool}(),
                         false)
    end

    prior_logpdf::Dict{Symbol, Function} = Dict()
    distribution::Dict{Symbol, Distribution{Univariate, Continuous}} = Dict()
    initialisation_distribution::Dict{Symbol, Distribution{Univariate, Continuous}} = Dict()
    prior_on_parameter_scale::Dict{Symbol, Bool} = Dict()
    for θ_name in θ_indices.xids[:estimate]
        which_parameter = findfirst(x -> x == string(θ_name),
                                    string.(parameters_df[!, :parameterId]))
        prior = parameters_df[which_parameter, :objectivePriorType]

        if ismissing(prior) || isempty(prior)
            continue
        end

        # In case a Julia prior is provided via Catalyst importer
        if occursin("__Julia__", prior)
            prior_parsed = eval(Meta.parse(parse_prior(prior)))
            distribution[θ_name] = prior_parsed
            prior_logpdf[θ_name] = (x) -> logpdf(prior_parsed, x)
            prior_on_parameter_scale[θ_name] = !parameters_df[which_parameter,
                                                              :priorOnLinearScale]

            # Check if the prior should also be on initialisation of parameters
            if :initializationPriorType ∈ propertynames(parameters_df)
                initialisation_prior = parameters_df[which_parameter,
                                                     :initializationPriorType]
                if ismissing(initialisation_prior) || isempty(initialisation_prior)
                    continue
                end
                initialisation_distribution[θ_name] = distribution[θ_name]
            end
            continue
        end

        # In case there is a prior is has associated parameters
        prior_parameters = parse.(Float64,
                                  split(parameters_df[which_parameter,
                                                      :objectivePriorParameters],
                                        ";"))
        if prior == "parameterScaleNormal"
            distribution[θ_name] = Normal(prior_parameters[1], prior_parameters[2])
            prior_logpdf[θ_name] = (x) -> logpdf(Normal(prior_parameters[1],
                                                        prior_parameters[2]), x)
            prior_on_parameter_scale[θ_name] = true

        elseif prior == "parameterScaleLaplace"
            distribution[θ_name] = Laplace(prior_parameters[1], prior_parameters[2])
            prior_logpdf[θ_name] = (x) -> logpdf(Laplace(prior_parameters[1],
                                                         prior_parameters[2]), x)
            prior_on_parameter_scale[θ_name] = true

        elseif prior == "normal"
            distribution[θ_name] = Normal(prior_parameters[1], prior_parameters[2])
            prior_logpdf[θ_name] = (x) -> logpdf(Normal(prior_parameters[1],
                                                        prior_parameters[2]), x)
            prior_on_parameter_scale[θ_name] = false

        elseif prior == "laplace"
            distribution[θ_name] = Laplace(prior_parameters[1], prior_parameters[2])
            prior_logpdf[θ_name] = (x) -> logpdf(Laplace(prior_parameters[1],
                                                         prior_parameters[2]), x)
            prior_on_parameter_scale[θ_name] = false

        elseif prior == "logNormal"
            distribution[θ_name] = LogNormal(prior_parameters[1], prior_parameters[2])
            prior_logpdf[θ_name] = (x) -> logpdf(LogNormal(prior_parameters[1],
                                                           prior_parameters[2]), x)
            prior_on_parameter_scale[θ_name] = false

        elseif prior == "logLaplace"
            @error "Error : Julia does not yet have support for log-laplace"
        else
            @error "Error : PeTab standard does not support a prior of type $priorF"
        end

        # Check if the prior should also be on initialisation of parameters
        if :initializationPriorType ∈ propertynames(parameters_df)
            initialisation_prior = parameters_df[which_parameter, :initializationPriorType]
            if ismissing(initialisation_prior) || isempty(initialisation_prior)
                continue
            end
            initialisation_distribution[θ_name] = distribution[θ_name]
        end
    end

    return PriorInfo(prior_logpdf, distribution, initialisation_distribution,
                     prior_on_parameter_scale, true)
end
# Helper function in case there is not any parameter priors
function no_prior(p::Real)::Real
    return 0.0
end

# Helper funciton to parse if prior has been provided via Catalyst interface
function parse_prior(str::String)
    _str = replace(str, "__Julia__" => "")
    str_parse = ""
    inside_parenthesis::Bool = false
    do_not_add::Bool = false
    for char in _str
        if char == '('
            inside_parenthesis = true
            do_not_add = true
            str_parse *= char
            continue
        end

        if inside_parenthesis == true && do_not_add == true && char == '='
            do_not_add = false
            continue
        end

        if inside_parenthesis == true && char == ','
            do_not_add = true
            str_parse *= char
            continue
        end

        if do_not_add == true
            continue
        end
        str_parse *= char
    end
    return str_parse
end
