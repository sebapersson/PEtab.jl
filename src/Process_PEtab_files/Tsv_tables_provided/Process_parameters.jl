"""
    process_parameters(parameters_file::CSV.File)::ParameterInfo

Process the PeTab parameters_file file into a type-stable Julia struct.
"""
function process_parameters(parameters_file::CSV.File; custom_parameter_values::Union{Nothing, Dict}=nothing)::ParametersInfo

    n_parameters = length(parameters_file[:estimate])

    # Pre-allocate arrays to hold data
    lower_bounds::Vector{Float64} = zeros(Float64, n_parameters)
    upper_bounds::Vector{Float64} = zeros(Float64, n_parameters)
    nominal_value::Vector{Float64} = zeros(Float64, n_parameters) # Vector with Nominal value in PeTab-file
    paramter_scale::Vector{Symbol} = Vector{Symbol}(undef, n_parameters)
    parameter_id::Vector{Symbol} = Vector{Symbol}(undef, n_parameters) # Faster to do comparisons with Symbols than Strings
    estimate::Vector{Bool} = Vector{Bool}(undef, n_parameters) #

    for i in eachindex(estimate)

        # If upper or lower bounds are missing assume +Inf and -Inf respectively.
        if ismissing(parameters_file[:lowerBound][i])
            lower_bounds[i] = -Inf
        else
            lower_bounds[i] = parameters_file[:lowerBound][i]
        end
        if ismissing(parameters_file[:upperBound][i])
            upper_bounds[i] = Inf
        else
            upper_bounds[i] = parameters_file[:upperBound][i]
        end

        nominal_value[i] = parameters_file[:nominalValue][i]
        parameter_id[i] = Symbol(string(parameters_file[:parameterId][i]))
        if parameters_file[:parameterScale][i] == "log10"
            paramter_scale[i] = :log10
        elseif parameters_file[:parameterScale][i] == "log"
            paramter_scale[i] = :log
        elseif parameters_file[:parameterScale][i] == "lin"
            paramter_scale[i] = :lin
        else
            error_str = "Parameter scale " * parameters_file[:parameterScale][i] * "not supported. Only log10, log and lin are supported in the Parameters PEtab file under the paramter_scale column."
            throw(PEtabFileError(error_str))
        end

        estimate[i] = parameters_file[:estimate][i] == 1 ? true : false

        # In some case when working with the model the user might want to change model parameters but not go the entire
        # way to the PEtab-files. This ensure ParametersInfo gets its parameters correct.
        if isnothing(custom_parameter_values)
            continue
        end
        keys_dict = collect(keys(custom_parameter_values))
        i_key = findfirst(x -> x == parameter_id[i], keys_dict)
        isnothing(i_key) && continue
        value_change_to = custom_parameter_values[keys_dict[i_key]]
        if typeof(value_change_to) <: Real
            estimate[i] = false
            nominal_value[i] = value_change_to
        elseif is_number(value_change_to)
            estimate[i] = false
            nominal_value[i] = parse(Float64, value_change_to)
        elseif value_change_to == "estimate"
            estimate[i] = true
        else
            PEtabFileError("For PEtab select a parameter must be set to either a estimate or a number not $value_change_to")
        end
    end

    n_parameters_esimtate::Int64 = Int64(sum(estimate))
    return ParametersInfo(nominal_value, lower_bounds, upper_bounds, parameter_id, paramter_scale, estimate, n_parameters_esimtate)
end


function process_priors(θ_indices::ParameterIndices, parameters_file::CSV.File)::PriorInfo

    # In case there are no model priors
    if :objectivePriorType ∉ parameters_file.names
        return PriorInfo(Dict{Symbol, Function}(), Dict{Symbol, Bool}(), false)
    end

    prior_logpdf::Dict{Symbol, Function} = Dict()
    prior_on_parameter_scale::Dict{Symbol, Bool} = Dict()
    for θ_name in θ_indices.θ_names

        which_parameter = findfirst(x -> x == string(θ_name), string.(parameters_file[:parameterId]))
        prior = parameters_file[which_parameter][:objectivePriorType]

        # In case the parameter lacks prior
        if ismissing(prior) || isempty(prior)
            prior_logpdf[θ_name] = no_prior
            prior_on_parameter_scale[θ_name] = false
            continue
        end

        # In case a Julia prior is provided via Catalyst importer
        if occursin("__Julia__", prior)
            prior_parsed = eval(Meta.parse(parse_prior(prior)))
            prior_logpdf[θ_name] = (x) -> logpdf(prior_parsed, x)
            prior_on_parameter_scale[θ_name] = !parameters_file[which_parameter][:priorOnLinearScale]
            continue
        end

        # In case there is a prior is has associated parameters
        prior_parameters = parse.(Float64, split(parameters_file[which_parameter][:objectivePriorParameters], ";"))
        if prior == "parameterScaleNormal"
            prior_logpdf[θ_name] = (x) -> logpdf(Normal(prior_parameters[1], prior_parameters[2]), x)
            prior_on_parameter_scale[θ_name] = true

        elseif prior == "parameterScaleLaplace"
            prior_logpdf[θ_name] = (x) -> logpdf(Laplace(prior_parameters[1], prior_parameters[2]), x)
            prior_on_parameter_scale[θ_name] = true

        elseif prior == "normal"
            prior_logpdf[θ_name] = (x) -> logpdf(Normal(prior_parameters[1], prior_parameters[2]), x)
            prior_on_parameter_scale[θ_name] = false

        elseif prior == "laplace"
            prior_logpdf[θ_name] = (x) -> logpdf(Laplace(prior_parameters[1], prior_parameters[2]), x)
            prior_on_parameter_scale[θ_name] = false

        elseif prior == "logNormal"
            prior_logpdf[θ_name] = (x) -> logpdf(LogNormal(prior_parameters[1], prior_parameters[2]), x)
            prior_on_parameter_scale[θ_name] = false

        elseif prior == "logLaplace"
            @error "Error : Julia does not yet have support for log-laplace"
        else
            @error "Error : PeTab standard does not support a prior of type $priorF"
        end
    end

    return PriorInfo(prior_logpdf, prior_on_parameter_scale, true)
end
# Helper function in case there is not any parameter priors
function no_prior(p::Real)::Real
    return 0.0
end


# Helper funciton to parse if prior has been provided via Catalyst interface 
function parse_prior(str::String)
    _str = replace(str, "__Julia__" => "")
    str_parse = ""
    inside_parenthesis::Bool=false
    do_not_add::Bool=false
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
