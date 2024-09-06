function prior(x::Vector{T}, xnames::Vector{Symbol}, prior_info::PriorInfo,
               θ_indices::ParameterIndices)::T where T <: Real
    if isempty(prior_info.logpdf)
        return 0.0
    end

    x_linear = transform_x(x, xnames, θ_indices, to_xscale = false)
    prior_val = 0.0
    for (xname, _logpdf) in prior_info.logpdf
        xname in prior_info.skip && continue
        ix = findfirst(x -> x == xname, xnames)
        if prior_info.prior_on_parameter_scale[xname] == true
            prior_val += _logpdf(x[ix])
        else
            prior_val += _logpdf(x_linear[ix])
        end
    end
    return prior_val
end
