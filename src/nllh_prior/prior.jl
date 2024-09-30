function prior(x::Vector{T}, xnames::Vector{Symbol}, priors::Priors,
               xindices::ParameterIndices)::T where {T <: Real}
    if isempty(priors.logpdf)
        return 0.0
    end

    x_linear = transform_x(x, xnames, xindices, to_xscale = false)
    prior_val = 0.0
    for (xname, _logpdf) in priors.logpdf
        xname in priors.skip && continue
        ix = findfirst(x -> x == xname, xnames)
        if priors.prior_on_parameter_scale[xname] == true
            prior_val += _logpdf(x[ix])
        else
            prior_val += _logpdf(x_linear[ix])
        end
    end
    return prior_val
end
