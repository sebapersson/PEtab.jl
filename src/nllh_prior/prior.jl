function prior(
        x::Vector{T}, xnames::Vector{Symbol}, priors::Priors, xindices::ParameterIndices
    )::T where {T <: Real}
    if isempty(priors.ix_prior)
        return 0.0
    end

    x_linear = transform_x(x, xnames, xindices, to_xscale = false)

    prior_val = 0.0
    for (i, ix) in pairs(priors.ix_prior)
        if priors.priors_on_parameter_scale[i] == true
            prior_val += priors.logpdfs[i](x[ix])
        else
            prior_val += priors.logpdfs[i](x_linear[ix])
        end
    end
    return prior_val
end
