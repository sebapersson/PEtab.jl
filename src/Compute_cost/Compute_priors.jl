# Evaluate prior contribution. Note, the prior can be on parameter-scale (θ) or on the transformed parameters
# scale (θT)
function compute_priors(θ_parameter_scale::AbstractVector,
                        θ_linear_scale::AbstractVector,
                        θ_names::Vector{Symbol},
                        prior_info::PriorInfo)::Real

    if prior_info.has_priors == false
        return 0.0
    end

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
