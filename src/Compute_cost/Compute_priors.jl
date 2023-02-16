# Evaluate prior contribution. Note, the prior can be on parameter-scale (θ) or on the transformed parameters
# scale (θT)
function computePriors(θ_parameterScale::AbstractVector,
                       θ_linearScale::AbstractVector,
                       θ_names::Vector{Symbol},
                       priorInfo::PriorInfo)::Real

    if priorInfo.hasPriors == false
        return 0.0
    end
    
    priorValue = 0.0
    for (i, θ_name) in pairs(θ_names)
        logpdf = priorInfo.logpdf[θ_name]
        if priorInfo.priorOnParameterScale[θ_name] == true
            priorValue += logpdf(θ_parameterScale[i])
        else
            priorValue += logpdf(θ_linearScale[i])
        end
    end

    return priorValue
end
