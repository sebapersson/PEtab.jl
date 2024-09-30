"""
    get_startguesses(prob::PEtabODEProblem, n::Integer; kwargs...)

Generate `n` random parameter vectors within the parameter bounds in `prob`.

If `n = 1`, a single random vector is returned. For `n > 1`, a vector of random parameter
vectors is returned. In both cases, parameter vectors are returned as a `ComponentArray`.
For details on how to interact with a `ComponentArray`, see the documentation and the
ComponentArrays.jl [documentation](https://github.com/jonniedie/ComponentArrays.jl).

See also [`calibrate`](@ref) and [`calibrate_multistart`](@ref).

## Keyword Arguments
- `sampling_method = LatinHypercubeSample()`: Method for sampling a diverse (spread out) set
   of parameter vectors. Any algorithm from
   [QuasiMonteCarlo](https://github.com/SciML/QuasiMonteCarlo.jl) is allowed, but the
   default `LatinHypercubeSample` is recommended as it usually performs well.
- `sample_prior::Bool = true`: Whether to sample random parameter values from the
   prior distribution if a parameter has a prior.
- `allow_inf::Bool = true`: Whether to return parameter vectors for which the likelihood
   cannot be computed (typically happens because the `ODEProblem` cannot be solved). Often
   it only makes sense to use starting points with a computable likelihood for
   parameter estimation, hence it typically does not make sense to change this option.
"""
function get_startguesses(prob::PEtabODEProblem, n::Integer; sample_prior::Bool = true,
                          allow_inf::Bool = false,
                          sampling_method::SamplingAlgorithm = LatinHypercubeSample())
    @unpack lower_bounds, upper_bounds, xnames, model_info = prob

    # Nothing prevents the user from sending in a parameter vector with zero parameters...
    length(lower_bounds) == 0 && return Float64[]
    # In this case a single component array is returned
    if n == 1
        return _single_startguess(prob, sample_prior, allow_inf)
    end

    # Returning a vector of vector
    out = Vector{ComponentArray{Float64}}(undef, 0)
    found_starts = 0
    for i in 1:1000
        # QuasiMonteCarlo is deterministic, so for sufficiently few start-guesses we can
        # end up in a never ending loop. To sidestep this if less than 10 starts are
        # left numbers are generated from random Uniform (with potential prior sampling)
        nsamples = n - found_starts
        if nsamples > 10
            _samples = QuasiMonteCarlo.sample(nsamples, lower_bounds, upper_bounds,
                                              sampling_method)
            _samples = [_samples[:, i] for i in 1:nsamples]
        else
            _samples = [_single_startguess(prob, false, allow_inf) for _ in 1:nsamples]
        end
        # Account for potential priors
        for _sample in _samples
            sample_prior == false && continue
            for (j, id) in pairs(xnames)
                !haskey(model_info.priors.initialisation_distribution, id) && continue
                _sample[j] = _sample_prior(id, model_info)
            end
        end
        allow_inf == true && break
        for _sample in _samples
            isinf(prob.nllh(_sample)) && continue
            x = deepcopy(prob.xnominal_transformed)
            x .= _sample
            push!(out, x)
            found_starts += 1
        end
        found_starts == n && break
        if i == 1000
            throw(PEtabInputError("Failed to generate $n startguess that with a finite \
                                   likelihood"))
        end
    end
    return out
end

function _single_startguess(prob::PEtabODEProblem, sample_prior::Bool,
                            allow_inf::Bool)::ComponentArray{Float64}
    @unpack model_info, xnames, xnominal_transformed, lower_bounds, upper_bounds = prob
    out = similar(xnominal_transformed)
    for k in 1:1000
        for (i, id) in pairs(xnames)
            if sample_prior && haskey(model_info.priors.initialisation_distribution, id)
                out[i] = _sample_prior(id, model_info)
            else
                out[i] = rand(Distributions.Uniform(lower_bounds[i], upper_bounds[i]))
            end
        end
        allow_inf == true && break
        !isinf(prob.nllh(out)) && break
        if k == 1000
            throw(PEtabInputError("Failed to generate a startguess with a finite \
                                   likelihood"))
        end
    end
    return out
end

function _sample_prior(id::Symbol, model_info::ModelInfo)::Float64
    @unpack priors, xindices = model_info
    dist = priors.initialisation_distribution[id]
    x = rand(dist)
    if priors.prior_on_parameter_scale[id] == false
        x = transform_x(x, xindices.xscale[id]; to_xscale = true)
    end
    return x
end
