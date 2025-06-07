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
- `allow_inf::Bool = false`: Whether to return parameter vectors for which the likelihood
   cannot be computed (typically happens because the `ODEProblem` cannot be solved). Often
   it only makes sense to use starting points with a computable likelihood for
   parameter estimation, hence it typically does not make sense to change this option.
"""
function get_startguesses(prob::PEtabODEProblem, n::Integer; sample_prior::Bool = true,
                          allow_inf::Bool = false,
                          sampling_method::SamplingAlgorithm = LatinHypercubeSample())
    @unpack lower_bounds, upper_bounds, xnames, model_info = prob
    rng = Random.default_rng()

    # Nothing prevents the user from sending in a parameter vector with zero parameters...
    if length(lower_bounds) == 0 && n == 1
        return Float64[]
    elseif length(lower_bounds) == 0
        return [Float64[] for _ in 1:n]
    end
    # In this case a single component array is returned
    if n == 1
        return _single_startguess(prob, sample_prior, allow_inf, rng)
    end

    # Returning a vector of vector
    out = Vector{ComponentArray{Float64}}(undef, n)
    found_starts = 0
    for i in 1:1000
        # QuasiMonteCarlo is deterministic, so for sufficiently few start-guesses we can
        # end up in a never ending loop. To sidestep this if less than 10 starts are
        # left numbers are generated from random Uniform (with potential prior sampling)
        nsamples = n - found_starts
        xmechs = _multiple_mech_startguess!(nsamples, prob, sample_prior, sampling_method)
        for (j, xmech) in pairs(xmechs)
            iout = j + found_starts
            out[iout] = similar(prob.xnominal_transformed)
            @views out[iout][1:length(xmech)] .= xmech
            for ml_model_id in _get_xnames_nn(xnames, model_info)
                @views out[iout][ml_model_id] .= _single_nn_startguess(prob, ml_model_id, rng)
            end
        end
        allow_inf == true && break
        for x in out
            isinf(prob.nllh(x)) && continue
            found_starts += 1
        end
        found_starts == n && break
        if i == 1000
            throw(PEtabInputError("Failed to generate $n startguess that with a finite \
                                   likelihood after 1000 tries"))
        end
    end
    return out
end

function _single_startguess(prob::PEtabODEProblem, sample_prior::Bool, allow_inf::Bool, rng)::ComponentArray{Float64}
    @unpack model_info, xnames, xnominal_transformed = prob
    out = similar(xnominal_transformed)

    # Neural net and mechanistic parameters needs to be treated differently, as they have
    # different modes of start-guess generation
    ix_mech = _get_ixnames_mech(xnames, model_info.petab_parameters)
    xnames_mech = xnames[ix_mech]
    xnames_nn = _get_xnames_nn(xnames, model_info)
    for k in 1:1000
        @views out[ix_mech] .= _single_mech_startguess(prob, xnames_mech, sample_prior)
        for ml_model_id in xnames_nn
            @views out[ml_model_id] .= _single_nn_startguess(prob, ml_model_id, rng)
        end
        allow_inf == true && break
        !isinf(prob.nllh(out)) && break
        if k == 1000
            throw(PEtabInputError("Failed to generate a startguess with a finite \
                                   likelihood within 1000 attempts"))
        end
    end
    return out
end

function _single_mech_startguess(prob::PEtabODEProblem, xnames_mech::Vector{Symbol}, sample_prior::Bool)::Vector{Float64}
    @unpack model_info, lower_bounds, upper_bounds = prob
    out = fill(0.0, length(xnames_mech))
    for (i, id) in pairs(xnames_mech)
        if sample_prior && haskey(model_info.priors.initialisation_distribution, id)
            out[i] = _sample_prior(id, model_info)
        else
            out[i] = rand(Distributions.Uniform(lower_bounds[i], upper_bounds[i]))
        end
    end
    return out
end

function _single_nn_startguess(prob::PEtabODEProblem, ml_model_id::Symbol, rng)::ComponentArray{Float64}
    petab_net_parameters = prob.model_info.petab_net_parameters
    netindices = _get_netindices(ml_model_id, petab_net_parameters.parameter_id)
    out = similar(prob.xnominal_transformed[ml_model_id])
    for netindex in netindices
        id = string(petab_net_parameters.parameter_id[netindex])
        prior = petab_net_parameters.initialisation_priors[netindex]
        if count(".", id) == 0
            out .= _get_nn_startguess(out, prior, rng)
        elseif count(".", id) == 1
            layerid = Symbol(split(id, ".")[2])
            @views out[layerid] .= _get_nn_startguess(out[layerid], prior, rng)
        else
            layerid = Symbol(split(id, ".")[2])
            pid = Symbol(split(id, ".")[3])
            @views out[layerid][pid] .= _get_nn_startguess(out[layerid][pid], prior, rng)
        end
    end
    return out
end

function _multiple_mech_startguess!(nsamples::Int64, prob::PEtabODEProblem, sample_prior::Bool, sampling_method::SamplingAlgorithm)::Vector{Vector{Float64}}
    @unpack model_info, xnames, xnominal_transformed, lower_bounds, upper_bounds = prob
    ix_mech = _get_ixnames_mech(xnames, model_info.petab_parameters)
    xnames_mech = xnames[ix_mech]
    if nsamples > 10
        samples = QuasiMonteCarlo.sample(nsamples, lower_bounds[ix_mech], upper_bounds[ix_mech], sampling_method)
        samples = [samples[:, i] for i in 1:nsamples]
    else
        samples = [_single_mech_startguess(prob, xnames_mech, false)[:] for _ in 1:nsamples]
    end
    # Account for potential priors
    for sample in samples
        sample_prior == false && continue
        for (j, id) in pairs(xnames_mech)
            !haskey(model_info.priors.initialisation_distribution, id) && continue
            sample[j] = _sample_prior(id, model_info)
        end
    end
    return samples
end

function _get_nn_startguess(ps, prior, rng)
    if !(ps isa ComponentArray)
        return prior(rng, ps)
    end
    out = similar(ps)
    for id in keys(ps)
        @views out[id] .= _get_nn_startguess(ps[id], prior, rng)
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
