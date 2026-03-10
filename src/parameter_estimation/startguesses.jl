"""
    get_startguesses([rng::AbstractRNG], prob::PEtabODEProblem, n::Integer; kwargs...)

Generate `n` random parameter vectors for parameter estimation of `prob`.

`rng` is optional and defaults to `Random.default_rng()`. If `n = 1`, a single parameter
vector is returned. For `n > 1`, a vector of parameter vectors is returned. In both cases,
parameters are returned as a [ComponentArray](https://github.com/jonniedie/ComponentArrays.jl).

For SciML problems (combining mechanistic and ML parameters), mechanistic parameters are
sampled within bounds using `sampling_method`. ML parameters are sampled using
`init_weight` and `init_bias`, which apply globally (to all ML models and layers). For
per-layer control, set initializers when constructing the Lux.jl ML model and use the
defaults `init_weight = nothing` and `init_bias = nothing`.

# Keyword Arguments
- `sampling_method = LatinHypercubeSample()`: Sampling method for mechanistic parameters.
  Any algorithm from [QuasiMonteCarlo](https://github.com/SciML/QuasiMonteCarlo.jl) is
  allowed. The default `LatinHypercubeSample` usually performs well.
- `init_weight = nothing`: Initialization used for ML weight parameters. If `nothing`, the
  default initialization specified by the Lux model is used. Should be an Initializer
  function on the form `init_weight(rng, dims...)`, returning an array of size `dims...`
  (e.g. `zeros64()` or an initializer from WeightInitializers.jl)
- `init_bias = nothing`: As `init_weight`, but for ML bias parameters.
- `sample_prior::Bool = true`: If `true`, parameters with priors are sampled from the prior,
  overriding other sampling schemes.
- `allow_inf::Bool = false`: If `true`, return parameter vectors even if the likelihood is
  not computable (e.g. ODE solver fails).

See also [`calibrate`](@ref) and [`calibrate_multistart`](@ref).
"""
function get_startguesses(
        prob::PEtabODEProblem, n::Integer; sample_prior::Bool = true,
        allow_inf::Bool = false, sampling_method::SamplingAlgorithm = LatinHypercubeSample(),
        init_weight::Union{Nothing, Function} = nothing,
        init_bias::Union{Nothing, Function} = nothing
    )
    rng = Random.default_rng()
    return get_startguesses(
        rng, prob, n; sample_prior = sample_prior, allow_inf = allow_inf,
        sampling_method = sampling_method, init_weight = init_weight, init_bias = init_bias
    )
end
function get_startguesses(
        rng::Random.AbstractRNG, prob::PEtabODEProblem, n::Integer;
        sample_prior::Bool = true, allow_inf::Bool = false,
        sampling_method::SamplingAlgorithm = LatinHypercubeSample(),
        init_weight::Union{Nothing, Function} = nothing,
        init_bias::Union{Nothing, Function} = nothing
    )
    @unpack xnominal_transformed, model_info = prob

    # Only a subset of QuasiMonteCarlo samplers have rng (e.g., Sobol does not)
    if hasproperty(sampling_method, :rng)
        @set sampling_method.rng = rng
    end

    # Nothing prevents the user from sending in a parameter vector with zero parameters...
    if length(xnominal_transformed) == 0 && n == 1
        return Float64[]
    elseif length(xnominal_transformed) == 0
        return [Float64[] for _ in 1:n]
    end

    if n == 1
        return _single_startguess(
            rng, prob, sample_prior, allow_inf, init_weight, init_bias
        )
    end

    # Returning a vector of vector
    ix_mech = _get_ix_mech(prob)
    out = Vector{ComponentArray{Float64}}(undef, 0)
    found_starts = 0
    for k in 1:1000
        # QuasiMonteCarlo is deterministic, so for sufficiently few start-guesses we can
        # end up in a never ending loop. To sidestep this if less than 10 starts are
        # left numbers are generated from random Uniform (with potential prior sampling)
        n_samples = n - found_starts
        candidates = [similar(xnominal_transformed) for _ in 1:n_samples]

        samples_mech = _multiple_mech_startguess(
            rng, n_samples, prob, sample_prior, sampling_method
        )

        for (jx, candidate) in pairs(candidates)
            @views candidate[ix_mech] .= samples_mech[jx]

            for ml_id in model_info.xindices.ids[:ml_est]
                ml_model = model_info.model.ml_models[ml_id]
                ix_ml = model_info.xindices.indices_est[ml_id]
                @views candidate[ix_ml] .= _ml_startguess(
                    rng, ml_model, model_info, init_weight, init_bias, sample_prior
                )
            end
        end

        allow_inf == true && break

        for candidate in candidates
            isinf(prob.nllh(candidate)) && continue
            push!(out, candidate)
            found_starts += 1
        end
        found_starts == n && break

        if k == 1000
            throw(PEtabInputError("Failed to generate a start guess with a finite \
                likelihood within 1000 attempts. Either the parameter bounds are to \
                wide, or something is wrong the with model structure"))
        end
    end
    return out
end

function _single_startguess(
        rng::Random.AbstractRNG, prob::PEtabODEProblem, sample_prior::Bool, allow_inf::Bool,
        init_weight, init_bias,
    )::ComponentArray{Float64}
    @unpack xnominal_transformed, model_info = prob
    out = similar(xnominal_transformed)

    ix_mech = _get_ix_mech(prob)
    for k in 1:1000
        @views out[ix_mech] .= _single_mech_startguess(rng, prob, sample_prior)

        for ml_id in model_info.xindices.ids[:ml_est]
            ml_model = model_info.model.ml_models[ml_id]
            ix_ml = model_info.xindices.indices_est[ml_id]
            @views out[ix_ml] .= _ml_startguess(
                rng, ml_model, model_info, init_weight, init_bias, sample_prior
            )
        end

        allow_inf == true && break
        !isinf(prob.nllh(out)) && break

        if k == 1000
            throw(PEtabInputError("Failed to generate a start guess with a finite \
                likelihood within 1000 attempts. Either the parameter bounds are to \
                wide, or something is wrong the with model structure"))
        end
    end
    return out
end

function _single_mech_startguess(
        rng::Random.AbstractRNG, prob::PEtabODEProblem, sample_prior::Bool
    )::Vector{Float64}
    @unpack model_info, lower_bounds, upper_bounds = prob

    ix_mech = _get_ix_mech(prob)
    out = fill(0.0, length(ix_mech))
    for ix in ix_mech
        if sample_prior && ix in model_info.priors.ix_prior
            out[ix] = _sample_prior(rng, ix, model_info)
        else
            out[ix] = rand(rng, Distributions.Uniform(lower_bounds[ix], upper_bounds[ix]))
        end
    end
    return out
end

function _ml_startguess(
        rng::Random.AbstractRNG, ml_model::MLModel, model_info::ModelInfo, init_weight,
        init_bias, sample_prior::Bool
    )
    x_ml = _get_lux_ps(rng, ComponentArray, ml_model)

    if !isnothing(init_weight)
        for layer_id in keys(x_ml)
            _init_array!(rng, x_ml, init_weight, layer_id, :weight)
        end
    end

    if !isnothing(init_bias)
        for layer_id in keys(x_ml)
            _init_array!(rng, x_ml, init_bias, layer_id, :bias)
        end
    end

    if sample_prior == true
        for ix in model_info.xindices.indices_est[ml_model.ml_id]
            !in(ix, model_info.priors.ix_prior) && continue
            ix_ml = ix - model_info.xindices.indices_est[ml_model.ml_id][1] + 1
            x_ml[ix_ml] = _sample_prior(rng, ix, model_info)
        end
    end
    return x_ml
end

function _multiple_mech_startguess(
        rng::Random.AbstractRNG, n_samples::Int64, prob::PEtabODEProblem,
        sample_prior::Bool, sampling_method::SamplingAlgorithm
    )::Vector{Vector{Float64}}
    @unpack model_info, lower_bounds, upper_bounds = prob

    ix_mech = _get_ix_mech(prob)
    if n_samples > 10
        samples = QuasiMonteCarlo.sample(
            n_samples, lower_bounds[ix_mech], upper_bounds[ix_mech], sampling_method
        )
        samples = [samples[:, i] for i in 1:n_samples]
    else
        samples = [
            _single_mech_startguess(rng, prob, sample_prior)[:] for _ in 1:n_samples
        ]
    end

    for sample in samples
        for ix in ix_mech
            !(sample_prior && ix in model_info.priors.ix_prior) && continue
            sample[ix] = _sample_prior(rng, ix, model_info)
        end
    end
    return samples
end

function _sample_prior(rng::Random.AbstractRNG, ix::Integer, model_info::ModelInfo)::Float64
    @unpack priors, xindices = model_info
    @unpack ix_prior, distributions, priors_on_parameter_scale = priors

    jx = findfirst(x -> x == ix, priors.ix_prior)
    x = rand(rng, distributions[jx])

    # ML parameters are always on linear-scale /so not transformation
    if priors_on_parameter_scale[jx] == false && jx in xindices.indices_est[:est_to_mech]
        x_id = xindices.ids[:estimate][jx]
        x = transform_x(x, xindices.xscale[x_id]; to_xscale = true)
    end
    return x
end

function _init_array!(
        rng::Random.AbstractRNG, x_ml::ComponentArray, init_f::Function, layer_id::Symbol,
        array_id::Symbol
    )::Nothing
    !haskey(x_ml[layer_id], array_id) && return nothing

    @views x_ml[layer_id][array_id] .= init_f(rng, size(x_ml[layer_id][array_id])...)
    return nothing
end

function _get_ix_mech(prob::PEtabODEProblem)
    @unpack xnames, model_info = prob
    ix_mech = findall(x -> !in(x, model_info.xindices.ids[:ml_est]), xnames)
    return ix_mech
end
