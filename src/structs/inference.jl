struct InferenceInfo{d1 <: Vector{<:Distribution{Univariate, Continuous}},
                     d2 <: Vector{<:Distribution{Univariate, Continuous}},
                     b1,
                     b2}
    priors::d1
    tpriors::d2
    bijectors::b1
    inv_bijectors::b2
    priors_scale::Vector{Symbol}
    parameters_scale::Vector{Symbol}
    parameters_id::Vector{Symbol}
end

"""
PEtabLogDensity(prob::PEtabODEProblem)

Create a `LogDensityProblem` using the posterior and gradient functions from `prob`.

This [`LogDensityProblem` interface](https://github.com/tpapp/LogDensityProblems.jl)
defines everything needed to perform Bayesian inference with packages such as
`AdvancedHMC.jl` (which includes algorithms like NUTS, used by `Turing.jl`), and
`AdaptiveMCMC.jl`.
"""
struct PEtabLogDensity{T <: InferenceInfo,
                       I <: Integer,
                       T2 <: AbstractFloat}
    inference_info::T
    logtarget::Any
    logtarget_gradient::Any
    initial_value::Vector{T2}
    dim::I
end
function (logpotential::PEtab.PEtabLogDensity)(x)
    return logpotential.logtarget(x)
end

"""
    LogLaplace(μ,θ)

The *log-Laplace distribution* with location `μ` and scale `θ` has probability density function

```math
f(x; \\mu, \\theta) = \\frac{1}{2 \\theta x} \\exp \\left(- \\frac{|ln \\, x - \\mu|}{\\theta} \\right)
```

## External links

* [Log-Laplace distribution on Wikipedia](https://en.wikipedia.org/wiki/Log-Laplace_distribution)

## Implementation note

LogLaplace does not yet have support in Distributions.jl, so to support it for
Bayesian inference PEtab.jl implements support for ids pdf, logpdf cdf, logcdf, median
and sampling
"""
struct LogLaplace{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    μ::T
    θ::T
    LogLaplace{T}(µ::T, θ::T) where {T} = new{T}(µ, θ)
end
function LogLaplace(μ::T, θ::T; check_args::Bool=true) where {T <: Real}
    Distributions.@check_args LogLaplace (μ, μ > zero(μ)) (θ, θ > zero(θ))
    return LogLaplace{T}(μ, θ)
end
LogLaplace(μ::Real, θ::Real; check_args::Bool=true) = LogLaplace(promote(μ, θ)...; check_args=check_args)

Distributions.@distr_support LogLaplace 0.0 Inf
