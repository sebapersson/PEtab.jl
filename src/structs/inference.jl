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
Bayesian inference PEtab.jl implements support its pdf, logpdf cdf, logcdf, median
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

"""
    Log10Normal(μ,σ)

The *log10 normal distribution* is the distribution of the exponential of a Normal
variate: if ``X \\sim \\operatorname{Normal}(\\mu, \\sigma)`` then
``\\exp10(X) \\sim \\operatorname{Log10Normal}(\\mu,\\sigma)``. The probability density
function is

```math
f(x; \\mu, \\sigma) = \\frac{1}{x \\sqrt{2 \\pi \\sigma^2 \\mathrm{log}(10) }}
\\exp \\left( - \\frac{(\\log_{10}(x) - \\mu)^2}{2 \\sigma^2} \\right),
\\quad x > 0
```


## Implementation note

Only the `logpdf` method is implemented for `Log10Normal` to preserve PEtab v1
compatibility, which allows measurement noise to be distributed in `log10` space. Because
it closely mirrors `LogNormal`, `Log10Normal` is not supported as a prior distribution.
"""
struct Log10Normal{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    μ::T
    σ::T
    Log10Normal{T}(µ::T, σ::T) where {T} = new{T}(µ, σ)
end
function Log10Normal(μ::T, σ::T; check_args::Bool=true) where {T <: Real}
    Distributions.@check_args Log10Normal (σ, σ > zero(σ))
    return Log10Normal{T}(μ, σ)
end
Log10Normal(μ::Real, σ::Real; check_args::Bool=true) = Log10Normal(promote(μ, σ)...; check_args=check_args)

Distributions.@distr_support Log10Normal 0.0 Inf

"""
    Log2Normal(μ,σ)

The *log2 normal distribution* is the distribution of the exponential of a Normal
variate: if ``X \\sim \\operatorname{Normal}(\\mu, \\sigma)`` then
``\\exp2(X) \\sim \\operatorname{Log2Normal}(\\mu,\\sigma)``. The probability density
function is

```math
f(x; \\mu, \\sigma) = \\frac{1}{x \\sqrt{2 \\pi \\sigma^2 \\mathrm{log}(2) }}
\\exp \\left( - \\frac{(\\log_{2}(x) - \\mu)^2}{2 \\sigma^2} \\right),
\\quad x > 0
```


## Implementation note

For the same reasons as for `Log10Normal`, only `logpdf` method is implemented.
"""
struct Log2Normal{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    μ::T
    σ::T
    Log2Normal{T}(µ::T, σ::T) where {T} = new{T}(µ, σ)
end
function Log2Normal(μ::T, σ::T; check_args::Bool=true) where {T <: Real}
    Distributions.@check_args Log2Normal (σ, σ > zero(σ))
    return Log2Normal{T}(μ, σ)
end
Log2Normal(μ::Real, σ::Real; check_args::Bool=true) = Log2Normal(promote(μ, σ)...; check_args=check_args)

Distributions.@distr_support Log2Normal 0.0 Inf
