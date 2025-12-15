import Base

#### Conversions
function Base.convert(::Type{LogLaplace{T}}, μ::S, θ::S) where {T <: Real, S <: Real}
    LogLaplace(T(μ), T(θ))
end
function Base.convert(::Type{LogLaplace{T}}, d::LogLaplace) where {T<:Real}
    LogLaplace{T}(T(d.μ), T(d.θ))
end
Base.convert(::Type{Laplace{T}}, d::LogLaplace{T}) where {T<:Real} = d

#### Parameters
Distributions.params(d::LogLaplace) = (d.μ, d.θ)
@inline Distributions.partype(d::LogLaplace{T}) where {T<:Real} = T

#### Statistics
Distributions.median(d::LogLaplace) = exp(d.μ)

#### pdf, cdf
exp_arg(d::LogLaplace, x::Real) = (log(x) - d.μ) / d.θ

Distributions.pdf(d::LogLaplace, x::Real) = 1 / (2d.θ * x) * exp(-(abs(log(x) - d.μ)) / d.θ)

Distributions.logpdf(d::LogLaplace, x::Real) = - log(2d.θ * x) - abs(log(x) - d.μ) / d.θ

function Distributions.cdf(d::LogLaplace, x::Real)
    t = exp_arg(d, x)
    if t < 0
        return 0.5 * exp(t)
    else
        return 1.0 - 0.5 * exp(-t)
    end
end

function Distributions.logcdf(d::LogLaplace, x::Real)
    t = exp_arg(d, x)
    if t < 0.0
        return t - log(2)
    else
        return log1p(-0.5 * exp(-t))
    end
end

#### Sampling
Distributions.rand(rng::AbstractRNG, d::LogLaplace) = exp(rand(rng, Laplace(d.μ, d.θ)))
