import Base

const NOISE_DISTRIBUTIONS = Dict(
    :Normal => (dist = Distributions.Normal, transform = identity),
    :LogNormal => (dist = Distributions.LogNormal, transform = log),
    :Log2Normal => (dist = Log2Normal, transform = log2),
    :Log10Normal => (dist = Log10Normal, transform = log10),
    :Laplace => (dist = Distributions.Laplace, transform = identity),
    :LogLaplace => (dist = LogLaplace, transform = log),
)

function _transform_h(x::T, distribution::Symbol)::T where {T <: Real}
    return NOISE_DISTRIBUTIONS[distribution].transform(x)
end

#### LogLaplace
function Base.convert(::Type{LogLaplace{T}}, μ::S, θ::S) where {T <: Real, S <: Real}
    return LogLaplace(T(μ), T(θ))
end
function Base.convert(::Type{LogLaplace{T}}, d::LogLaplace) where {T <: Real}
    return LogLaplace{T}(T(d.μ), T(d.θ))
end
Base.convert(::Type{Laplace{T}}, d::LogLaplace{T}) where {T <: Real} = d

Distributions.params(d::LogLaplace) = (d.μ, d.θ)
@inline Distributions.partype(d::LogLaplace{T}) where {T <: Real} = T

Distributions.median(d::LogLaplace) = exp(d.μ)

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

Distributions.rand(rng::AbstractRNG, d::LogLaplace) = exp(rand(rng, Laplace(d.μ, d.θ)))

#### Log2Normal
function Distributions.logpdf(d::Log2Normal, x::Real)
    log2_x = log2(x)
    res = (log2_x - d.μ) / d.σ
    return -log(d.σ) - 0.5log(2π) - log(log(2)) - log(2) * log2_x - 0.5 * res^2
end

#### Log10Normal
function Distributions.logpdf(d::Log10Normal, x::Real)
    log10_x = log10(x)
    res = (d.μ - log10_x) / d.σ
    return -log(d.σ) - 0.5 * log(2π) - log(log(10)) - log(10) * log10_x - 0.5 * res^2
end
