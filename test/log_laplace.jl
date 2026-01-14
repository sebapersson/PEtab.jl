#=
    PEtab v2 supports the LogLaplace as prior. This distribution is not yet supported
    in Distributions.jl, so PEtab.jl implements light-weight support for it. These
    tests test whether the implemented logpdf, logcdf and sampling functions are
    correct.
=#

using Distributions, Test
using HypothesisTests: ExactOneSampleKSTest, pvalue
import PEtab: LogLaplace
using StableRNGs: StableRNG

rng = StableRNG(1)
for _ in 1:100
    x1, x2 = rand(rng) * 10.0, rand(rng) * 10.0

    _dist = LogLaplace(x1, x2)
    @test params(_dist) == (x1, x2)

    @test support(_dist).lb == 0.0
    @test isinf(support(_dist).ub)

    x3 = rand(rng, _dist)
    pdf_val = pdf(_dist, x3)
    pdf_log_val = logpdf(_dist, x3)

    if !isinf(abs(pdf_val))
        @test log(pdf_val) ≈ pdf_log_val
    end

    cdf_val = cdf(_dist, x3)
    cdf_log_val = logcdf(_dist, x3)
    @test log(cdf_val) ≈ cdf_log_val atol = 1e-12
end

# Exact ExactOneSampleKSTest testes whether a sample x comes from a distribution
rng = StableRNG(1)
for _ in 1:3
    x1, x2 = rand(rng) * 10.0, rand(rng) * 10.0
    dist = LogLaplace(x1, x2)
    samples = rand(rng, dist, 1000000)
    res = ExactOneSampleKSTest(samples, dist)
    @test pvalue(res) > 0.05
end

# Reference value from PEtab v2 test-suite
dist = truncated(LogLaplace(3.0, 5.0), 0.0, 10.0)
@test logpdf(dist, 5.0) ≈ -3.35750526098019
@test support(dist).lb == 0.0
@test support(dist).ub == 10.0

# Test promotion works as expected
dist = LogLaplace(1.0, 1)
@test params(dist) == (1.0, 1.0)

# Conversion
d1 = LogLaplace(1.0, 1)
d2 = convert(LogLaplace{Float32}, d1)
@test d2 isa LogLaplace{Float32}
@test partype(d2) == Float32
