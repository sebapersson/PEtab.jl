struct PEtabFileError <: Exception
    var::String
end
struct PEtabFormatError <: Exception
    var::String
end
struct PEtabInputError <: Exception
    var::String
end

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

Construct a `LogDensityProblem` using the likelihood and gradient method from the `PEtabODEProblem`.

This LogDensityProblem method defines everything needed to perform Bayesian inference
with libraries such as `AdvancedHMC.jl` (which includes algorithms like NUTS, used by `Turing.jl`),
`AdaptiveMCMC.jl` for adaptive Markov Chain Monte Carlo methods, and `Pigeon.jl` for parallel tempering
methods. For examples on how to perform inference, see the documentation.
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
