"""
    ProfileLikelihoodProblem(res, prob::PEtabODEProblem; kwargs...)

Create a LikelihoodProfiler.jl `ProfileLikelihoodProblem` for practical identifiability
analysis using the profile likelihood method.

`res` can be either a parameter estimation result, such as a `PEtabMultistartResult`, or a
`Vector` of parameter values in the order expected by `prob` (see [`get_x`](@ref)). The
bounds in the profile likelihood problem are taken from `prob`.

For how to solve the resulting `ProfileLikelihoodProblem`, see the online documentation and
the [LikelihoodProfiler.jl documentation](https://insysbio.github.io/LikelihoodProfiler.jl/stable/).
"""
function ProfileLikelihoodProblem end
