module PEtabSciMLSensitivityExtension

using ChainRulesCore
using SciMLBase
using ModelingToolkit
using DiffEqCallbacks
using SteadyStateDiffEq
using OrdinaryDiffEq
using ForwardDiff
using ReverseDiff
using SciMLSensitivity
using PEtab
import ArrayInterface

const AdjointAlg = Union{QuadratureAdjoint, InterpolatingAdjoint, GaussAdjoint}
const ForwardAlg = Union{ForwardSensitivity, ForwardDiffSensitivity}

include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "adjoint.jl"))
include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "create_adjoint.jl"))
include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "forward_eqs.jl"))

end
