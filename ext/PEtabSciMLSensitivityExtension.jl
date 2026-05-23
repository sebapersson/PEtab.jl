module PEtabSciMLSensitivityExtension

using Catalyst: @unpack
using SciMLBase
using DiffEqCallbacks
using OrdinaryDiffEqBDF
using OrdinaryDiffEqSDIRK
using ForwardDiff
using ReverseDiff
using SciMLSensitivity
using PEtab
import ArrayInterface
import DiffEqBase
import SymbolicIndexingInterface: SymbolicIndexingInterface, state_values

const AdjointAlg = Union{QuadratureAdjoint, InterpolatingAdjoint, GaussAdjoint}

include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "adjoint.jl"))
include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "create_adjoint.jl"))

end
