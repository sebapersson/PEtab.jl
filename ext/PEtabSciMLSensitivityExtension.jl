module PEtabSciMLSensitivityExtension

import ArrayInterface
import Catalyst: @unpack
import ForwardDiff
import PEtab
import SciMLBase: SciMLBase, AbstractSciMLAlgorithm, CallbackSet, ODEProblem, ODESolution,
    ReturnCode, remake, solve
import SciMLSensitivity: SciMLSensitivity, GaussAdjoint, InterpolatingAdjoint,
    ODEAdjointProblem, QuadratureAdjoint, ReverseDiffVJP, SteadyStateAdjoint
import SymbolicIndexingInterface: SymbolicIndexingInterface, state_values

const AdjointAlg = Union{QuadratureAdjoint, InterpolatingAdjoint, GaussAdjoint}

include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "adjoint.jl"))
include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "create_adjoint.jl"))

end
