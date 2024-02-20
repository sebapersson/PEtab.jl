module PEtabSciMLSensitivityExtension

using ChainRulesCore
using SciMLBase
using ModelingToolkit
using DiffEqCallbacks
using SteadyStateDiffEq
using OrdinaryDiffEq
using ForwardDiff
using ReverseDiff
using Zygote
using SciMLSensitivity
using PEtab
import ArrayInterface

include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "Common.jl"))
include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "Adjoint.jl"))
include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "Forward_equations.jl"))
include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "Objective_Zygote.jl"))
include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "Gradient_Zygote.jl"))

end
