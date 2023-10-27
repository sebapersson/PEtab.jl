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

include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "Helper_functions.jl"))
include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "Adjoint.jl"))
include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "Forward_equations.jl"))

include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "Helper_Zygote.jl"))
include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "Cost_Zygote.jl"))
include(joinpath(@__DIR__, "PEtabSciMLSensitivityExtension", "Gradient_Zygote.jl"))

end