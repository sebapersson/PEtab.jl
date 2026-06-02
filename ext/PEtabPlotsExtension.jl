module PEtabPlotsExtension

import Catalyst: @unpack
using ComponentArrays
using DataFrames
using ModelingToolkitBase
using PEtab
using Plots
using PreallocationTools
using RuntimeGeneratedFunctions
using Symbolics

RuntimeGeneratedFunctions.init(@__MODULE__)

# For Optimization and model selection
include(joinpath(@__DIR__, "PEtabPlotsExtension", "optimisation_trajectory_recipes.jl"))
include(joinpath(@__DIR__, "PEtabPlotsExtension", "optimised_solution_recipes.jl"))
include(joinpath(@__DIR__, "PEtabPlotsExtension", "ude_functions_recipes.jl"))

end
