module PEtabPlotsExtension


println("Plots extension laoded")
GGF(X::Float64) = println("PEtab Plots extension loaded sucesfully")

import PEtab: PEtabOptimisationResult, PEtabMultistartOptimisationResult
using Plots
using RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

# For Optimization and model selection
include(joinpath(@__DIR__, "PEtabPlotsExtension", "Plot_recipes.jl"))

end