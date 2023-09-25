module PlotsExtension

import PEtab: PEtabOptimisationResult, PEtabMultistartOptimisationResult
using Plots
using RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

# For Optimization and model selection
include(joinpath(@__DIR__, "PEtabPlotsExtension", "Plot_recipes.jl"))

end