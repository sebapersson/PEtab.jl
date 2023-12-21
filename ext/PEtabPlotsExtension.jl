module PEtabPlotsExtension

import PEtab: PEtabOptimisationResult, PEtabMultistartOptimisationResult, PEtabODEProblem, get_odesol, get_obs_comparison_plots
using Plots
using RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

# For Optimization and model selection
include(joinpath(@__DIR__, "PEtabPlotsExtension", "optimisation_trajectory_recipes.jl"))
include(joinpath(@__DIR__, "PEtabPlotsExtension", "optimised_solution_recipes.jl"))

end