module PEtabPlotsExtension

import Catalyst: @unpack
import PEtab: PEtab, PEtabODEProblem
import ModelingToolkitBase
import Plots
import SciMLBase: ODEProblem

# For Optimization and model selection
include(joinpath(@__DIR__, "PEtabPlotsExtension", "optimisation_trajectory_recipes.jl"))
include(joinpath(@__DIR__, "PEtabPlotsExtension", "optimised_solution_recipes.jl"))
include(joinpath(@__DIR__, "PEtabPlotsExtension", "ude_functions_recipes.jl"))

end
