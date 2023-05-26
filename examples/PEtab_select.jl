#= 
    Example on how to run PEtab-select with PEtab.jl. For additional details see 
    the documentation.
=#

# Setup correct Python environment with PEtab Select installed 
using PyCall
pathPythonExe = joinpath("/", "home", "sebpe", "anaconda3", "envs", "PeTab", "bin", "python")
ENV["PYTHON"] = pathPythonExe
import Pkg; Pkg.build("PyCall")

using PEtab 
using OrdinaryDiffEq
using Optim
pathYAML = joinpath(@__DIR__, "0002", "petab_select_problem.yaml")
pathSave = runPEtabSelect(pathYAML, IPNewton(), 
                          nOptimisationStarts=10, 
                          odeSolverOptions=ODESolverOptions(Rodas5P()),
                          gradientMethod=:ForwardDiff, 
                          hessianMethod=:ForwardDiff)
