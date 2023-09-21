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
path_yaml = joinpath(@__DIR__, "0002", "petab_select_problem.yaml")
path_save = run_PEtab_select(path_yaml, IPNewton(), 
                             n_multistarts=10, 
                             ode_solver=ODESolver(Rodas5P()),
                             gradient_method=:ForwardDiff, 
                             hessian_method=:ForwardDiff)
