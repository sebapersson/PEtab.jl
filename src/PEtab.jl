module PEtab

using PyCall
using ModelingToolkit
using CSV
using SciMLBase
using SciMLSensitivity
using OrdinaryDiffEq
using DiffEqCallbacks
using SteadyStateDiffEq
using ForwardDiff
using ReverseDiff
import ChainRulesCore 
using Zygote
using StatsBase
using Sundials
using Random
using LinearAlgebra
using Distributions
using Printf
using Requires
using YAML
using RuntimeGeneratedFunctions
using PreallocationTools
using NonlinearSolve
using PrecompileTools
using Optim
using QuasiMonteCarlo

RuntimeGeneratedFunctions.init(@__MODULE__)

include("PEtab_structs.jl")

include("Common.jl")

# Files related to computing the cost (likelihood)
include(joinpath("Compute_cost", "Compute_priors.jl"))
include(joinpath("Compute_cost", "Compute_cost.jl"))
include(joinpath("Compute_cost", "Compute_cost_zygote.jl"))

# Files related to computing derivatives
include(joinpath("Derivatives", "Hessian.jl"))
include(joinpath("Derivatives", "Gradient.jl"))
include(joinpath("Derivatives", "Adjoint_sensitivity_analysis.jl"))
include(joinpath("Derivatives", "Forward_sensitivity_equations.jl"))
include(joinpath("Derivatives", "Gauss_newton.jl"))
include(joinpath("Derivatives", "Common.jl"))
include(joinpath("Derivatives", "ForwardDiff_run_over_chunks.jl"))

# Files related to solving the ODE-system
include(joinpath("Solve_ODE", "Change_experimental_condition.jl"))
include(joinpath("Solve_ODE", "Solve_ode_Zygote.jl"))
include(joinpath("Solve_ODE", "Solve_ode_model.jl"))
include(joinpath("Solve_ODE", "Solve_for_steady_state.jl"))

# Files related to distributed computing
include(joinpath("Distributed", "Distributed.jl"))

# Files related to processing PEtab files
include(joinpath("Process_PEtab_files", "Common.jl"))
include(joinpath("Process_PEtab_files", "Get_simulation_info.jl"))
include(joinpath("Process_PEtab_files", "Get_parameter_indices.jl"))
include(joinpath("Process_PEtab_files", "Process_measurements.jl"))
include(joinpath("Process_PEtab_files", "Process_parameters.jl"))
include(joinpath("Process_PEtab_files", "Process_callbacks.jl"))
include(joinpath("Process_PEtab_files", "Observables", "Common.jl"))
include(joinpath("Process_PEtab_files", "Observables", "Create_h_sigma_derivatives.jl"))
include(joinpath("Process_PEtab_files", "Observables", "Create_u0_h_sigma.jl"))
include(joinpath("Process_PEtab_files", "Read_PEtab_files.jl"))

# For creating a PEtab ODE problem
include(joinpath("Create_PEtab_ODEProblem.jl"))

# Creating the PEtab model
include("Create_PEtab_model.jl")

# Importing SBML models 
include(joinpath("SBML", "SBML_to_ModellingToolkit.jl"))
include(joinpath("SBML", "Common.jl"))
include(joinpath("SBML", "Process_functions.jl"))
include(joinpath("SBML", "Process_rules.jl"))

# For Optimization and model selection 
include(joinpath("Optimization", "Setup_optim.jl"))
include(joinpath("Optimization", "Setup_fides.jl"))
include(joinpath("Optimization", "Callibration.jl"))
include(joinpath("PEtab_select", "PEtab_select.jl"))

#=
# Reduce time for reading a PEtabModel and for building a PEtabODEProblem 
@setup_workload begin
    pathYAML = joinpath(@__DIR__, "..", "test", "Test_model3", "Test_model3.yaml")
    @compile_workload begin
        petabModel = readPEtabModel(pathYAML, verbose=false)
        petabProblem = createPEtabODEProblem(petabModel, verbose=false)
        petabProblem.computeCost(petabProblem.θ_nominalT)
    end
end
=#

export PEtabModel, PEtabODEProblem, ODESolverOptions, SteadyStateSolverOptions, readPEtabModel, createPEtabODEProblem, createOptimProblem, createFidesProblem, callibrateModel, remakePEtabProblem, Fides, runPEtabSelect

end
