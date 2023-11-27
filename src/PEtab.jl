module PEtab

using ModelingToolkit
using CSV
using SciMLBase
using OrdinaryDiffEq
using DiffEqCallbacks
using DataFrames
using SteadyStateDiffEq
using ForwardDiff
using ReverseDiff
import ChainRulesCore
using SBMLImporter
using StatsBase
using Sundials
using Random
using LinearAlgebra
using Distributions
using Printf
using YAML
using RuntimeGeneratedFunctions
using PreallocationTools
using NonlinearSolve
using PrecompileTools
using QuasiMonteCarlo

RuntimeGeneratedFunctions.init(@__MODULE__)

include("PEtab_structs.jl")
include("Create_PEtab_model.jl")
include(joinpath("Create_PEtabODEProblem", "Create_PEtab_ODEProblem.jl"))

include("Common.jl")

# Files related to computing the cost (likelihood)
include(joinpath("Compute_cost", "Compute_priors.jl"))
include(joinpath("Compute_cost", "Compute_cost.jl"))

# Files related to computing derivatives
include(joinpath("Derivatives", "Hessian.jl"))
include(joinpath("Derivatives", "Gradient.jl"))
include(joinpath("Derivatives", "Forward_sensitivity_equations.jl"))
include(joinpath("Derivatives", "Gauss_newton.jl"))
include(joinpath("Derivatives", "Common.jl"))
include(joinpath("Derivatives", "ForwardDiff_run_over_chunks.jl"))

# Files related to solving the ODE-system
include(joinpath("Solve_ODE", "Change_experimental_condition.jl"))
include(joinpath("Solve_ODE", "Helper.jl"))
include(joinpath("Solve_ODE", "Solve_ode_model.jl"))
include(joinpath("Solve_ODE", "Solve_for_steady_state.jl"))

# Files related to distributed computing
include(joinpath("Distributed", "Distributed.jl"))

# Files related to processing PEtab files as tables 
include(joinpath("Process_PEtab_files", "Tsv_tables_provided", "Process_measurements.jl"))
include(joinpath("Process_PEtab_files", "Tsv_tables_provided", "Process_parameters.jl"))
include(joinpath("Process_PEtab_files", "Tsv_tables_provided", "Read_PEtab_files.jl"))
# Process a PEtab model when files are provided in Julia (without tsv-tables)
include(joinpath("Process_PEtab_files", "Julia_tables_provided", "Parse_input.jl"))
# Common functionality independent of how model is provided 
include(joinpath("Process_PEtab_files", "Common.jl"))
include(joinpath("Process_PEtab_files", "Get_simulation_info.jl"))
include(joinpath("Process_PEtab_files", "Get_parameter_indices.jl"))
include(joinpath("Process_PEtab_files", "Process_callbacks.jl"))
include(joinpath("Process_PEtab_files", "Observables", "Common.jl"))
include(joinpath("Process_PEtab_files", "Observables", "Create_h_sigma_derivatives.jl"))
include(joinpath("Process_PEtab_files", "Observables", "Create_u0_h_sigma.jl"))

# For creating a PEtab ODE problem
include(joinpath("Create_PEtabODEProblem", "Set_defaults.jl"))
include(joinpath("Create_PEtabODEProblem", "Remake_PEtabODEProblem.jl"))

# Creating the PEtab model
include(joinpath("Process_PEtab_files", "Julia_tables_provided", "Create_PEtab_model.jl"))

# Nice util functions 
include(joinpath("Utility.jl"))

# For correct struct printing
include(joinpath("Show.jl"))

# Reduce time for reading a PEtabModel and for building a PEtabODEProblem
@setup_workload begin
    path_yaml = joinpath(@__DIR__, "..", "test", "Test_model3", "Test_model3.yaml")
    @compile_workload begin
        petab_model = PEtabModel(path_yaml, verbose=false, build_julia_files=true, write_to_file=false)
        petab_problem = PEtabODEProblem(petab_model, verbose=false)
        petab_problem.compute_cost(petab_problem.θ_nominalT)
    end
end

export PEtabModel, PEtabODEProblem, ODESolver, SteadyStateSolver, PEtabModel, PEtabODEProblem, remake_PEtab_problem, Fides, PEtabOptimisationResult, IpoptOptions, IpoptOptimiser, PEtabParameter, PEtabObservable, PEtabMultistartOptimisationResult, generate_startguesses, get_ps, get_u0, get_odeproblem, get_odesol, PEtabEvent

# These are given as extensions, but their docstrings are availble in the 
# general documentation 
include(joinpath("Model_callibration", "Common.jl"))
export calibrate_model, calibrate_model_multistart, run_PEtab_select

if !isdefined(Base, :get_extension)
    include(joinpath(@__DIR__, "..", "ext", "PEtabIpoptExtension.jl"))
    include(joinpath(@__DIR__, "..", "ext", "PEtabOptimExtension.jl"))
    include(joinpath(@__DIR__, "..", "ext", "PEtabFidesExtension.jl"))
    include(joinpath(@__DIR__, "..", "ext", "PEtabOptimizationExtension.jl"))
end
if !isdefined(Base, :get_extension)
    include(joinpath(@__DIR__, "..", "ext", "PEtabSelectExtension.jl"))
end
if !isdefined(Base, :get_extension)    
    include(joinpath(@__DIR__, "..", "ext", "PEtabSciMLSensitivityExtension.jl"))
end
if !isdefined(Base, :get_extension)
    include(joinpath(@__DIR__, "..", "ext", "PEtabCatalystExtension.jl"))
end
if !isdefined(Base, :get_extension)
    include(joinpath(@__DIR__, "..", "ext", "PEtabPlotsExtension.jl"))
end


end
