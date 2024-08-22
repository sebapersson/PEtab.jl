module PEtab

using ModelingToolkit
using CSV
using SciMLBase
using OrdinaryDiffEq
using Catalyst
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
using StyledStrings

RuntimeGeneratedFunctions.init(@__MODULE__)

include("Structs.jl")
include(joinpath("PEtabModel", "Julia_input.jl"))

include("Common.jl")
include("logging.jl")

include(joinpath("petab_files", "common.jl"))
include(joinpath("petab_files", "table_info.jl"))
include(joinpath("petab_files", "read.jl"))
include(joinpath("petab_files", "parameters.jl"))
include(joinpath("petab_files", "measurements.jl"))
include(joinpath("petab_files", "conditions.jl"))
include(joinpath("petab_files", "simulations.jl"))
include(joinpath("petab_files", "petab_model.jl"))

# Files related to computing the cost (likelihood)
include(joinpath("Objective", "Priors.jl"))
include(joinpath("Objective", "Objective.jl"))

# Files related to computing derivatives
include(joinpath("Derivatives", "Hessian.jl"))
include(joinpath("Derivatives", "Gradient.jl"))
include(joinpath("Derivatives", "Forward_sensitivity_equations.jl"))
include(joinpath("Derivatives", "Gauss_newton.jl"))
include(joinpath("Derivatives", "Common.jl"))
include(joinpath("Derivatives", "ForwardDiff_chunks.jl"))

# Files related to solving the ODE-system
include(joinpath("Solve_ODE", "Switch_condition.jl"))
include(joinpath("Solve_ODE", "Common.jl"))
include(joinpath("Solve_ODE", "Solve.jl"))
include(joinpath("Solve_ODE", "Steady_state.jl"))

# Files related to processing user input
include(joinpath("Process_input", "Julia_input.jl"))
include(joinpath("Process_input", "Common.jl"))
include(joinpath("Process_input", "Callbacks.jl"))
include(joinpath("Process_input", "Observables", "Common.jl"))
include(joinpath("Process_input", "Observables", "h_sigma_derivatives.jl"))
include(joinpath("Process_input", "Observables", "u0_h_sigma.jl"))

# For creating a PEtabODEProblem
include(joinpath("PEtabODEProblem", "Defaults.jl"))
include(joinpath("PEtabODEProblem", "Remake.jl"))
include(joinpath("PEtabODEProblem", "Cache.jl"))
include(joinpath("PEtabODEProblem", "Create.jl"))

# Nice util functions
include(joinpath("Utility.jl"))

# For correct struct printing
include(joinpath("Show.jl"))

#=
# Reduce time for reading a PEtabModel and for building a PEtabODEProblem
@setup_workload begin
    path_yaml = joinpath(@__DIR__, "..", "test", "Test_model3", "Test_model3.yaml")
    @compile_workload begin
        petab_model = PEtabModel(path_yaml, verbose = false, build_julia_files = true,
                                 write_to_file = false)
        petab_problem = PEtabODEProblem(petab_model, verbose = false)
        petab_problem.compute_cost(petab_problem.θ_nominalT)
    end
end
=#

export PEtabModel, PEtabODEProblem, ODESolver, SteadyStateSolver, PEtabModel,
       PEtabODEProblem, remake_PEtab_problem, Fides, PEtabOptimisationResult, IpoptOptions,
       IpoptOptimiser, PEtabParameter, PEtabObservable, PEtabMultistartOptimisationResult,
       generate_startguesses, get_ps, get_u0, get_odeproblem, get_odesol, PEtabEvent,
       PEtabLogDensity, solve_all_conditions, compute_runtime_accuracy, PEtabPigeonReference

# These are given as extensions, but their docstrings are availble in the
# general documentation
include(joinpath("Calibrate", "Common.jl"))
export calibrate_model, calibrate_model_multistart, run_PEtab_select
function get_obs_comparison_plots end
export get_obs_comparison_plots

# Functions that only appear in extension
function compute_llh end
function compute_prior end
function get_correction end
function correct_gradient! end

"""
    to_prior_scale(xpetab, target::PEtabLogDensity)::AbstractVector

Transforms parameter `xpetab` from the PEtab problem scale to the prior scale.

This conversion is essential for Bayesian inference, as in PEtab.jl Bayesian inference
is performed on the prior scale.

!!! note
    To use this function Bijectors, LogDensityProblems, LogDensityProblemsAD must be loaded;
    `using Bijectors, LogDensityProblems, LogDensityProblemsAD`
"""
function to_prior_scale end

"""
    to_chains(res, target::PEtabLogDensity; start_time=nothing, end_time=nothing)::MCMCChains

Converts Bayesian inference results obtained with `PEtabLogDensity` into a `MCMCChains`.

`res` can be the inference results from AdvancedHMC.jl, AdaptiveMCMC.jl, or Pigeon.jl.
The out chain has the inferred parameters on the prior scale.

# Keyword Arguments
- `start_time`: Optional starting time for the inference, obtained with `now()`.
- `end_time`: Optional ending time for the inference, obtained with `now()`.

!!! note
    To use this function MCMCChains must be loaded; `using MCMCChains`
"""
function to_chains end

if !isdefined(Base, :get_extension)
    include(joinpath(@__DIR__, "..", "ext", "PEtabIpoptExtension.jl"))
    include(joinpath(@__DIR__, "..", "ext", "PEtabOptimExtension.jl"))
    include(joinpath(@__DIR__, "..", "ext", "PEtabPyCallExtension.jl"))
    include(joinpath(@__DIR__, "..", "ext", "PEtabOptimizationExtension.jl"))
    include(joinpath(@__DIR__, "..", "ext", "PEtabSciMLSensitivityExtension.jl"))
    include(joinpath(@__DIR__, "..", "ext", "PEtabLogDensityProblemsExtension.jl"))
    include(joinpath(@__DIR__, "..", "ext", "PEtabPlotsExtension.jl"))
    include(joinpath(@__DIR__, "..", "ext", "PEtabPigeonsExtension.jl"))
end

export to_chains, to_prior_scale

end
