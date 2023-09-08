module PEtab

using ModelingToolkit
using CSV
using SciMLBase
using OrdinaryDiffEq
using DiffEqCallbacks
using SteadyStateDiffEq
using ForwardDiff
using ReverseDiff
import ChainRulesCore
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
using SBML
using SpecialFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

include("PEtab_structs.jl")

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
include(joinpath("Create_PEtabODEProblem", "Set_defaults.jl"))
include(joinpath("Create_PEtabODEProblem", "Remake_PEtabODEProblem.jl"))
include(joinpath("Create_PEtabODEProblem", "Create_PEtab_ODEProblem.jl"))

# Creating the PEtab model
include("Create_PEtab_model.jl")

# Importing SBML models
include(joinpath("SBML", "SBML_to_ModellingToolkit.jl"))
include(joinpath("SBML", "Common.jl"))
include(joinpath("SBML", "Process_functions.jl"))
include(joinpath("SBML", "Process_rules.jl"))
include(joinpath("SBML", "Solve_SBML_model.jl"))

# For correct struct printing
include(joinpath("Show.jl"))


# Reduce time for reading a PEtabModel and for building a PEtabODEProblem
@setup_workload begin
    pathYAML = joinpath(@__DIR__, "..", "test", "Test_model3", "Test_model3.yaml")
    @compile_workload begin
        petabModel = readPEtabModel(pathYAML, verbose=false, forceBuildJuliaFiles=true, writeToFile=false)
        petabProblem = createPEtabODEProblem(petabModel, verbose=false)
        petabProblem.computeCost(petabProblem.θ_nominalT)
    end
end


export PEtabModel, PEtabODEProblem, ODESolverOptions, SteadyStateSolverOptions, readPEtabModel, createPEtabODEProblem, remakePEtabProblem, Fides, solveSBMLModel, PEtabOptimisationResult, IpoptOptions, IpoptOptimiser


# To make these extension functions exportable, and to allow docstrings to be generated for them.
function createOptimProblem end
function createFidesProblem end 
"""
    calibrateModel(petabProblem::PEtabODEProblem,
                   optimizer;
                   <keyword arguments>)

Perform multi-start local optimization for a given PEtabODEProblem and return (fmin, minimizer) for all runs.

# Arguments
- `petabProblem::PEtabODEProblem`: The PEtabODEProblem to be calibrated.
- `optimizer`: The optimizer algorithm to be used. Currently, we support three different algorithms:
    1. `Fides()`: The Newton trust-region Fides optimizer from Python. Please refer to the documentation for setup
        examples. This optimizer performs well when computing the full Hessian is not possible, and the Gauss-Newton Hessian approximation can be used.
    2. `IPNewton()`: The interior-point Newton method from Optim.jl. This optimizer performs well when it is
        computationally feasible to compute the full Hessian.
    3. `LBFGS()` or `BFGS()` from Optim.jl: These optimizers are suitable when the computation of the Gauss-Newton
        Hessian approximation is too expensive, such as when adjoint sensitivity analysis is required for the gradient.
- `nOptimisationStarts::Int`: Number of multi-starts to be performed. Defaults to 100.
- `samplingMethod`: Method for generating start guesses. Any method from QuasiMonteCarlo.jl is supported, with LatinHypercube as the default.
- `options`: Optimization options. For Optim.jl optimizers, it accepts an `Optim.Options` struct. For Fides, please refer to the Fides documentation and the PEtab.jl documentation for information on setting options.
"""
function callibrateModel end
"""
    runPEtabSelect(pathYAML, optimizer; <keyword arguments>)

Given a PEtab-select YAML file perform model selection with the algorithms specified in the YAML file.

Results are written to a YAML file in the same directory as the PEtab-select YAML file.

Each candidate model produced during the model selection undergoes parameter estimation using local multi-start
optimization. Three optimizers are supported: `optimizer=Fides()` (Fides Newton-trust region), `optimizer=IPNewton()`
from Optim.jl, and `optimizer=LBFGS()` from Optim.jl. Additional keywords for the optimisation are
`nOptimisationStarts::Int`- number of multi-starts for parameter estimation (defaults to 100) and
`optimizationSamplingMethod` - which is any sampling method from QuasiMonteCarlo.jl for generating start guesses
(defaults to LatinHypercubeSample). See also (add callibrate model)

Simulation options can be set using any keyword argument accepted by the `createPEtabODEProblem` function.
For example, setting `gradientMethod=:ForwardDiff` specifies the use of forward-mode automatic differentiation for
gradient computation. If left blank, we automatically select appropriate options based on the size of the problem.
"""
function runPEtabSelect end


if !isdefined(Base, :get_extension)
    include(joinpath(@__DIR__, "..", "ext", "ParameterEstimationExtension.jl"))
end
if !isdefined(Base, :get_extension)    
    include(joinpath(@__DIR__, "..", "ext", "SciMLSensitivityExtension.jl"))
    include(joinpath(@__DIR__, "..", "ext", "ZygoteExtension.jl"))
end


export createOptimProblem, createFidesProblem, callibrateModel, runPEtabSelect


end
