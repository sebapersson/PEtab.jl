module PEtab

using ModelingToolkit
using CSV
using SciMLBase
using OrdinaryDiffEqBDF
using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqSDIRK
using Catalyst
using ComponentArrays
using DataFrames
using ForwardDiff
using ReverseDiff
using DiffEqCallbacks
using Distributed
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
import SciMLBase.remake
import QuasiMonteCarlo: LatinHypercubeSample, SamplingAlgorithm

RuntimeGeneratedFunctions.init(@__MODULE__)

const ModelSystem = Union{ODESystem, ReactionSystem}
const NonlinearAlg = Union{Nothing, NonlinearSolve.AbstractNonlinearSolveAlgorithm}

include(joinpath("structs", "petab_model.jl"))
include(joinpath("structs", "petab_odeproblem.jl"))
include(joinpath("structs", "parameter_estimation.jl"))
include(joinpath("structs", "inference.jl"))

const EstimationResult = Union{PEtabOptimisationResult, PEtabMultistartResult,
                               Vector{<:AbstractFloat}, ComponentArray}

include("common.jl")
include("logging.jl")
include("show.jl")

include(joinpath("petab_files", "common.jl"))
include(joinpath("petab_files", "conditions.jl"))
include(joinpath("petab_files", "measurements.jl"))
include(joinpath("petab_files", "observables.jl"))
include(joinpath("petab_files", "parameters.jl"))
include(joinpath("petab_files", "petab_model.jl"))
include(joinpath("petab_files", "read.jl"))
include(joinpath("petab_files", "simulations.jl"))
include(joinpath("petab_files", "table_info.jl"))

include(joinpath("julia_input", "events.jl"))
include(joinpath("julia_input", "maps.jl"))
include(joinpath("julia_input", "petab_model.jl"))
include(joinpath("julia_input", "to_tables.jl"))

include(joinpath("nllh_prior", "nllh.jl"))
include(joinpath("nllh_prior", "prior.jl"))

include(joinpath("derivatives", "common.jl"))
include(joinpath("derivatives", "forward_eqs.jl"))
include(joinpath("derivatives", "forward_ad_chunks.jl"))
include(joinpath("derivatives", "gauss_newton.jl"))
include(joinpath("derivatives", "gradient.jl"))
include(joinpath("derivatives", "hessian.jl"))

include(joinpath("solve", "helper.jl"))
include(joinpath("solve", "solve.jl"))
include(joinpath("solve", "steady_state.jl"))

include(joinpath("petab_odeproblem", "cache.jl"))
include(joinpath("petab_odeproblem", "create.jl"))
include(joinpath("petab_odeproblem", "defaults.jl"))
include(joinpath("petab_odeproblem", "derivative_functions.jl"))
include(joinpath("petab_odeproblem", "problem_info.jl"))
include(joinpath("petab_odeproblem", "remake.jl"))
include(joinpath("petab_odeproblem", "ss_solver.jl"))

include(joinpath("parameter_estimation", "multistart.jl"))
include(joinpath("parameter_estimation", "petab_select.jl"))
include(joinpath("parameter_estimation", "plot.jl"))
include(joinpath("parameter_estimation", "singlestart.jl"))
include(joinpath("parameter_estimation", "startguesses.jl"))

include(joinpath("util.jl"))
#=
# Reduce time for reading a PEtabModel and for building a PEtabODEProblem
@setup_workload begin
    path_yaml = joinpath(@__DIR__, "..", "test", "analytic_ss", "Test_model3.yaml")
    @compile_workload begin
        model = PEtabModel(path_yaml, verbose = false, build_julia_files = true,
                           write_to_file = false)
        petab_problem = PEtabODEProblem(model, verbose = false)
        petab_problem.nllh(petab_problem.xnominal_transformed)
    end
end
=#

# Functions that only appear in extension
function compute_llh end
function compute_prior end
function get_correction end
function correct_gradient! end

export PEtabModel, PEtabODEProblem, ODESolver, SteadyStateSolver, PEtabModel,
       PEtabODEProblem, remake, Fides, PEtabOptimisationResult, IpoptOptions,
       IpoptOptimizer, PEtabParameter, PEtabObservable, PEtabMultistartResult,
       get_startguesses, get_ps, get_u0, get_odeproblem, get_odesol, get_system, PEtabEvent,
       PEtabLogDensity, solve_all_conditions, get_x, calibrate, calibrate_multistart,
       petab_select, get_obs_comparison_plots

"""
    to_prior_scale(xpetab, target::PEtabLogDensity)

Transforms parameter `x` from the PEtab problem scale to the prior scale.

This conversion is needed for Bayesian inference, as in PEtab.jl Bayesian inference is
performed on the prior scale.

!!! note
    To use this function, the Bijectors, LogDensityProblems, and LogDensityProblemsAD
    packages must be loaded: `using Bijectors, LogDensityProblems, LogDensityProblemsAD`
"""
function to_prior_scale end

"""
    to_chains(res, target::PEtabLogDensity; kwargs...)::MCMCChains

Converts Bayesian inference results obtained with `PEtabLogDensity` into an `MCMCChains`.

`res` can be the inference results from AdvancedHMC.jl or AdaptiveMCMC.jl. The returned
chain has the parameters on the prior scale.

# Keyword Arguments
- `start_time`: Optional starting time for the inference, obtained with `now()`.
- `end_time`: Optional ending time for the inference, obtained with `now()`.

!!! note
    To use this function, the MCMCChains package must be loaded: `using MCMCChains`
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
end

export to_chains, to_prior_scale

end
