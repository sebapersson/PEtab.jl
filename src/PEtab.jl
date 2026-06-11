module PEtab

import Catalyst: Catalyst, ReactionSystem, @unpack
import ComponentArrays: ComponentArrays, ComponentVector
import CSV
import DataFrames: DataFrames, DataFrame
import Distributed
import Distributions: Distributions, Distribution, Univariate, Continuous
import ForwardDiff
import HDF5
import LinearAlgebra
import ModelingToolkitBase: ModelingToolkitBase, ODESystem, equations, observables,
    parameters, unknowns, @named, @parameters
import NonlinearSolve
import OrdinaryDiffEqBDF: QNDF
import OrdinaryDiffEqRosenbrock: Rodas5P
import OrdinaryDiffEqSDIRK: KenCarp4
import OrdinaryDiffEqTsit5: Tsit5
import PreallocationTools: DiffCache, get_tmp
import PrecompileTools
import Printf: @sprintf
import QuasiMonteCarlo: QuasiMonteCarlo, LatinHypercubeSample, SamplingAlgorithm
import Random
import ReverseDiff
import RuntimeGeneratedFunctions: RuntimeGeneratedFunctions, @RuntimeGeneratedFunction
import SBMLImporter: SBMLImporter, ModelSBML
import SciMLBase: SciMLBase, AbstractSciMLAlgorithm, CallbackSet, ContinuousCallback,
    DiscreteCallback, ODEProblem, ODESolution, ReturnCode, remake, solve, terminate!
import SciMLLogging
import Setfield: @set
import StatsBase
import StyledStrings: styled, @styled_str
import Sundials: CVODE_Adams, CVODE_BDF
import Symbolics
import Symbolics.SymbolicUtils
import SymbolicIndexingInterface
import YAML

RuntimeGeneratedFunctions.init(@__MODULE__)

const AllowedLogging = Union{
    SciMLLogging.None, SciMLLogging.Minimal, SciMLLogging.Standard,
    SciMLLogging.Detailed, SciMLLogging.All,
}
const ConditionExp = Union{String, Symbol, Pair{String, String}, Pair{Symbol, Symbol}}
const ContDistribution = Distribution{Univariate, Continuous}
const DistInput = Union{Real, Symbolics.Num}
const ModelSystem = Union{ODESystem, ReactionSystem, ODEProblem}
const NonlinearAlg = Union{Nothing, NonlinearSolve.AbstractNonlinearSolveAlgorithm}
const PEtabTables = Union{Dict{Symbol, Union{DataFrame, Dict}}}
const UserFormula = Union{Symbolics.Num, AbstractString, Symbol}

include(joinpath("structs", "petab_model.jl"))
include(joinpath("structs", "petab_odeproblem.jl"))
include(joinpath("structs", "parameter_estimation.jl"))
include(joinpath("structs", "inference.jl"))

const EstimationResult = Union{
    PEtabOptimisationResult, PEtabMultistartResult, Vector{<:Real}, ComponentVector,
}
const UserParameter = Union{PEtabParameter, PEtabMLParameter}

include("common.jl")
include("logging.jl")
include("show.jl")

include(joinpath("petab_files", "callbacks.jl"))
include(joinpath("petab_files", "common.jl"))
include(joinpath("petab_files", "conditions", "check_input.jl"))
include(joinpath("petab_files", "conditions", "common.jl"))
include(joinpath("petab_files", "conditions", "conditions.jl"))
include(joinpath("petab_files", "conditions", "maps.jl"))
include(joinpath("petab_files", "conditions", "ids.jl"))
include(joinpath("petab_files", "conditions", "indices.jl"))
include(joinpath("petab_files", "measurements.jl"))
include(joinpath("petab_files", "observables.jl"))
include(joinpath("petab_files", "parameters.jl"))
include(joinpath("petab_files", "petab_model.jl"))
include(joinpath("petab_files", "read.jl"))
include(joinpath("petab_files", "simulations.jl"))
include(joinpath("petab_files", "table_info.jl"))
include(joinpath("petab_files", "v2_to_v1_tables.jl"))

include(joinpath("julia_input", "events.jl"))
include(joinpath("julia_input", "maps.jl"))
include(joinpath("julia_input", "petab_model.jl"))
include(joinpath("julia_input", "to_tables.jl"))

include(joinpath("nllh_prior", "distributions.jl"))
include(joinpath("nllh_prior", "nllh.jl"))
include(joinpath("nllh_prior", "prior.jl"))

include(joinpath("derivatives", "common.jl"))
include(joinpath("derivatives", "forward_eqs.jl"))
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
include(joinpath("petab_odeproblem", "export.jl"))
include(joinpath("petab_odeproblem", "problem_info.jl"))
include(joinpath("petab_odeproblem", "remake.jl"))
include(joinpath("petab_odeproblem", "ss_solver.jl"))

include(joinpath("parameter_estimation", "multistart.jl"))
include(joinpath("parameter_estimation", "petab_select.jl"))
include(joinpath("parameter_estimation", "plot.jl"))
include(joinpath("parameter_estimation", "profile.jl"))
include(joinpath("parameter_estimation", "singlestart.jl"))
include(joinpath("parameter_estimation", "startguesses.jl"))

include(joinpath("util.jl"))

include(joinpath("ml_models", "array_files.jl"))
include(joinpath("ml_models", "inputs_outputs.jl"))
include(joinpath("ml_models", "model_info.jl"))
include(joinpath("ml_models", "parameters.jl"))
include(joinpath("ml_models", "pre_simulate.jl"))
include(joinpath("ml_models", "sys_ml_calls.jl"))
include(joinpath("ml_models", "templates.jl"))

# Reduce time for reading a PEtabModel and for building a PEtabODEProblem
PrecompileTools.@setup_workload begin
    path_yaml = joinpath(@__DIR__, "..", "test", "analytic_ss", "Test_model3.yaml")
    PrecompileTools.@compile_workload begin
        model = PEtabModel(path_yaml)
        petab_problem = PEtabODEProblem(model, verbose = false)
    end
end

# Functions that only appear in extension
# Bayesian inference
function compute_llh end
function compute_prior end
function get_correction end
function correct_gradient! end
# For ML models
function _get_lux_ps end
function _set_ml_model_ps! end
function _reshape_io_data end
function parse_to_lux end
function _reshape_array end
function ml_ps_to_hdf5 end
function MLModel end
function _get_n_ml_parameters end

export PEtabModel, PEtabODEProblem, ODESolver, SteadyStateSolver, PEtabModel,
    PEtabODEProblem, remake, PEtabOptimisationResult, IpoptOptions, IpoptOptimizer,
    PEtabParameter, PEtabCondition, PEtabObservable, PEtabMultistartResult,
    get_startguesses, get_ps, get_u0, get_odeproblem, get_odesol, get_system, PEtabEvent,
    PEtabLogDensity, solve_all_conditions, get_x, calibrate, calibrate_multistart,
    petab_select, get_obs_comparison_plots, export_petab, LogLaplace, MLModel, MLModels,
    UDEProblem, OptimisersOptions

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

export to_chains, to_prior_scale, PEtabMLParameter

end
