using PyCall
using ModelingToolkit
using DataFrames
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
using OrdinaryDiffEq
using Sundials # For CVODE_BDF
using SciMLSensitivity
using Printf
using Test

RuntimeGeneratedFunctions.init(@__MODULE__)
cd("/home/damiano/dam/PEtab.jl")

include(pwd() * "/src/PEtab_structs.jl")

include(pwd() * "/src/Common.jl")

# Files related to computing the cost (likelihood)
include(pwd() * joinpath("/src", "Compute_cost", "Compute_priors.jl"))
include(pwd() * joinpath("/src", "Compute_cost", "Compute_cost.jl"))
include(pwd() * joinpath("/src", "Compute_cost", "Compute_cost_zygote.jl"))

# Files related to computing derivatives
include(pwd() * joinpath("/src", "Derivatives", "Hessian.jl"))
include(pwd() * joinpath("/src", "Derivatives", "Gradient.jl"))
include(pwd() * joinpath("/src", "Derivatives", "Adjoint_sensitivity_analysis.jl"))
include(pwd() * joinpath("/src", "Derivatives", "Forward_sensitivity_equations.jl"))
include(pwd() * joinpath("/src", "Derivatives", "Gauss_newton.jl"))
include(pwd() * joinpath("/src", "Derivatives", "Common.jl"))

# Files related to solving the ODE-system
include(pwd() * joinpath("/src", "Solve_ODE", "Change_experimental_condition.jl"))
include(pwd() * joinpath("/src", "Solve_ODE", "Solve_ode_Zygote.jl"))
include(pwd() * joinpath("/src", "Solve_ODE", "Solve_ode_model.jl"))
include(pwd() * joinpath("/src", "Solve_ODE", "Solve_for_steady_state.jl"))

# Files related to distributed computing
include(pwd() * joinpath("/src", "Distributed", "Distributed.jl"))

# Files related to processing PEtab files
include(pwd() * joinpath("/src", "Process_PEtab_files", "Common.jl"))
include(pwd() * joinpath("/src", "Process_PEtab_files", "Get_simulation_info.jl"))
include(pwd() * joinpath("/src", "Process_PEtab_files", "Get_parameter_indices.jl"))
include(pwd() * joinpath("/src", "Process_PEtab_files", "Process_measurements.jl"))
include(pwd() * joinpath("/src", "Process_PEtab_files", "Process_parameters.jl"))
include(pwd() * joinpath("/src", "Process_PEtab_files", "Process_callbacks.jl"))
include(pwd() * joinpath("/src", "Process_PEtab_files", "Observables", "Common.jl"))
include(pwd() * joinpath("/src", "Process_PEtab_files", "Observables", "Create_h_sigma_derivatives.jl"))
include(pwd() * joinpath("/src", "Process_PEtab_files", "Observables", "Create_u0_h_sigma.jl"))
include(pwd() * joinpath("/src", "Process_PEtab_files", "Read_PEtab_files.jl"))

# For creating a PEtab ODE problem
include(pwd() * joinpath("/src", "Create_PEtab_ODEProblem.jl"))

# Creating the PEtab model
include(pwd() * "/src/Create_PEtab_model.jl")

# Importing SBML models 
include(pwd() * joinpath("/src", "SBML", "SBML_to_ModellingToolkit.jl"))
include(pwd() * joinpath("/src", "SBML", "Common.jl"))
include(pwd() * joinpath("/src", "SBML", "Process_functions.jl"))
include(pwd() * joinpath("/src", "SBML", "Process_rules.jl"))

# Used to test cost-value at the nominal parameter value 
function testLogLikelihoodValue(petabModel::PEtabModel, 
                                referenceValue::Float64; atol=1e-3)
    
    odeSolverOptions = getODESolverOptions(Rodas4P(), abstol=1e-8, reltol=1e-8)
    petabProblem = setupPEtabODEProblem(petabModel, odeSolverOptions, 
                                gradientMethod=:ForwardDiff, 
                                hessianMethod=:ForwardDiff, 
                                splitOverConditions=true)

    cost = petabProblem.computeCost(petabProblem.θ_nominalT)
    println("Model : ", petabModel.modelName)
    @test cost ≈ referenceValue atol=atol    

end

@testset "Log likelihood values and gradients for benchmark collection" begin
    
    # Beer model - Numerically challenging gradient as we have callback rootfinding
    pathYML = joinpath(@__DIR__, "JuliaImport", "Beer", "Beer_MolBioSystems2014.yaml")
    petabModel = readPEtabModel(pathYML, verbose=false, jlFile=true)     
    testLogLikelihoodValue(petabModel, -58622.9145631413)
    
    # Boehm model 
    pathYML = joinpath(@__DIR__, "JuliaImport", "Boehm", "Boehm_JProteomeRes2014.yaml")
    petabModel = readPEtabModel(pathYML, verbose=false, jlFile=true)     
    testLogLikelihoodValue(petabModel, 138.22199693517703)
    
    # Brännmark model. Model has pre-equlibration criteria so here we test all gradients. Challenging to compute gradients.
    pathYML = joinpath(@__DIR__, "JuliaImport", "Brannmark", "Brannmark_JBC2010.yaml")
    petabModel = readPEtabModel(pathYML, verbose=false, jlFile=true)
    testLogLikelihoodValue(petabModel, 141.889113770537)
    
    # Fujita model. Challangeing to compute accurate gradients  
    pathYML = joinpath(@__DIR__, "JuliaImport", "Fujita", "Fujita_SciSignal2010.yaml")
    petabModel = readPEtabModel(pathYML, verbose=false, jlFile=true)
    testLogLikelihoodValue(petabModel, -53.08377736998929)
    
    # Isensee model. Accurate gradients are computed (but the code takes ages to run with low tolerances)
    pathYML = joinpath(@__DIR__, "JuliaImport", "Isensee", "Isensee_JCB2018.yaml")
    petabModel = readPEtabModel(pathYML, verbose=false, jlFile=true)
    testLogLikelihoodValue(petabModel, 3949.375966548649-4.45299970460275, atol=1e-2)
    
    # Weber model. Challanging as it sensitivity to steady state tolerances 
    pathYML = joinpath(@__DIR__, "JuliaImport", "Weber", "Weber_BMC2015.yaml")
    petabModel = readPEtabModel(pathYML, verbose=false, jlFile=true)
    testLogLikelihoodValue(petabModel, 296.2017922646865)

end
