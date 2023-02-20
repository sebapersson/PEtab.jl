#=
    Check the accruacy of the PeTab importer by checking the log-likelihood value against known values for several
    published models. Also check gradients for selected models using FiniteDifferences package
=#


using PEtab
using Test
using OrdinaryDiffEq
using SciMLSensitivity
using CSV
using DataFrames
using ForwardDiff
using LinearAlgebra
using FiniteDifferences
using Sundials


# Used to test cost-value at the nominal parameter value
function testLogLikelihoodValue(petabModel::PEtabModel,
                                referenceValue::Float64,
                                solver; absTol=1e-12, relTol=1e-12, atol=1e-3)

    petabProblem1 = setUpPEtabODEProblem(petabModel, solver, solverAbsTol=absTol, solverRelTol=relTol, costMethod=:Standard)
    petabProblem2 = setUpPEtabODEProblem(petabModel, solver, solverAbsTol=absTol, solverRelTol=relTol, costMethod=:Zygote)
    cost = petabProblem1.computeCost(petabProblem1.θ_nominalT)
    costZygote = petabProblem2.computeCost(petabProblem1.θ_nominalT)
    println("Model : ", petabModel.modelName)
    @test cost ≈ referenceValue atol=atol
    @test costZygote ≈ referenceValue atol=atol
end


function testGradientFiniteDifferences(petabModel::PEtabModel, solver, tol::Float64;
                                       solverForwardEq=CVODE_BDF(),
                                       checkForwardEquations::Bool=false,
                                       checkAdjoint::Bool=false,
                                       testTol::Float64=1e-3,
                                       sensealgSS=SteadyStateAdjoint(),
                                       sensealgAdjoint=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                                       solverSSRelTol=1e-8, solverSSAbsTol=1e-10,
                                       onlyCheckAutoDiff::Bool=false,
                                       splitOverConditions=false)

    # Testing the gradient via finite differences
    petabProblem1 = setUpPEtabODEProblem(petabModel, solver, solverAbsTol=tol, solverRelTol=tol,
                                         splitOverConditions=splitOverConditions, 
                                         solverSSRelTol=solverSSRelTol, solverSSAbsTol=solverSSAbsTol)
    θ_use = petabProblem1.θ_nominalT
    gradientFinite = FiniteDifferences.grad(central_fdm(5, 1), petabProblem1.computeCost, θ_use)[1]
    gradientForward = zeros(length(θ_use))
    petabProblem1.computeGradient!(gradientForward, θ_use)
    @test norm(gradientFinite - gradientForward) ≤ testTol
    
    if checkForwardEquations == true
        petabProblem1 = setUpPEtabODEProblem(petabModel, solver, solverAbsTol=tol, solverRelTol=tol,
                                             gradientMethod=:ForwardEquations, sensealgForwardEquations=:ForwardDiff,
                                             odeSolverForwardEquations=solver, splitOverConditions=splitOverConditions, 
                                             solverSSRelTol=solverSSRelTol, solverSSAbsTol=solverSSAbsTol)
        gradientForwardEquations1 = zeros(length(θ_use))
        petabProblem1.computeGradient!(gradientForwardEquations1, θ_use)
        @test norm(gradientFinite - gradientForwardEquations1) ≤ testTol
        
        if onlyCheckAutoDiff == false
            petabProblem2 = setUpPEtabODEProblem(petabModel, solver, solverAbsTol=tol, solverRelTol=tol,
                                                 gradientMethod=:ForwardEquations, sensealgForwardEquations=ForwardSensitivity(),
                                                 odeSolverForwardEquations=solverForwardEq, splitOverConditions=splitOverConditions, 
                                                 solverSSRelTol=solverSSRelTol, solverSSAbsTol=solverSSAbsTol)
            gradientForwardEquations2 = zeros(length(θ_use))
            petabProblem2.computeGradient!(gradientForwardEquations2, θ_use)
            @test norm(gradientFinite - gradientForwardEquations2) ≤ testTol
        end
    end

    if checkAdjoint == true
        petabProblem1 = setUpPEtabODEProblem(petabModel, solver, solverAbsTol=tol, solverRelTol=tol,
                                             gradientMethod=:Adjoint, sensealgAdjoint=sensealgAdjoint, sensealgAdjointSS=sensealgSS,
                                             solverAdjointAbsTol=tol, solverAdjointRelTol=tol, odeSolverAdjoint=solver,
                                             splitOverConditions=splitOverConditions, 
                                             solverSSRelTol=solverSSRelTol, solverSSAbsTol=solverSSAbsTol)
        gradientAdjoint = zeros(length(θ_use))
        petabProblem1.computeGradient!(gradientAdjoint, θ_use)
        @test norm(gradientFinite - gradientAdjoint) ≤ testTol
    end
end


# Bachman model
pathYML = joinpath(@__DIR__, "Test_ll", "Bachmann_MSB2011", "Bachmann_MSB2011.yaml")
petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
testLogLikelihoodValue(petabModel, -418.40573341425295, Rodas4P())
testGradientFiniteDifferences(petabModel, Rodas5(), 1e-8, checkForwardEquations=true)

# Beer model - Numerically challenging gradient as we have callback rootfinding
pathYML = joinpath(@__DIR__, "Test_ll", "Beer_MolBioSystems2014", "Beer_MolBioSystems2014.yaml")
petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
testLogLikelihoodValue(petabModel, -58622.9145631413, Rodas4P())
testGradientFiniteDifferences(petabModel, Rodas4P(), 1e-8, testTol=1e-1, onlyCheckAutoDiff=true, checkForwardEquations=true, splitOverConditions=true)

# Boehm model
pathYML = joinpath(@__DIR__, "Test_ll", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml")
petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
testLogLikelihoodValue(petabModel, 138.22199693517703, Rodas4P())

# Brännmark model. Model has pre-equlibration criteria so here we test all gradients. Challenging to compute gradients.
pathYML = joinpath(@__DIR__, "Test_ll", "Brannmark_JBC2010", "Brannmark_JBC2010.yaml")
petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
testLogLikelihoodValue(petabModel, 141.889113770537, Rodas4P())
testGradientFiniteDifferences(petabModel, Rodas5(), 1e-8, onlyCheckAutoDiff=true, checkForwardEquations=true, testTol=2e-3)

# Bruno model
pathYML = joinpath(@__DIR__, "Test_ll", "Bruno_JExpBot2016", "Bruno_JExpBot2016.yaml")
petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
testLogLikelihoodValue(petabModel, -46.688176988431806, Rodas4P())

# Crauste model. The model is numerically challanging and computing a gradient via Finite-differences is not possible
pathYML = joinpath(@__DIR__, "Test_ll", "Crauste_CellSystems2017", "Crauste_CellSystems2017.yaml")
petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
testLogLikelihoodValue(petabModel, 190.96521897435176, Rodas4P(), atol=1e-2)

# Fujita model. Challangeing to compute accurate gradients
pathYML = joinpath(@__DIR__, "Test_ll", "Fujita_SciSignal2010", "Fujita_SciSignal2010.yaml")
petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
testLogLikelihoodValue(petabModel, -53.08377736998929, Rodas4P())

# Schwen model. Model has priors so here we want to test all gradients
pathYML = joinpath(@__DIR__, "Test_ll", "Schwen_PONE2014", "Schwen_PONE2014.yaml")
petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
testLogLikelihoodValue(petabModel, 943.9992988598723-12.519137073132825, Rodas4P())

# Sneyd model - Test against World problem
function testSneyd()
    pathYML = joinpath(@__DIR__, "Test_ll", "Sneyd_PNAS2002", "Sneyd_PNAS2002.yaml")
    petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
    testLogLikelihoodValue(petabModel, -319.79177818768756, Rodas4P())
end
testSneyd()
