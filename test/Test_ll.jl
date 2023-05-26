#=
    Check the accruacy of the PeTab importer by checking the log-likelihood value against known values for several
    published models. Also check gradients for selected models using FiniteDifferences package
=#


using PEtab
using Test
using OrdinaryDiffEq
using SciMLSensitivity
using CSV
using ForwardDiff
using LinearAlgebra
using FiniteDifferences
using Sundials


# Used to test cost-value at the nominal parameter value
function testLogLikelihoodValue(petabModel::PEtabModel,
                                referenceValue::Float64,
                                solverOptions; atol=1e-3,
                                verbose=false, checkZygote=true)

    modelName = petabModel.modelName
    @info "Model : $modelName"
    petabProblem1 = createPEtabODEProblem(petabModel, odeSolverOptions=solverOptions, costMethod=:Standard, verbose=verbose)
    petabProblem2 = createPEtabODEProblem(petabModel, odeSolverOptions=solverOptions, costMethod=:Zygote, verbose=verbose)
    cost = petabProblem1.computeCost(petabProblem1.θ_nominalT)
    @test cost ≈ referenceValue atol = atol

    if checkZygote == true
        costZygote = petabProblem2.computeCost(petabProblem1.θ_nominalT)
        @test costZygote ≈ referenceValue atol = atol
    end
end


function testGradientFiniteDifferences(petabModel::PEtabModel, solverOptions;
                                       solverGradientOptions=nothing,
                                       checkForwardEquations::Bool=false,
                                       checkAdjoint::Bool=false,
                                       testTol::Float64=1e-3,
                                       sensealgSS=SteadyStateAdjoint(),
                                       sensealgAdjoint=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                                       ssOptions=nothing,
                                       onlyCheckAutoDiff::Bool=false,
                                       splitOverConditions=false)

    if isnothing(solverGradientOptions)
        solverGradientOptions = deepcopy(solverOptions)
    end

    # Testing the gradient via finite differences
    petabProblem1 = createPEtabODEProblem(petabModel, odeSolverOptions=solverOptions,
                                          gradientMethod=:ForwardDiff,
                                          splitOverConditions=splitOverConditions,
                                          ssSolverOptions=ssOptions,
                                          verbose=false)
    θ_use = petabProblem1.θ_nominalT
    gradientFinite = FiniteDifferences.grad(central_fdm(5, 1), petabProblem1.computeCost, θ_use)[1]
    gradientForward = zeros(length(θ_use))
    petabProblem1.computeGradient!(gradientForward, θ_use)
    @test norm(gradientFinite - gradientForward) ≤ testTol

    if checkForwardEquations == true
        petabProblem1 = createPEtabODEProblem(petabModel, odeSolverOptions=solverOptions,
                                             gradientMethod=:ForwardEquations, sensealg=:ForwardDiff,
                                             ssSolverOptions=ssOptions,
                                             verbose=false)
        gradientForwardEquations1 = zeros(length(θ_use))
        petabProblem1.computeGradient!(gradientForwardEquations1, θ_use)
        @test norm(gradientFinite - gradientForwardEquations1) ≤ testTol

        if onlyCheckAutoDiff == false
            petabProblem2 = createPEtabODEProblem(petabModel, odeSolverOptions=solverOptions, odeSolverGradientOptions=solverGradientOptions,
                                                 gradientMethod=:ForwardEquations, sensealg=ForwardSensitivity(),
                                                 ssSolverOptions=ssOptions,
                                                 verbose=false)
            gradientForwardEquations2 = zeros(length(θ_use))
            petabProblem2.computeGradient!(gradientForwardEquations2, θ_use)
            @test norm(gradientFinite - gradientForwardEquations2) ≤ testTol
        end
    end

    if checkAdjoint == true
        petabProblem1 = createPEtabODEProblem(petabModel, odeSolverOptions=solverOptions, odeSolverGradientOptions=solverGradientOptions,
                                             gradientMethod=:Adjoint, sensealg=sensealgAdjoint, sensealgSS=sensealgSS,
                                             splitOverConditions=splitOverConditions,
                                             ssSolverOptions=ssOptions,
                                             verbose=false)
        gradientAdjoint = zeros(length(θ_use))
        petabProblem1.computeGradient!(gradientAdjoint, θ_use)
        @test norm(gradientFinite - gradientAdjoint) ≤ testTol
    end
end


# Bachman model
pathYML = joinpath(@__DIR__, "Test_ll", "Bachmann_MSB2011", "Bachmann_MSB2011.yaml")
petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
testLogLikelihoodValue(petabModel, -418.40573341425295, ODESolverOptions(Rodas4P(), abstol=1e-12, reltol=1e-12))
testGradientFiniteDifferences(petabModel, ODESolverOptions(Rodas4P(), abstol=1e-9, reltol=1e-9),
                             solverGradientOptions=ODESolverOptions(CVODE_BDF(), abstol=1e-9, reltol=1e-9),
                             checkForwardEquations=true, checkAdjoint=true, testTol=1e-2)

# Beer model - Numerically challenging gradient as we have callback time triggering parameters to 
# estimate. Splitting over conditions spped up hessian computations with factor 48
pathYML = joinpath(@__DIR__, "Test_ll", "Beer_MolBioSystems2014", "Beer_MolBioSystems2014.yaml")
petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
testLogLikelihoodValue(petabModel, -58622.9145631413, ODESolverOptions(Rodas4P(), abstol=1e-12, reltol=1e-12))
testGradientFiniteDifferences(petabModel, ODESolverOptions(Rodas4P(), abstol=1e-8, reltol=1e-8), testTol=1e-1, onlyCheckAutoDiff=true, checkForwardEquations=true, splitOverConditions=true)

# Brännmark model. Model has pre-equlibration criteria so here we test all gradients. Challenging to compute gradients.
pathYML = joinpath(@__DIR__, "Test_ll", "Brannmark_JBC2010", "Brannmark_JBC2010.yaml")
petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
testLogLikelihoodValue(petabModel, 141.889113770537, ODESolverOptions(Rodas4P(), abstol=1e-12, reltol=1e-12))
testGradientFiniteDifferences(petabModel, ODESolverOptions(Rodas5(), abstol=1e-8, reltol=1e-8), onlyCheckAutoDiff=true, checkForwardEquations=true, testTol=2e-3)

# Crauste model. The model is numerically challanging and computing a gradient via Finite-differences is not possible
pathYML = joinpath(@__DIR__, "Test_ll", "Crauste_CellSystems2017", "Crauste_CellSystems2017.yaml")
petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
testLogLikelihoodValue(petabModel, 190.96521897435176, ODESolverOptions(Rodas4P(), abstol=1e-12, reltol=1e-12), atol=1e-2)

# Fujita model. Challangeing to compute accurate gradients
pathYML = joinpath(@__DIR__, "Test_ll", "Fujita_SciSignal2010", "Fujita_SciSignal2010.yaml")
petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
testLogLikelihoodValue(petabModel, -53.08377736998929, ODESolverOptions(Rodas4P(), abstol=1e-12, reltol=1e-12))

# Iseense - tricky model with pre-eq criteria and priors
pathYML = joinpath(@__DIR__, "Test_ll", "Isensee_JCB2018", "Isensee_JCB2018.yaml")
petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
testLogLikelihoodValue(petabModel, 3949.375966548649 + 4.45299970460275, ODESolverOptions(Rodas4P(), abstol=1e-12, reltol=1e-12), checkZygote=false)

# Sneyd model - Test against World problem by wrapping inside function
function testSneyd()
    pathYML = joinpath(@__DIR__, "Test_ll", "Sneyd_PNAS2002", "Sneyd_PNAS2002.yaml")
    petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
    testLogLikelihoodValue(petabModel, -319.79177818768756, ODESolverOptions(Rodas4P(), abstol=1e-12, reltol=1e-12))
end
testSneyd()
