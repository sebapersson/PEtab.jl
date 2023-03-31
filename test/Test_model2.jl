#=
    Check the accruacy of the PeTab importer for a simple linear ODE;
        s' = alpha*s; s(0) = 8.0 -> s(t) = 8.0 * exp(alpha*t)
        d' = beta*d;  d(0) = 4.0 -> d(t) = 4.0 * exp(beta*t)
    This ODE is solved analytically, and using the analytical solution the accuracy of
    the ODE solver, cost function, gradient and hessian of the PeTab importer is checked.
    The accuracy of the optimizers is further checked.
    The measurment data is avaible in test/Test_model2/
 =#

using PEtab
using Test
using OrdinaryDiffEq
using SciMLSensitivity
using CSV
using DataFrames
using ForwardDiff
using LinearAlgebra

import PEtab: readPEtabFiles, processMeasurements, processParameters, computeIndicesθ, processSimulationInfo, setParamToFileValues!
import PEtab: _changeExperimentalCondition!, solveODEAllExperimentalConditions, PEtabODESolverCache, createPEtabODESolverCache, _getSteadyStateSolverOptions


include(joinpath(@__DIR__, "Common.jl"))


"""
    testOdeSol(solver, tol; printRes=false)
    Compare analytical vs numeric ODE solver using a provided solver with
    tolerance tol for the Test_model2.
    Returns true if passes test (sqDiff less than 1e-8) else returns false.
"""
function testODESolverTestModel2(petabModel::PEtabModel, solverOptions)

    # Set values to PeTab file values
    experimentalConditionsFile, measurementDataFile, parameterDataFile, observablesDataFile = readPEtabFiles(petabModel)
    measurementData = processMeasurements(measurementDataFile, observablesDataFile)
    paramData = processParameters(parameterDataFile)
    θ_indices = computeIndicesθ(paramData, measurementData, petabModel.odeSystem, experimentalConditionsFile)
    simulationInfo = processSimulationInfo(petabModel, measurementData)
    setParamToFileValues!(petabModel.parameterMap, petabModel.stateMap, paramData)

    # Parameter values where to teast accuracy. Each column is a alpha, beta, gamma and delta
    u0 = [8.0, 4.0]
    parametersTest = reshape([2.0, 3.0,
                              1.0, 2.0,
                              1.0, 0.4,
                              4.0, 3.0,
                              0.01, 0.02], (2, 5))

    for i in 1:5

        alpha, beta = parametersTest[:, i]
        # Set parameter values for ODE
        petabModel.parameterMap[2] = Pair(petabModel.parameterMap[2].first, alpha)
        petabModel.parameterMap[3] = Pair(petabModel.parameterMap[3].first, beta)
        prob = ODEProblem(petabModel.odeSystem, petabModel.stateMap, (0.0, 5e3), petabModel.parameterMap, jac=true)
        prob = remake(prob, p = convert.(Float64, prob.p), u0 = convert.(Float64, prob.u0))
        θ_dynamic = getFileODEvalues(petabModel)[1:2]

        ssOptions = getSteadyStateSolverOptions(:Simulate)
        _ssSolverOptions = _getSteadyStateSolverOptions(ssOptions, prob, solverOptions.abstol / 100.0, 
                                                        solverOptions.reltol / 100.0, solverOptions.maxiters)

        # Solve ODE system
        petabODESolverCache = createPEtabODESolverCache(:nothing, :nothing, petabModel, simulationInfo, θ_indices, nothing)
        θ_dynamic = [alpha, beta]
        odeSolutions, success = solveODEAllExperimentalConditions(prob, petabModel, θ_dynamic, petabODESolverCache, simulationInfo, θ_indices, solverOptions, _ssSolverOptions)
        odeSolution = odeSolutions[simulationInfo.experimentalConditionId[1]]

        # Compare against analytical solution
        sqDiff = 0.0
        for t in odeSolution.t
            solAnalytic = [u0[1]*exp(alpha*t), u0[2]*exp(beta*t)]
            sqDiff += sum((odeSolution(t)[1:2] - solAnalytic).^2)
        end

        @test sqDiff ≤ 1e-6
    end
end


function computeCostAnalyticTestModel2(paramVec)

    u0 = [8.0, 4.0]
    alpha, beta = paramVec[1:2]
    measurementData = CSV.read(joinpath(@__DIR__, "Test_model2/measurementData_Test_model2.tsv"), DataFrame)

    # Extract correct parameter for observation i and compute logLik
    logLik = 0.0
    for i in 1:nrow(measurementData)

        # Specs for observation i
        obsID = measurementData[i, :observableId]
        noiseID = measurementData[i, :noiseParameters]
        yObs = measurementData[i, :measurement]
        t = measurementData[i, :time]
        # Extract correct sigma
        if noiseID == "sd_sebastian_new"
            sigma = paramVec[3]
        elseif noiseID == "sd_damiano_new"
            sigma = paramVec[4]
        end

        sol = [u0[1]*exp(alpha*t), u0[2]*exp(beta*t)]
        if obsID == "sebastian_measurement"
            yMod = sol[1]
        elseif obsID == "damiano_measurement"
            yMod = sol[2]
        end

        logLik += log(sigma) + 0.5*log(2*pi) + 0.5 * ((yObs - yMod) / sigma)^2
    end

    return logLik
end


"""
    testCostGradientOrHessianTestModel2(solver, tol; printRes::Bool=false)
    Compare cost, gradient and hessian computed via the analytical solution
    vs the PeTab importer functions (to check PeTab importer) for five random
    parameter vectors for Test_model2. For the analytical solution the gradient
    and hessian are computed via ForwardDiff.
"""
function testCostGradientOrHessianTestModel2(petabModel::PEtabModel, solverOptions)

    # Cube with random parameter values for testing
    cube = Matrix(CSV.read(joinpath(@__DIR__, "Test_model2", "Julia_model_files", "CubeTest_model2.csv") , DataFrame))

    for i in 1:1

        p = cube[i, :]
        referenceCost = computeCostAnalyticTestModel2(p)
        referenceGradient = ForwardDiff.gradient(computeCostAnalyticTestModel2, p)
        referenceHessian = ForwardDiff.hessian(computeCostAnalyticTestModel2, p)

        # Test both the standard and Zygote approach to compute the cost
        cost = _testCostGradientOrHessian(petabModel, solverOptions, p, computeCost=true, costMethod=:Standard)
        @test cost ≈ referenceCost atol=1e-3
        costZygote = _testCostGradientOrHessian(petabModel, solverOptions, p, computeCost=true, costMethod=:Zygote)
        @test costZygote ≈ referenceCost atol=1e-3

        # Test all gradient combinations. Note we test sensitivity equations with and without autodiff
        gradientForwardDiff = _testCostGradientOrHessian(petabModel, solverOptions, p, computeGradient=true, gradientMethod=:ForwardDiff)
        @test norm(gradientForwardDiff - referenceGradient) ≤ 1e-2
        gradientZygote = _testCostGradientOrHessian(petabModel, solverOptions, p, computeGradient=true, gradientMethod=:Zygote, sensealg=ForwardDiffSensitivity())
        @test norm(gradientZygote - referenceGradient) ≤ 1e-2
        gradientAdjoint = _testCostGradientOrHessian(petabModel, solverOptions, p, computeGradient=true, gradientMethod=:Adjoint, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(false)))
        @test norm(normalize(gradientAdjoint) - normalize((referenceGradient))) ≤ 1e-2
        gradientForward1 = _testCostGradientOrHessian(petabModel, solverOptions, p, computeGradient=true, gradientMethod=:ForwardEquations, sensealg=:ForwardDiff)
        @test norm(gradientForward1 - referenceGradient) ≤ 1e-2
        gradientForward2 = _testCostGradientOrHessian(petabModel, solverOptions, p, computeGradient=true, gradientMethod=:ForwardEquations, sensealg=ForwardDiffSensitivity())
        @test norm(gradientForward2 - referenceGradient) ≤ 1e-2

        # Testing "exact" hessian via autodiff
        hessian = _testCostGradientOrHessian(petabModel, solverOptions, p, computeHessian=true, hessianMethod=:ForwardDiff)
        @test norm(hessian - referenceHessian) ≤ 1e-2
    end
end


# Used to check against world-age problem
function createModelInsideFunction()
    _petabModel = readPEtabModel(joinpath(@__DIR__, "Test_model2/Test_model2.yaml"), forceBuildJuliaFiles=false, verbose=true)
    return _petabModel
end
petabModel = createModelInsideFunction()

@testset "ODE solver" begin
    testODESolverTestModel2(petabModel, getODESolverOptions(Vern9(), abstol=1e-9, reltol=1e-9))
end

@testset "Cost gradient and hessian" begin
    testCostGradientOrHessianTestModel2(petabModel, getODESolverOptions(Vern9(), abstol=1e-15, reltol=1e-15))
end

checkGradientResiduals(petabModel, getODESolverOptions(Rodas5P(), abstol=1e-9, reltol=1e-9))
@testset "Gradient of residuals" begin
    checkGradientResiduals(petabModel, getODESolverOptions(Rodas5P(), abstol=1e-9, reltol=1e-9))
end
