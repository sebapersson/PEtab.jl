#=
    Check the accruacy of the PeTab importer for a simple linear ODE;
        s' = alpha*s; s(0) = 8.0 -> s(t) = 8.0 * exp(alpha*t)
        d' = beta*d;  d(0) = 4.0 -> d(t) = 4.0 * exp(beta*t)
    This ODE is solved analytically, and using the analytical solution the accuracy of
    the ODE solver, cost function, gradient and hessian of the PeTab importer is checked.
    The accuracy of the optimizers is further checked.
    The measurment data is avaible in tests/Test_model2/
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
import PEtab: _changeExperimentalCondition!, solveODEAllExperimentalConditions


include(joinpath(@__DIR__, "Common.jl"))


"""
    testOdeSol(solver, tol; printRes=false)
    Compare analytical vs numeric ODE solver using a provided solver with
    tolerance tol for the Test_model2.
    Returns true if passes test (sqDiff less than 1e-8) else returns false.
"""
function testODESolverTestModel2(petabModel::PEtabModel, solver, tol)

    # Set values to PeTab file values
    experimentalConditionsFile, measurementDataFile, parameterDataFile, observablesDataFile = readPEtabFiles(petabModel)
    measurementData = processMeasurements(measurementDataFile, observablesDataFile)
    paramData = processParameters(parameterDataFile)
    θ_indices = computeIndicesθ(paramData, measurementData, petabModel.odeSystem, experimentalConditionsFile)
    simulationInfo = processSimulationInfo(petabModel, measurementData, paramData)
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
        θ_est = getFileODEvalues(petabModel)
        changeExperimentalCondition! = (pVec, u0Vec, expID) -> _changeExperimentalCondition!(pVec, u0Vec, expID, θ_est, petabModel, θ_indices)

        # Solve ODE system
        odeSolutions, success = solveODEAllExperimentalConditions(prob, changeExperimentalCondition!, simulationInfo, solver, tol, tol, petabModel.computeTStops)
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
function testCostGradientOrHessianTestModel2(petabModel::PEtabModel, solver, tol)

    petabProblem1 = setUpPEtabODEProblem(petabModel, solver, solverAbsTol=tol, solverRelTol=tol,
                                         sensealgZygote = ForwardDiffSensitivity(),
                                         odeSolverForwardEquations=Vern9(), sensealgForwardEquations = ForwardDiffSensitivity(),
                                         odeSolverAdjoint=solver, solverAdjointAbsTol=tol, solverAdjointRelTol=tol,
                                         sensealgAdjoint=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

    petabProblem2 = setUpPEtabODEProblem(petabModel, solver, solverAbsTol=tol, solverRelTol=tol,
                                         sensealgForwardEquations=:AutoDiffForward, odeSolverForwardEquations=Vern9())

    # Cube with random parameter values for testing
    cube = Matrix(CSV.read(petabProblem1.pathCube, DataFrame))

    for i in 1:5

        p = cube[i, :]
        referenceCost = computeCostAnalyticTestModel2(p)
        referenceGradient = ForwardDiff.gradient(computeCostAnalyticTestModel2, p)
        referenceHessian = ForwardDiff.hessian(computeCostAnalyticTestModel2, p)

        # Test both the standard and Zygote approach to compute the cost
        cost = _testCostGradientOrHessian(petabProblem1, p, cost=true)
        @test cost ≈ referenceCost atol=1e-3
        costZygote = _testCostGradientOrHessian(petabProblem1, p, costZygote=true)
        @test costZygote ≈ referenceCost atol=1e-3

        # Test all gradient combinations. Note we test sensitivity equations with and without autodiff
        gradientAutoDiff = _testCostGradientOrHessian(petabProblem1, p, gradientAutoDiff=true)
        @test norm(gradientAutoDiff - referenceGradient) ≤ 1e-2
        gradientZygote = _testCostGradientOrHessian(petabProblem1, p, gradientZygote=true)
        @test norm(gradientZygote - referenceGradient) ≤ 1e-2
        gradientAdjoint = _testCostGradientOrHessian(petabProblem1, p, gradientAdjoint=true)
        @test norm(normalize(gradientAdjoint) - normalize((referenceGradient))) ≤ 1e-2
        gradientForwardEquations1 = _testCostGradientOrHessian(petabProblem1, p, gradientForwardEquations=true)
        @test norm(gradientForwardEquations1 - referenceGradient) ≤ 1e-2
        gradientForwardEquations2 = _testCostGradientOrHessian(petabProblem2, p, gradientForwardEquations=true)
        @test norm(gradientForwardEquations2 - referenceGradient) ≤ 1e-2

        # Testing "exact" hessian via autodiff
        hessian = _testCostGradientOrHessian(petabProblem1, p, hessian=true)
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
    testODESolverTestModel2(petabModel, Vern9(), 1e-9)
end

@testset "Cost gradient and hessian" begin
    testCostGradientOrHessianTestModel2(petabModel, Vern9(), 1e-15)
end

@testset "Gradient of residuals" begin
    checkGradientResiduals(petabModel, Rodas5(), 1e-9)
end
