using PEtab
using Test
using OrdinaryDiffEq
using Sundials
using SciMLSensitivity
using CSV
using DataFrames
using ForwardDiff
using LinearAlgebra


include(joinpath(@__DIR__, "Common.jl"))


function compareAgainstPyPestoBoehm(petabModel::PEtabModel, solverOptions)

    dirValues = joinpath(@__DIR__, "Test_ll", "Boehm_JProteomeRes2014")
    paramVals = CSV.read(joinpath(dirValues, "Parameters_PyPesto.csv"), DataFrame)
    paramMat = paramVals[!, Not([:Id, :ratio, :specC17])]

    # Reference value computed in PyPesto
    costPython = (CSV.read(joinpath(dirValues, "Cost_PyPesto.csv"), DataFrame))[!, :Cost]
    gradPythonMat = CSV.read(joinpath(dirValues, "Grad_PyPesto.csv"), DataFrame)
    gradPythonMat = gradPythonMat[!, Not([:Id, :ratio, :specC17])]
    hessPythonMat = CSV.read(joinpath(dirValues, "Hess_PyPesto.csv"), DataFrame)
    hessPythonMatCols = names(hessPythonMat)
    hessFilter=findall( x -> occursin("ratio", x) || occursin("specC17", x), hessPythonMatCols)
    hessPythonMat = hessPythonMat[!, Not(["Id", hessPythonMatCols[hessFilter]...])]
    
    for i in 1:5
        
        p = collect(paramMat[i, :])
        referenceCost = costPython[i]
        referenceGradient = collect(gradPythonMat[i, :])
        referenceHessian = collect(hessPythonMat[i, :])

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
        gradientAdjoint = _testCostGradientOrHessian(petabModel, solverOptions, p, computeGradient=true, gradientMethod=:Adjoint, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false)))
        @test norm(normalize(gradientAdjoint) - normalize((referenceGradient))) ≤ 1e-2
        gradientForward1 = _testCostGradientOrHessian(petabModel, solverOptions, p, computeGradient=true, gradientMethod=:ForwardEquations, sensealg=:ForwardDiff)
        @test norm(gradientForward1 - referenceGradient) ≤ 1e-2
        gradientForward2 = _testCostGradientOrHessian(petabModel, ODESolverOptions(CVODE_BDF(), abstol=1e-9, reltol=1e-9), p, computeGradient=true, gradientMethod=:ForwardEquations, sensealg=ForwardSensitivity())
        @test norm(gradientForward2 - referenceGradient) ≤ 1e-2

        # Testing "exact" hessian via autodiff 
        hessian = _testCostGradientOrHessian(petabModel, solverOptions, p, computeHessian=true, hessianMethod=:GaussNewton)
        @test norm(hessian[:] - referenceHessian) ≤ 1e-3
    
    end
end


petabModel = readPEtabModel(joinpath(@__DIR__, "Test_ll", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml"), forceBuildJuliaFiles=true)
@testset "Against PyPesto : Boehm" begin 
    compareAgainstPyPestoBoehm(petabModel, ODESolverOptions(Rodas4P(), abstol=1e-9, reltol=1e-9))
end