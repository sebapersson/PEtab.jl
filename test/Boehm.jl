using PEtab
using Test
using OrdinaryDiffEq
using Sundials
using Zygote
using SciMLSensitivity
using CSV
using ForwardDiff
using LinearAlgebra


include(joinpath(@__DIR__, "Common.jl"))


function compareAgainstPyPestoBoehm(petab_model::PEtabModel, solverOptions)

    dirValues = joinpath(@__DIR__, "Test_ll", "Boehm_JProteomeRes2014")
    paramVals = CSV.File(joinpath(dirValues, "Parameters_PyPesto.csv"), drop=[:Id, :ratio, :specC17])
    paramMat = paramVals

    # Reference value computed in PyPesto
    costPython = CSV.File(joinpath(dirValues, "Cost_PyPesto.csv"))[:Cost]
    gradPythonMat = CSV.File(joinpath(dirValues, "Grad_PyPesto.csv"), drop=[:Id, :ratio, :specC17])
    hessPythonMat = CSV.File(joinpath(dirValues, "Hess_PyPesto.csv"))
    hessPythonMatCols = string.(hessPythonMat.names)
    hessFilter=findall( x -> x != "Id" && !occursin("ratio", x) && !occursin("specC17", x), hessPythonMatCols)

    for i in 1:5

        p = Float64.(collect(paramMat[i]))
        referenceCost = Float64.(collect(costPython[i]))
        referenceGradient = Float64.(collect(gradPythonMat[i]))
        referenceHessian = Float64.(collect(hessPythonMat[i])[hessFilter])

        # Test both the standard and Zygote approach to compute the cost
        cost = _testCostGradientOrHessian(petab_model, solverOptions, p, compute_cost=true, cost_method=:Standard)
        @test cost ≈ referenceCost atol=1e-3
        costZygote = _testCostGradientOrHessian(petab_model, solverOptions, p, compute_cost=true, cost_method=:Zygote)
        @test costZygote ≈ referenceCost atol=1e-3

        # Test all gradient combinations. Note we test sensitivity equations with and without autodiff
        gradientForwardDiff = _testCostGradientOrHessian(petab_model, solverOptions, p, compute_gradient=true, gradient_method=:ForwardDiff)
        @test norm(gradientForwardDiff - referenceGradient) ≤ 1e-2
        gradientZygote = _testCostGradientOrHessian(petab_model, solverOptions, p, compute_gradient=true, gradient_method=:Zygote, sensealg=ForwardDiffSensitivity())
        @test norm(gradientZygote - referenceGradient) ≤ 1e-2
        gradient_adjoint = _testCostGradientOrHessian(petab_model, solverOptions, p, compute_gradient=true, gradient_method=:Adjoint, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false)))
        @test norm(normalize(gradient_adjoint) - normalize((referenceGradient))) ≤ 1e-2
        gradientForward1 = _testCostGradientOrHessian(petab_model, solverOptions, p, compute_gradient=true, gradient_method=:ForwardEquations, sensealg=:ForwardDiff)
        @test norm(gradientForward1 - referenceGradient) ≤ 1e-2
        gradientForward2 = _testCostGradientOrHessian(petab_model, ODESolver(CVODE_BDF(), abstol=1e-9, reltol=1e-9), p, compute_gradient=true, gradient_method=:ForwardEquations, sensealg=ForwardSensitivity())
        @test norm(gradientForward2 - referenceGradient) ≤ 1e-2

        # Testing "exact" hessian via autodiff
        hessian = _testCostGradientOrHessian(petab_model, solverOptions, p, compute_hessian=true, hessian_method=:GaussNewton)
        @test norm(hessian[:] - referenceHessian) ≤ 1e-3

    end
end

path_yaml = joinpath(@__DIR__, "Test_ll", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml")
petab_model = PEtabModel(path_yaml, verbose=false, build_julia_files=true, write_to_file=false)
@testset "Against PyPesto : Boehm" begin
    compareAgainstPyPestoBoehm(petab_model, ODESolver(Rodas5P(), abstol=1e-9, reltol=1e-9))
end
