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


function compare_Boehm_pyPESTO(petab_model::PEtabModel, ode_solver::ODESolver)

    dir_values = joinpath(@__DIR__, "Test_ll", "Boehm_JProteomeRes2014")
    paramVals = CSV.File(joinpath(dir_values, "Parameters_PyPesto.csv"), drop=[:Id, :ratio, :specC17])
    paramMat = paramVals

    # Reference value computed in PyPesto
    cost_pyPESTO = CSV.File(joinpath(dir_values, "Cost_PyPesto.csv"))[:Cost]
    grad_pyPESTO = CSV.File(joinpath(dir_values, "Grad_PyPesto.csv"), drop=[:Id, :ratio, :specC17])
    hess_pyPESTO = CSV.File(joinpath(dir_values, "Hess_PyPesto.csv"))
    hess_pyPESTO_cols = string.(hess_pyPESTO.names)
    hessFilter=findall( x -> x != "Id" && !occursin("ratio", x) && !occursin("specC17", x), hess_pyPESTO_cols)

    for i in 1:5

        p = Float64.(collect(paramMat[i]))
        reference_cost = Float64.(collect(cost_pyPESTO[i]))
        reference_gradient = Float64.(collect(grad_pyPESTO[i]))
        reference_hessian = Float64.(collect(hess_pyPESTO[i])[hessFilter])

        # Test both the standard and Zygote approach to compute the cost
        cost = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_cost=true, cost_method=:Standard)
        @test cost ≈ reference_cost atol=1e-3
        cost_zygote = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_cost=true, cost_method=:Zygote)
        @test cost_zygote ≈ reference_cost atol=1e-3

        # Test all gradient combinations. Note we test sensitivity equations with and without autodiff
        gradient_forwarddiff = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_gradient=true, gradient_method=:ForwardDiff)
        @test norm(gradient_forwarddiff - reference_gradient) ≤ 1e-2
        gradient_zygote = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_gradient=true, gradient_method=:Zygote, sensealg=ForwardDiffSensitivity())
        @test norm(gradient_zygote - reference_gradient) ≤ 1e-2
        gradient_adjoint = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_gradient=true, gradient_method=:Adjoint, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false)))
        @test norm(normalize(gradient_adjoint) - normalize((reference_gradient))) ≤ 1e-2
        gradient_forward1 = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_gradient=true, gradient_method=:ForwardEquations, sensealg=:ForwardDiff)
        @test norm(gradient_forward1 - reference_gradient) ≤ 1e-2
        gradient_forward2 = _test_cost_gradient_hessian(petab_model, ODESolver(CVODE_BDF(), abstol=1e-9, reltol=1e-9), p, compute_gradient=true, gradient_method=:ForwardEquations, sensealg=ForwardSensitivity())
        @test norm(gradient_forward2 - reference_gradient) ≤ 1e-2

        # Testing "exact" hessian via autodiff
        hessian = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_hessian=true, hessian_method=:GaussNewton)
        @test norm(hessian[:] - reference_hessian) ≤ 1e-3
    end
end

path_yaml = joinpath(@__DIR__, "Test_ll", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml")
petab_model = PEtabModel(path_yaml, verbose=false, build_julia_files=true, write_to_file=false)
@testset "Against PyPesto : Boehm" begin
    compare_Boehm_pyPESTO(petab_model, ODESolver(Rodas5P(), abstol=1e-9, reltol=1e-9))
end
