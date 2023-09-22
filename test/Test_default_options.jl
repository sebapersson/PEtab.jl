#=
    Test that default options (user does not provide ODE-solver and gradient-method)
    is handled correctly.
=#


using PEtab
using Zygote 
using SciMLSensitivity
using OrdinaryDiffEq
using Sundials
using Test


@testset "Test default options" begin
    # Check that we get correct default setting
    path_yaml = joinpath(@__DIR__, "Test_ll", "Bachmann_MSB2011", "Bachmann_MSB2011.yaml")
    petab_model = PEtabModel(path_yaml, verbose=false, build_julia_files=false)
    petab_problem = PEtabODEProblem(petab_model, verbose=false)
    @test petab_problem.gradient_method === :ForwardDiff
    @test petab_problem.hessian_method === :GaussNewton
    @test typeof(petab_problem.ode_solver.solver) <: QNDF
    petab_problem = PEtabODEProblem(petab_model, reuse_sensitivities=true, verbose=false)
    @test petab_problem.gradient_method === :ForwardEquations
    @test petab_problem.hessian_method === :GaussNewton
    @test typeof(petab_problem.ode_solver.solver) <: QNDF

    path_yaml = joinpath(@__DIR__, "Test_ll", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml")
    petab_model = PEtabModel(path_yaml, verbose=false, build_julia_files=false)
    petab_problem = PEtabODEProblem(petab_model, verbose=false)
    @test petab_problem.gradient_method === :ForwardDiff
    @test petab_problem.hessian_method === :ForwardDiff
    @test typeof(petab_problem.ode_solver.solver) <: Rodas5P

    path_yaml = joinpath(@__DIR__, "Test_ll", "Beer_MolBioSystems2014", "Beer_MolBioSystems2014.yaml")
    petab_model = PEtabModel(path_yaml, verbose=false, build_julia_files=false)
    petab_problem = PEtabODEProblem(petab_model, verbose=false)
    @test petab_problem.gradient_method === :Adjoint
    @test petab_problem.hessian_method === :GaussNewton
    @test typeof(petab_problem.ode_solver.solver) <: CVODE_BDF
    petab_problem = PEtabODEProblem(petab_model, gradient_method=:ForwardDiff, verbose=false)
    @test typeof(petab_problem.ode_solver.solver) <: KenCarp4
end