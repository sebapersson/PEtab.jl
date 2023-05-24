#=
    Test that default options (user does not provide ODE-solver and gradient-method)
    is handled correctly.
=#


using PEtab
using OrdinaryDiffEq
using Sundials
using Test


@testset "Test default options" begin
    # Check that we get correct default setting 
    pathYML = joinpath(@__DIR__, "Test_ll", "Bachmann_MSB2011", "Bachmann_MSB2011.yaml")
    petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
    petabProblem = createPEtabODEProblem(petabModel, verbose=false)
    @test petabProblem.gradientMethod === :ForwardDiff
    @test petabProblem.hessianMethod === :GaussNewton
    @test typeof(petabProblem.odeSolverOptions.solver) <: QNDF
    petabProblem = createPEtabODEProblem(petabModel, reuseS=true, verbose=false)
    @test petabProblem.gradientMethod === :ForwardEquations
    @test petabProblem.hessianMethod === :GaussNewton
    @test typeof(petabProblem.odeSolverOptions.solver) <: QNDF

    pathYML = joinpath(@__DIR__, "Test_ll", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml")
    petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
    petabProblem = createPEtabODEProblem(petabModel, verbose=false)
    @test petabProblem.gradientMethod === :ForwardDiff
    @test petabProblem.hessianMethod === :ForwardDiff
    @test typeof(petabProblem.odeSolverOptions.solver) <: Rodas5P

    pathYML = joinpath(@__DIR__, "Test_ll", "Beer_MolBioSystems2014", "Beer_MolBioSystems2014.yaml")
    petabModel = readPEtabModel(pathYML, verbose=false, forceBuildJuliaFiles=false)
    petabProblem = createPEtabODEProblem(petabModel, verbose=false)
    @test petabProblem.gradientMethod === :Adjoint
    @test petabProblem.hessianMethod === :nothing
    @test typeof(petabProblem.odeSolverOptions.solver) <: CVODE_BDF
    petabProblem = createPEtabODEProblem(petabModel, gradientMethod=:ForwardDiff, verbose=false)
    @test typeof(petabProblem.odeSolverOptions.solver) <: KenCarp4
end