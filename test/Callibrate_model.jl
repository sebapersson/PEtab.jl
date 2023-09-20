using OrdinaryDiffEq
using Test
using QuasiMonteCarlo
using Optim 
using Ipopt
using PEtab


#= 
    Test model callibration 
=#


@testset "Test model callibration" begin
    # The Crauste model is numerically horrible, so here several start guesses will fail properly testing 
    # the start-guess scheme 
    pathYAML = joinpath(@__DIR__, "Test_ll", "Crauste_CellSystems2017", "Crauste_CellSystems2017.yaml")
    petabModel = readPEtabModel(pathYAML, verbose=false)
    petabProblem = createPEtabODEProblem(petabModel, odeSolverOptions=ODESolverOptions(Rodas5P()), verbose=false)
    startGuesses = PEtab.generateStartGuesses(petabProblem, QuasiMonteCarlo.LatinHypercubeSample(), 100; verbose=false)
    @test startGuesses isa Matrix{Float64}
    @info "Done with start-guess test"

    # Test optimisers 
    pathYAML = joinpath(@__DIR__, "Test_model2", "Test_model2.yaml")
    petabModel = readPEtabModel(pathYAML, verbose=false)
    petabProblem = createPEtabODEProblem(petabModel, odeSolverOptions=ODESolverOptions(Rodas5P()), verbose=false)
    p0 = petabProblem.θ_nominalT .* 0.5

    res1 = calibrateModel(petabProblem, p0, Optim.IPNewton())
    @test all(abs.(res1.xMin - petabProblem.θ_nominalT) .< 1e-2)

    res2 = calibrateModel(petabProblem, p0, IpoptOptimiser(false), options=IpoptOptions(print_level=0))
    @test all(abs.(res2.xMin - petabProblem.θ_nominalT) .< 1e-2)

    res3 = calibrateModelMultistart(petabProblem, Optim.IPNewton(), 10, nothing)
    @test all(abs.(res3.xMin - petabProblem.θ_nominalT) .< 1e-2)

    res4 = calibrateModelMultistart(petabProblem, IpoptOptimiser(true), 10, nothing, options=IpoptOptions(print_level=0))
    @test all(abs.(res4.xMin - petabProblem.θ_nominalT) .< 1e-2)
end