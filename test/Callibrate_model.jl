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
    path_yaml = joinpath(@__DIR__, "Test_ll", "Crauste_CellSystems2017", "Crauste_CellSystems2017.yaml")
    petab_model = PEtabModel(path_yaml, verbose=false)
    petab_problem = PEtabODEProblem(petab_model, ode_solver=ODESolver(Rodas5P()), verbose=false)
    startguesses = generate_startguesses(petab_problem, 100; verbose=false, 
                                         sampling_method=QuasiMonteCarlo.LatinHypercubeSample())
    @test startguesses isa Matrix{Float64}
    @info "Done with start-guess test"

    # Test optimisers 
    path_yaml = joinpath(@__DIR__, "Test_model2", "Test_model2.yaml")
    petab_model = PEtabModel(path_yaml, verbose=false)
    petab_problem = PEtabODEProblem(petab_model, ode_solver=ODESolver(Rodas5P()), verbose=false)
    p0 = petab_problem.θ_nominalT .* 0.5

    res1 = calibrate_model(petab_problem, p0, Optim.IPNewton())
    @test all(abs.(res1.xmin - petab_problem.θ_nominalT) .< 1e-2)

    res2 = calibrate_model(petab_problem, p0, IpoptOptimiser(false), options=IpoptOptions(print_level=0))
    @test all(abs.(res2.xmin - petab_problem.θ_nominalT) .< 1e-2)

    res3 = calibrate_model_multistart(petab_problem, Optim.IPNewton(), 10, nothing)
    @test all(abs.(res3.xmin - petab_problem.θ_nominalT) .< 1e-2)

    res4 = calibrate_model_multistart(petab_problem, IpoptOptimiser(true), 10, nothing, options=IpoptOptions(print_level=0))
    @test all(abs.(res4.xmin - petab_problem.θ_nominalT) .< 1e-2)
end