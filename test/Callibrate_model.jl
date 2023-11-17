using OrdinaryDiffEq
using Test
using QuasiMonteCarlo
using Optim
using Ipopt
using PEtab
using Optimization
using OptimizationOptimJL


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

    dir_save = joinpath(@__DIR__, "test_model_calibration")
    res3 = calibrate_model_multistart(petab_problem, Optim.IPNewton(), 10, dir_save, save_trace=true)
    res_read = PEtabMultistartOptimisationResult(dir_save)
    @test all(abs.(res3.xmin - petab_problem.θ_nominalT) .< 1e-2)
    @test all(abs.(res_read.xmin - petab_problem.θ_nominalT) .< 1e-2)
    @test res3.fmin == res_read.fmin
    @test all(res3.runs[1].ftrace .== res_read.runs[1].ftrace)
    @test all(res3.runs[1].xtrace .== res_read.runs[1].xtrace)
    rm(dir_save, recursive=true)

    res4 = calibrate_model_multistart(petab_problem, IpoptOptimiser(true), 10, nothing, options=IpoptOptions(print_level=0))
    @test all(abs.(res4.xmin - petab_problem.θ_nominalT) .< 1e-2)

    # Test Optimization.jl
    # Interior-point Newton
    optimization_problem = PEtab.OptimizationProblem(petab_problem; interior_point_alg=true)
    res = PEtab.calibrate_model(optimization_problem, p0, IPNewton(); abstol=1e-8)
    @test all(abs.(res.xmin - petab_problem.θ_nominalT) .< 1e-2)
    # Particle swarm
    optimization_problem = PEtab.OptimizationProblem(petab_problem; interior_point_alg=false)
    res = PEtab.calibrate_model(optimization_problem, p0, Optim.ParticleSwarm(); abstol=1e-8)
    @test all(abs.(res.xmin - petab_problem.θ_nominalT) .< 1e-2)
    # TrustRegionNewton
    optimization_problem = PEtab.OptimizationProblem(petab_problem; interior_point_alg=false, box_constraints=false)
    res = PEtab.calibrate_model(optimization_problem, p0, NewtonTrustRegion(); abstol=1e-8)
    @test all(abs.(res.xmin - petab_problem.θ_nominalT) .< 1e-2)
    # Test Optimization.jl multistart
    optimization_problem = PEtab.OptimizationProblem(petab_problem; interior_point_alg=true)
    res = calibrate_model_multistart(optimization_problem, petab_problem, IPNewton(),
                                     10, nothing; abstol=1e-8)
    @test all(abs.(res.xmin - petab_problem.θ_nominalT) .< 1e-2)
end