using OrdinaryDiffEq
using Test
using QuasiMonteCarlo
using Optim
using Ipopt
using PEtab
using Optimization
using OptimizationOptimJL
using Catalyst
using DataFrames
using Distributions
using Printf

#=
    Test model callibration
=#

@testset "Test model callibration" begin
    # The Crauste model is numerically horrible, so here several start guesses will fail properly testing
    # the start-guess scheme
    path_yaml = joinpath(@__DIR__, "published_models", "Crauste_CellSystems2017", "Crauste_CellSystems2017.yaml")
    model = PEtabModel(path_yaml, verbose=false)
    petab_problem = PEtabODEProblem(model, ode_solver=ODESolver(Rodas5P()), verbose=false)
    startguesses = generate_startguesses(petab_problem, 100; verbose=false,
                                         sampling_method=QuasiMonteCarlo.LatinHypercubeSample())
    @test startguesses isa Matrix{Float64}
    startguesses = generate_startguesses(petab_problem, 1; verbose=false,
                                         sampling_method=QuasiMonteCarlo.LatinHypercubeSample())
    @test startguesses isa Vector{Float64}
    @info "Done with Crauste start-guess test"

    # Test sample start-guesses when we have a prior that we sample from
    rn = @reaction_network begin
        @parameters a0 b0
        @species A(t)=a0 B(t)=b0
        (k1, k2), A <--> B
    end
    measurements = DataFrame(obs_id=["obs_a", "obs_a"],
                             time=[0, 10.0],
                             measurement=[0.7, 0.1],
                             noise_parameters=0.5)
    parameters = [PEtabParameter(:a0, value=1.0, scale=:lin, prior=Normal(1.5, 0.1)),
                        PEtabParameter(:b0, value=0.0, scale=:lin, prior=Normal(1.5, 0.001)),
                        PEtabParameter(:k1, value=0.8, scale=:lin),
                        PEtabParameter(:k2, value=0.6, scale=:lin)]
    @unpack A = rn
    observables = Dict("obs_a" => PEtabObservable(A, 0.5))
    model = PEtabModel(rn, observables, measurements,
                             parameters, verbose=false)
    petab_problem = PEtabODEProblem(model, verbose=false)
    startguesses = generate_startguesses(petab_problem, 10000; verbose=false)
    @test mean(startguesses[1, :]) ≈ 1.5 atol=1e-1
    @test std(startguesses[1, :]) ≈ 0.1 atol=1e-1
    startguesses = generate_startguesses(petab_problem, 1; verbose=false)
    @test startguesses isa Vector{Float64}
    @test startguesses[2] ≤ 1.6 && startguesses[2] ≥ 1.4

    # Test optimisers
    path_yaml = joinpath(@__DIR__, "Test_model2", "Test_model2.yaml")
    model = PEtabModel(path_yaml, verbose=false)
    petab_problem = PEtabODEProblem(model, ode_solver=ODESolver(Rodas5P()), verbose=false)
    p0 = petab_problem.xnominal_transformed .* 0.5

    res1 = calibrate_model(petab_problem, p0, Optim.IPNewton())
    @test all(abs.(res1.xmin - petab_problem.xnominal_transformed) .< 1e-2)

    res2 = calibrate_model(petab_problem, p0, IpoptOptimiser(false), options=IpoptOptions(print_level=0))
    @test all(abs.(res2.xmin - petab_problem.xnominal_transformed) .< 1e-2)

    dir_save = joinpath(@__DIR__, "test_model_calibration")
    res3 = calibrate_model_multistart(petab_problem, Optim.IPNewton(), 10, dir_save, save_trace=true)
    res_read = PEtabMultistartOptimisationResult(dir_save)
    @test all(abs.(res3.xmin - petab_problem.xnominal_transformed) .< 1e-2)
    @test all(abs.(res_read.xmin - petab_problem.xnominal_transformed) .< 1e-2)
    @test res3.fmin == res_read.fmin
    @test all(res3.runs[1].ftrace .== res_read.runs[1].ftrace)
    @test all(res3.runs[1].xtrace .== res_read.runs[1].xtrace)
    rm(dir_save, recursive=true)

    res4 = calibrate_model_multistart(petab_problem, IpoptOptimiser(true), 10, nothing, options=IpoptOptions(print_level=0))
    @test all(abs.(res4.xmin - petab_problem.xnominal_transformed) .< 1e-2)

    # Test Optimization.jl
    # Interior-point Newton
    optimization_problem = PEtab.OptimizationProblem(petab_problem; interior_point_alg=true)
    res = PEtab.calibrate_model(optimization_problem, petab_problem, p0, IPNewton(); abstol=1e-8)
    @test all(abs.(res.xmin - petab_problem.xnominal_transformed) .< 1e-2)
    # Particle swarm
    optimization_problem = PEtab.OptimizationProblem(petab_problem; interior_point_alg=false)
    res = PEtab.calibrate_model(optimization_problem, petab_problem, p0, Optim.ParticleSwarm(); abstol=1e-8)
    @test all(abs.(res.xmin - petab_problem.xnominal_transformed) .< 1e-2)
    # TrustRegionNewton
    optimization_problem = PEtab.OptimizationProblem(petab_problem; interior_point_alg=false, box_constraints=false)
    res = PEtab.calibrate_model(optimization_problem, petab_problem, p0, NewtonTrustRegion(); abstol=1e-8)
    @test all(abs.(res.xmin - petab_problem.xnominal_transformed) .< 1e-2)
    # Test Optimization.jl multistart
    optimization_problem = PEtab.OptimizationProblem(petab_problem; interior_point_alg=true)
    res = calibrate_model_multistart(optimization_problem, petab_problem, IPNewton(),
                                     10, nothing; abstol=1e-8)
    @test all(abs.(res.xmin - petab_problem.xnominal_transformed) .< 1e-2)
end
