#=
    Test that parameter estimation wrappers work as they should, from start-guess
    generation, to single-start and multistart parameter estimation.
=#

using PEtab, Distributions, CSV, DataFrames, OrdinaryDiffEqRosenbrock, Catalyst,
    ComponentArrays, Optim, Ipopt, Optimization, OptimizationOptimJL, Test
import Random

@testset "Generate startguesses" begin
    # Test startguesses for a hard to integrate ODE model
    path_yaml = joinpath(@__DIR__, "published_models", "Crauste_CellSystems2017", "Crauste_CellSystems2017.yaml")
    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model; odesolver=ODESolver(Rodas5P(); verbose = false),
                           verbose=false)
    xstart = get_startguesses(prob, 1)
    @test xstart isa ComponentArray
    @test !isinf(prob.nllh(xstart))
    xstarts = get_startguesses(prob, 100)
    @test length(xstarts) == 100
    @test xstarts[1] != xstarts[end]
    for xstart in xstarts
        @test !isinf(prob.nllh(xstart))
    end

    # Test start-guesses for a model with priors
    rn = @reaction_network begin
        @parameters a0 b0
        @species A(t)=a0 B(t)=b0
        (k1, k2), A <--> B
    end
    measurements = DataFrame(obs_id=["obs_a", "obs_a"], time=[0, 10.0], measurement=[0.7, 0.1],
                             noise_parameters=0.5)
    parameters = [PEtabParameter(:a0, value=1.0, scale=:lin, prior=Normal(1.5, 0.1)),
                  PEtabParameter(:b0, value=0.0, scale=:lin, prior=Normal(1.5, 0.001)),
                  PEtabParameter(:k1, value=0.8, scale=:lin),
                  PEtabParameter(:k2, value=0.6, scale=:lin, lb = 0.4, ub = 0.8)]
    @unpack A = rn
    observables = PEtabObservable("obs_a", A, 0.5)
    model = PEtabModel(rn, observables, measurements, parameters; verbose=false)
    prob = PEtabODEProblem(model)
    xstarts = get_startguesses(prob, 10000)
    @test mean([x.a0 for x in xstarts]) ≈ 1.5 atol=1e-1
    @test std([x.a0 for x in xstarts]) ≈ 0.1 atol=1e-2
    @test mean([x.b0 for x in xstarts]) ≈ 1.5 atol=1e-1
    @test std([x.b0 for x in xstarts]) ≈ 0.001 atol=1e-2
    @test all([x.k2 for x in xstarts] .> 0.4)
    @test all([x.k2 for x in xstarts] .< 0.8)
    # Test seed-provided RNG behaves as expected
    rng1, rng2, rng3 = Random.Xoshiro(1), Random.Xoshiro(1), Random.Xoshiro(2)
    x1 = get_startguesses(rng1, prob, 10)
    x2 = get_startguesses(rng2, prob, 10)
    x3 = get_startguesses(rng3, prob, 10)
    @test x1 == x2
    @test x1 != x3
end

@testset "Calibrate single start" begin
    path_yaml = joinpath(@__DIR__, "analytic_solution", "Test_model2.yaml")
    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model)
    x0 = get_x(prob) .* 0.5
    # Testing Optim.jl
    res1 = calibrate(prob, x0, Optim.IPNewton())
    res2 = calibrate(prob, x0, Optim.BFGS())
    res3 = calibrate(prob, x0, Optim.LBFGS())
    @test all(.≈(res1.xmin, get_x(prob), atol = 1e-2))
    @test all(.≈(res2.xmin, get_x(prob), atol = 1e-2))
    @test all(.≈(res3.xmin, get_x(prob), atol = 1e-2))
    # Testing Ipopt.jl
    res4 = calibrate(prob, x0, IpoptOptimizer(true), options=IpoptOptions(print_level=0))
    res5 = calibrate(prob, x0, IpoptOptimizer(false), options=IpoptOptions(print_level=0))
    @test all(.≈(res4.xmin, get_x(prob), atol = 1e-2))
    @test all(.≈(res5.xmin, get_x(prob), atol = 1e-2))
    # Testing Optimization.jl (this package is set to have heavy updates, hence limited support)
    optprob = OptimizationProblem(prob; box_constraints = true)
    optprob.u0 .= x0
    res6 = solve(optprob, ParticleSwarm())
    @test all(.≈(res6.u, get_x(prob), atol = 1e-2))
end

@testset "Calibrate multi-start" begin
    dirsave = joinpath(@__DIR__, "calibrate_tmp")
    path_yaml = joinpath(@__DIR__, "analytic_solution", "Test_model2.yaml")
    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model)
    res1 = calibrate_multistart(prob, Optim.IPNewton(), 10; save_trace=true,
                                dirsave = dirsave)
    rng = Random.Xoshiro(1)
    res2 = calibrate_multistart(rng, prob, IpoptOptimizer(true), 10; save_trace=false)
    res_read = PEtabMultistartResult(dirsave)
    @test all(.≈(res1.xmin, get_x(prob), atol = 1e-2))
    @test all(.≈(res2.xmin, get_x(prob), atol = 1e-2))
    @test all(.≈(res_read.xmin, get_x(prob), atol = 1e-2))
    rm(dirsave, recursive = true)
end
