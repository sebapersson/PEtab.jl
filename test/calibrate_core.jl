#=
    Test that parameter estimation wrappers work as they should, from start-guess
    generation, to single-start and multistart parameter estimation.
=#

using PEtab, Distributions, CSV, DataFrames, OrdinaryDiffEqRosenbrock, Catalyst, Lux,
    ComponentArrays, Optim, Ipopt, Optimization, OptimizationOptimJL, Test
import Random

@testset "Generate startguesses" begin
    # Test startguesses for a hard to integrate ODE model
    path_yaml = joinpath(
        @__DIR__, "published_models", "Crauste_CellSystems2017",
        "Crauste_CellSystems2017.yaml"
    )
    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model; odesolver = ODESolver(Rodas5P(); verbose = false))
    # Single start-guess
    x_start = get_startguesses(prob, 1)
    @test x_start isa ComponentArray
    @test !isinf(prob.nllh(x_start))
    # Multiple start-guesses
    x_starts = get_startguesses(prob, 100)
    @test length(x_starts) == 100
    @test x_starts[1] != x_starts[end]
    for x_start in x_starts
        @test !isinf(prob.nllh(x_start))
    end

    # Test start-guesses for a model with priors
    rn = @reaction_network begin
        @parameters a0 b0
        @species A(t) = a0 B(t) = b0
        (k1, k2), A <--> B
    end
    measurements = DataFrame(
        obs_id = ["obs_a", "obs_a"], time = [0, 10.0], measurement = [0.7, 0.1],
        noise_parameters = 0.5
    )
    parameters = [
        PEtabParameter(:a0, value = 1.0, scale = :lin, prior = Normal(1.5, 0.1)),
        PEtabParameter(:b0, value = 0.0, scale = :lin, prior = Normal(1.5, 0.001)),
        PEtabParameter(:k1, value = 0.8, scale = :lin),
        PEtabParameter(:k2, value = 0.6, scale = :lin, lb = 0.4, ub = 0.8),
    ]
    @unpack A = rn
    observables = PEtabObservable("obs_a", A, 0.5)
    model = PEtabModel(rn, observables, measurements, parameters; verbose = false)
    prob = PEtabODEProblem(model)
    x_starts = get_startguesses(prob, 10000)
    @test mean([x.a0 for x in x_starts]) ≈ 1.5 atol = 1.0e-1
    @test std([x.a0 for x in x_starts]) ≈ 0.1 atol = 1.0e-2
    @test mean([x.b0 for x in x_starts]) ≈ 1.5 atol = 1.0e-1
    @test std([x.b0 for x in x_starts]) ≈ 0.001 atol = 1.0e-2
    @test all([x.k2 for x in x_starts] .> 0.4)
    @test all([x.k2 for x in x_starts] .< 0.8)
    # Test seed-provided RNG behaves as expected
    rng1, rng2, rng3 = Random.Xoshiro(1), Random.Xoshiro(1), Random.Xoshiro(2)
    x1 = get_startguesses(rng1, prob, 10)
    x2 = get_startguesses(rng2, prob, 10)
    x3 = get_startguesses(rng3, prob, 10)
    @test x1 == x2
    @test x1 != x3

    # SciML model without priors and 2 MLModels
    path_yaml = joinpath(
        @__DIR__, "petab_sciml_testsuite", "test_cases", "sciml_problem_import", "008", "petab",
        "problem.yaml"
    )
    ml_models = MLModels(path_yaml)
    model = PEtabModel(path_yaml; ml_models = ml_models)
    prob = PEtabODEProblem(model; odesolver = ODESolver(Rodas5P(), verbose = false))
    # Single start-guess
    x = get_startguesses(prob, 1)
    @test x isa ComponentArray
    @test !isinf(prob.nllh(x))
    # Multiple start-guesses
    x = get_startguesses(prob, 20)
    @test !any([isinf(prob.nllh(_x)) for _x in x])
    @test x[1] != x[end]
    # Initialization of weights ...
    x = get_startguesses(prob, 1; init_weight = Lux.zeros64)
    @test all(x.net1.layer1.bias .!= 0.0)
    @test all(x.net1.layer1.weight .== 0.0)
    @test all(x.net1.layer2.weight .== 0.0)
    @test all(x.net1.layer3.weight .== 0.0)
    @test all(x.net2.layer1.weight .== 0.0)
    @test all(x.net2.layer2.weight .== 0.0)
    @test all(x.net2.layer3.weight .== 0.0)
    # and biases
    x = get_startguesses(prob, 1; init_bias = Lux.zeros64)
    @test all(x.net1.layer1.weight .!= 0.0)
    @test all(x.net1.layer1.bias .== 0.0)
    @test all(x.net1.layer2.bias .== 0.0)
    @test all(x.net1.layer3.bias .== 0.0)
    @test all(x.net2.layer1.bias .== 0.0)
    @test all(x.net2.layer2.bias .== 0.0)
    @test all(x.net2.layer3.bias .== 0.0)
    # Test rng is propagated as expected
    rng1, rng2, rng3 = Random.Xoshiro(1), Random.Xoshiro(1), Random.Xoshiro(2)
    x1 = get_startguesses(rng1, prob, 10)
    x2 = get_startguesses(rng2, prob, 10)
    x3 = get_startguesses(rng3, prob, 10)
    @test x1 == x2
    @test x1 != x3

    # SciML model with priors
    rng = Random.Xoshiro(3)
    path_yaml = joinpath(
        @__DIR__, "petab_sciml_testsuite", "test_cases", "sciml_problem_import", "033",
        "petab", "problem.yaml"
    )
    ml_models = MLModels(path_yaml)
    model = PEtabModel(path_yaml; ml_models = ml_models)
    prob = PEtabODEProblem(model; odesolver = ODESolver(Rodas5P(), verbose = false))
    x_starts = get_startguesses(rng, prob, 20000)
    @test mean([x.net1.layer1[1] for x in x_starts]) ≈ 0.0 atol = 1.0e-1
    @test std([x.net1.layer1[end] for x in x_starts]) ≈ 2.0 atol = 1.0e-2
    @test mean([x.net1.layer2[1] for x in x_starts]) ≈ 0.0 atol = 1.0e-1
    @test std([x.net1.layer2[end] for x in x_starts]) ≈ 1.0 atol = 1.0e-2
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
    @test all(.≈(res1.xmin, get_x(prob), atol = 1.0e-2))
    @test all(.≈(res2.xmin, get_x(prob), atol = 1.0e-2))
    @test all(.≈(res3.xmin, get_x(prob), atol = 1.0e-2))
    # Testing Ipopt.jl
    res4 = calibrate(
        prob, x0, IpoptOptimizer(true), options = IpoptOptions(print_level = 0)
    )
    res5 = calibrate(
        prob, x0, IpoptOptimizer(false), options = IpoptOptions(print_level = 0)
    )
    @test all(.≈(res4.xmin, get_x(prob), atol = 1.0e-2))
    @test all(.≈(res5.xmin, get_x(prob), atol = 1.0e-2))
    # Testing Optimization.jl (this package is set to have heavy updates, hence limited support)
    optprob = OptimizationProblem(prob; box_constraints = true)
    optprob.u0 .= x0
    res6 = solve(optprob, ParticleSwarm())
    @test all(.≈(res6.u, get_x(prob), atol = 1.0e-2))
end

@testset "Calibrate multi-start" begin
    dirsave = joinpath(@__DIR__, "calibrate_tmp")
    path_yaml = joinpath(@__DIR__, "analytic_solution", "Test_model2.yaml")
    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model)
    res1 = calibrate_multistart(
        prob, Optim.IPNewton(), 10; save_trace = true, dirsave = dirsave
    )
    rng = Random.Xoshiro(1)
    res2 = calibrate_multistart(rng, prob, IpoptOptimizer(true), 10; save_trace = false)
    res_read = PEtabMultistartResult(dirsave)
    @test all(.≈(res1.xmin, get_x(prob), atol = 1.0e-2))
    @test all(.≈(res2.xmin, get_x(prob), atol = 1.0e-2))
    @test all(.≈(res_read.xmin, get_x(prob), atol = 1.0e-2))
    rm(dirsave, recursive = true)
end
