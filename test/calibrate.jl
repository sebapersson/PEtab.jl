#=
    Test that parameter estimation wrappers work as they should, from start-guess
    generation, to single-start and multistart parameter estimation.
=#

using PEtab, Distributions, CSV, DataFrames, OrdinaryDiffEqRosenbrock, Catalyst, ComponentArrays,
      Optim, Ipopt, Fides, Optimization, OptimizationOptimJL, Test

@testset "Generate startguesses" begin
    # Test startguesses for a hard to integrate ODE model
    path_yaml = joinpath(@__DIR__, "published_models", "Crauste_CellSystems2017", "Crauste_CellSystems2017.yaml")
    model = PEtabModel(path_yaml; verbose=false)
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

    # Test starguesses for a model with priors
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
    observables = PEtabObservable(:obs_a, A, 0.5)
    model = PEtabModel(rn, observables, measurements, parameters; verbose=false)
    prob = PEtabODEProblem(model; verbose=false)
    xstarts = get_startguesses(prob, 10000)
    @test mean([x.a0 for x in xstarts]) ≈ 1.5 atol=1e-1
    @test std([x.a0 for x in xstarts]) ≈ 0.1 atol=1e-2
    @test mean([x.b0 for x in xstarts]) ≈ 1.5 atol=1e-1
    @test std([x.b0 for x in xstarts]) ≈ 0.001 atol=1e-2
    @test all([x.k2 for x in xstarts] .> 0.4)
    @test all([x.k2 for x in xstarts] .< 0.8)
end

@testset "Calibrate single start" begin
    path_yaml = joinpath(@__DIR__, "analytic_solution", "Test_model2.yaml")
    model = PEtabModel(path_yaml; verbose=false)
    prob = PEtabODEProblem(model; verbose=false)
    x0 = prob.xnominal_transformed .* 0.5
    # Testing Optim.jl
    res1 = calibrate(prob, x0, Optim.IPNewton())
    res2 = calibrate(prob, x0, Optim.BFGS())
    res3 = calibrate(prob, x0, Optim.LBFGS())
    @test all(.≈(res1.xmin, prob.xnominal_transformed, atol = 1e-2))
    @test all(.≈(res2.xmin, prob.xnominal_transformed, atol = 1e-2))
    @test all(.≈(res3.xmin, prob.xnominal_transformed, atol = 1e-2))
    # Testing Ipopt.jl
    res4 = calibrate(prob, x0, IpoptOptimizer(true), options=IpoptOptions(print_level=0))
    res5 = calibrate(prob, x0, IpoptOptimizer(false), options=IpoptOptions(print_level=0))
    @test all(.≈(res4.xmin, prob.xnominal_transformed, atol = 1e-2))
    @test all(.≈(res5.xmin, prob.xnominal_transformed, atol = 1e-2))
    # Testing Optimization.jl (this package is set to have heavy updates, hence limited support)
    optprob = OptimizationProblem(prob; box_constraints = true)
    optprob.u0 .= x0
    res6 = Optimization.solve(optprob, ParticleSwarm())
    @test all(.≈(res6.u, prob.xnominal_transformed, atol = 1e-2))
    # Fides.jl (the package uses PythonCall to wrap fides.py)
    res7 = calibrate(prob, x0, Fides.CustomHessian())
    res8 = calibrate(prob, x0, Fides.BFGS())
    @test all(.≈(res7.xmin, prob.xnominal_transformed, atol = 1e-2))
    @test all(.≈(res8.xmin, prob.xnominal_transformed, atol = 1e-2))
end

@testset "Calibrate multi-start Fides" begin
    dirsave = joinpath(@__DIR__, "calibrate_tmp")
    path_yaml = joinpath(@__DIR__, "analytic_solution", "Test_model2.yaml")
    model = PEtabModel(path_yaml; verbose=false)
    prob = PEtabODEProblem(model; verbose=false)
    res1 = calibrate_multistart(prob, Optim.IPNewton(), 10; save_trace=true,
                                dirsave = dirsave)
    res2 = calibrate_multistart(prob, IpoptOptimizer(true), 10; save_trace=false)
    res3 = calibrate_multistart(prob, Fides.BFGS(), 10; save_trace=false)
    res_read = PEtabMultistartResult(dirsave)
    @test all(.≈(res1.xmin, prob.xnominal_transformed, atol = 1e-2))
    @test all(.≈(res2.xmin, prob.xnominal_transformed, atol = 1e-2))
    @test all(.≈(res3.xmin, prob.xnominal_transformed, atol = 1e-2))
    @test all(.≈(res_read.xmin, prob.xnominal_transformed, atol = 1e-2))
    rm(dirsave, recursive = true)
end

@testset "Calibrate multi-start parallel" begin
    path_yaml = joinpath(@__DIR__, "analytic_solution", "Test_model2.yaml")
    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model)
    x0 = get_x(prob) .* 0.5
    dirsave = joinpath(@__DIR__, "calibrate_tmp")
    res1 = calibrate_multistart(prob, Optim.IPNewton(), 10; save_trace=true,
                                dirsave = dirsave, nprocs = 2)
    res2 = calibrate_multistart(prob, IpoptOptimizer(true), 10; nprocs = 2)
    @info "Starting Fides on multiple workers"
    res3 = calibrate_multistart(prob, Fides.BFGS(), 10; nprocs = 2)
    @info "Done with Fides on multiple workers"
    res_read = PEtabMultistartResult(dirsave)
    @test all(.≈(res1.xmin, get_x(prob), atol = 1e-2))
    @test all(.≈(res2.xmin, get_x(prob), atol = 1e-2))
    @test all(.≈(res3.xmin, get_x(prob), atol = 1e-2))
    @test all(.≈(res_read.xmin, get_x(prob), atol = 1e-2))
    # Due to startup overhead many multistarts must be performed two find the effect
    # in runtime
    b1 = @elapsed res1 = calibrate_multistart(prob, Fides.BFGS(), 2500; nprocs = 1)
    b2 = @elapsed res2 = calibrate_multistart(prob, Fides.BFGS(), 2500; nprocs = 2)
    @test b1 > b2
    rm(dirsave; recursive = true)
end
