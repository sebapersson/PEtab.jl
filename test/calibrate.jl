using PEtab, Distributions, CSV, DataFrames, OrdinaryDiffEq, Catalyst, ComponentArrays,
      Optim, Ipopt, Optimization, OptimizationOptimJL, PyCall, Test

@testset "Calibrate single start Fides" begin
    path_yaml = joinpath(@__DIR__, "analytic_solution", "Test_model2.yaml")
    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model)
    x0 = get_x(prob) .* 0.5
    # Fides.py (requires a PyCall, but Fides often works well so...)
    res1 = calibrate(prob, x0, Fides(nothing))
    res2 = calibrate(prob, x0, Fides(:BFGS))
    @test all(.≈(res1.xmin, get_x(prob), atol = 1e-2))
    @test all(.≈(res2.xmin, get_x(prob), atol = 1e-2))
end

@testset "Calibrate multi-start Fides" begin
    dirsave = joinpath(@__DIR__, "calibrate_tmp")
    path_yaml = joinpath(@__DIR__, "analytic_solution", "Test_model2.yaml")
    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model)
    res = calibrate_multistart(prob, Fides(:BFGS), 10; save_trace=false)
    @test all(.≈(res.xmin, get_x(prob), atol = 1e-2))
end

@testset "Calibrate multi-start parallel" begin
    path_yaml = joinpath(@__DIR__, "analytic_solution", "Test_model2.yaml")
    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model)
    x0 = get_x(prob) .* 0.5
    dirsave = joinpath(@__DIR__, "calibrate_tmp")
    res1 = calibrate_multistart(prob, Optim.IPNewton(), 10; save_trace=true,
                                dirsave = dirsave, nprocs = 2)
    #res2 = calibrate_multistart(prob, IpoptOptimizer(true), 10; nprocs = 2)
    res3 = calibrate_multistart(prob, Fides(:BFGS), 10; nprocs = 2)
    res_read = PEtabMultistartResult(dirsave)
    @test all(.≈(res1.xmin, get_x(prob), atol = 1e-2))
    #@test all(.≈(res2.xmin, get_x(prob), atol = 1e-2))
    @test all(.≈(res3.xmin, get_x(prob), atol = 1e-2))
    @test all(.≈(res_read.xmin, get_x(prob), atol = 1e-2))
    # Due to startup overhead many multistarts must be performed two find the effect
    # in runtime
    b1 = @elapsed res1 = calibrate_multistart(prob, Optim.IPNewton(), 2500; nprocs = 1)
    b2 = @elapsed res2 = calibrate_multistart(prob, Optim.IPNewton(), 2500; nprocs = 2)
    @test b1 > b2
    rm(dirsave; recursive = true)
end
