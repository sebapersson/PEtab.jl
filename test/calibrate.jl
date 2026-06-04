#=
    Test that parameter estimation wrappers work as they should, from start-guess
    generation, to single-start and multistart parameter estimation.
=#

using PEtab, Distributions, CSV, DataFrames, OrdinaryDiffEqRosenbrock, Catalyst,
    ComponentArrays, Optim, Ipopt, Fides, Optimization, OptimizationOptimJL, Test

@testset "Calibrate single start Fides" begin
    path_yaml = joinpath(@__DIR__, "analytic_solution", "Test_model2.yaml")
    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model)
    x0 = get_x(prob) .* 0.5
    # Fides.jl (the package uses PythonCall to wrap fides.py)
    res1 = calibrate(prob, x0, Fides.CustomHessian())
    res2 = calibrate(prob, x0, Fides.BFGS())
    @test all(.≈(res1.xmin, prob.xnominal_transformed, atol = 1.0e-2))
    @test all(.≈(res2.xmin, prob.xnominal_transformed, atol = 1.0e-2))
end

@testset "Calibrate multi-start Fides" begin
    dirsave = joinpath(@__DIR__, "calibrate_tmp")
    !isdir(dirsave) && mkdir(dirsave)
    path_yaml = joinpath(@__DIR__, "analytic_solution", "Test_model2.yaml")
    prob = PEtabModel(path_yaml) |>
        PEtabODEProblem
    res1 = calibrate_multistart(
        prob, Fides.BFGS(), 10; save_trace = false, dirsave = dirsave
    )
    res_read = PEtabMultistartResult(dirsave)
    @test all(.≈(res1.xmin, prob.xnominal_transformed, atol = 1.0e-2))
    @test all(.≈(res_read.xmin, prob.xnominal_transformed, atol = 1.0e-2))
    rm(dirsave, recursive = true)
end

@testset "Calibrate multi-start parallel" begin
    dirsave = joinpath(@__DIR__, "calibrate_tmp")
    !isdir(dirsave) && mkdir(dirsave)
    prob = PEtabModel(joinpath(@__DIR__, "analytic_solution", "Test_model2.yaml")) |>
        PEtabODEProblem
    res1 = calibrate_multistart(
        prob, Optim.IPNewton(), 10; save_trace = true, dirsave = dirsave, nprocs = 2
    )
    res2 = calibrate_multistart(prob, IpoptOptimizer(true), 10; nprocs = 2)
    @info "Starting Fides on multiple workers"
    res3 = calibrate_multistart(prob, Fides.BFGS(), 10; nprocs = 2)
    @info "Done with Fides on multiple workers"
    res_read = PEtabMultistartResult(dirsave)
    @test all(.≈(res1.xmin, get_x(prob), atol = 1.0e-2))
    @test all(.≈(res2.xmin, get_x(prob), atol = 1.0e-2))
    @test all(.≈(res3.xmin, get_x(prob), atol = 1.0e-2))
    @test all(.≈(res_read.xmin, get_x(prob), atol = 1.0e-2))
    rm(dirsave; recursive = true)
end
