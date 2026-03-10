#=
    PEtab-select test-suite (except the very long time to run Blasi model)
=#

using PEtab, Optim, Fides, PEtabSelect, YAML, Test, OrdinaryDiffEqRosenbrock

dir_tests = joinpath(@__DIR__, "petab_select_testsuite")

@testset "PEtab-select" begin
    path_yaml = joinpath(dir_tests, "0001", "petab_select_problem.yaml")
    path_exepected = joinpath(dir_tests, "0001", "expected.yaml")
    path_save = petab_select(path_yaml, IPNewton())
    expected_res = YAML.load_file(path_exepected)
    data_res = YAML.load_file(path_save)
    @test expected_res["criteria"]["NLLH"] ≈ data_res["criteria"]["NLLH"] atol = 1.0e-3
    rm(path_save)

    path_yaml = joinpath(dir_tests, "0002", "petab_select_problem.yaml")
    path_exepected = joinpath(dir_tests, "0002", "expected.yaml")
    path_save = petab_select(
        path_yaml, IPNewton(), gradient_method = :ForwardDiff,
        hessian_method = :ForwardDiff, nmultistarts = 10
    )
    expected_res = YAML.load_file(path_exepected)
    data_res = YAML.load_file(path_save)
    @test expected_res["criteria"]["NLLH"] ≈ data_res["criteria"]["NLLH"] atol = 1.0e-3
    rm(path_save)

    path_yaml = joinpath(dir_tests, "0003", "petab_select_problem.yaml")
    path_exepected = joinpath(dir_tests, "0003", "expected.yaml")
    path_save = petab_select(
        path_yaml, Fides.CustomHessian(), nmultistarts = 10;
        odesolver = ODESolver(Rodas5P(); verbose = false),
        options = FidesOptions(maxiter = 1000)
    )
    expected_res = YAML.load_file(path_exepected)
    data_res = YAML.load_file(path_save)
    @test expected_res["criteria"]["NLLH"] ≈ data_res["criteria"]["NLLH"] atol = 1.0e-3
    rm(path_save)

    path_yaml = joinpath(dir_tests, "0004", "petab_select_problem.yaml")
    path_exepected = joinpath(dir_tests, "0004", "expected.yaml")
    path_save = petab_select(
        path_yaml, Fides.CustomHessian(), gradient_method = :ForwardEquations,
        hessian_method = :GaussNewton, reuse_sensitivities = true, sensealg = :ForwardDiff,
        nmultistarts = 10
    )
    expected_res = YAML.load_file(path_exepected)
    data_res = YAML.load_file(path_save)
    @test expected_res["criteria"]["NLLH"] ≈ data_res["criteria"]["NLLH"] atol = 1.0e-3
    rm(path_save)

    path_yaml = joinpath(dir_tests, "0005", "petab_select_problem.yaml")
    path_exepected = joinpath(dir_tests, "0005", "expected.yaml")
    path_save = petab_select(
        path_yaml, Fides.BFGS(), gradient_method = :ForwardDiff, nmultistarts = 10
    )
    expected_res = YAML.load_file(path_exepected)
    data_res = YAML.load_file(path_save)
    @test expected_res["criteria"]["NLLH"] ≈ data_res["criteria"]["NLLH"] atol = 1.0e-3
    rm(path_save)

    path_yaml = joinpath(dir_tests, "0006", "petab_select_problem.yaml")
    path_exepected = joinpath(dir_tests, "0006", "expected.yaml")
    path_save = petab_select(path_yaml, IPNewton(), nmultistarts = 20)
    expected_res = YAML.load_file(path_exepected)
    data_res = YAML.load_file(path_save)
    @test expected_res["criteria"]["NLLH"] ≈ data_res["criteria"]["NLLH"] atol = 1.0e-3
    rm(path_save)

    path_yaml = joinpath(dir_tests, "0007", "petab_select_problem.yaml")
    path_exepected = joinpath(dir_tests, "0007", "expected.yaml")
    path_save = petab_select(
        path_yaml, Fides.CustomHessian(), gradient_method = :ForwardEquations,
        hessian_method = :GaussNewton, reuse_sensitivities = true,
        sensealg = :ForwardDiff, nmultistarts = 10
    )
    expected_res = YAML.load_file(path_exepected)
    data_res = YAML.load_file(path_save)
    @test expected_res["criteria"]["NLLH"] ≈ data_res["criteria"]["NLLH"] atol = 1.0e-3
    rm(path_save)

    path_yaml = joinpath(dir_tests, "0008", "petab_select_problem.yaml")
    path_exepected = joinpath(dir_tests, "0008", "expected.yaml")
    path_save = petab_select(
        path_yaml, Fides.CustomHessian(), gradient_method = :ForwardEquations,
        hessian_method = :GaussNewton, reuse_sensitivities = true, sensealg = :ForwardDiff,
        nmultistarts = 10
    )
    expected_res = YAML.load_file(path_exepected)
    data_res = YAML.load_file(path_save)
    @test expected_res["criteria"]["NLLH"] ≈ data_res["criteria"]["NLLH"] atol = 1.0e-3
    rm(path_save)
end
