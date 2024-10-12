#=
    PEtab-select test-suite (except the very long time to run Blasi model)
=#

using PEtab, Optim, PyCall, YAML, Test

@testset "PEtab-select" begin
    path_yaml = joinpath(@__DIR__, "petab_select", "0001", "petab_select_problem.yaml")
    path_exepected = joinpath(@__DIR__, "petab_select", "0001", "expected.yaml")
    path_save = petab_select(path_yaml, IPNewton())
    expected_res = YAML.load_file(path_exepected)
    data_res = YAML.load_file(path_save)
    @test expected_res["criteria"]["NLLH"] ≈ data_res["criteria"]["NLLH"] atol=1e-3
    rm(path_save)

    path_yaml = joinpath(@__DIR__, "petab_select", "0002", "petab_select_problem.yaml")
    path_exepected = joinpath(@__DIR__, "petab_select", "0002", "expected.yaml")
    path_save = petab_select(path_yaml, IPNewton(), gradient_method=:ForwardDiff,
                             hessian_method=:ForwardDiff, nmultistarts=10)
    expected_res = YAML.load_file(path_exepected)
    data_res = YAML.load_file(path_save)
    @test expected_res["criteria"]["NLLH"] ≈ data_res["criteria"]["NLLH"] atol=1e-3
    rm(path_save)

    path_yaml = joinpath(@__DIR__, "petab_select", "0003", "petab_select_problem.yaml")
    path_exepected = joinpath(@__DIR__, "petab_select", "0003", "expected.yaml")
    path_save = petab_select(path_yaml, Fides(nothing), nmultistarts=10)
    expected_res = YAML.load_file(path_exepected)
    data_res = YAML.load_file(path_save)
    @test expected_res["criteria"]["NLLH"] ≈ data_res["criteria"]["NLLH"] atol=1e-3
    rm(path_save)

    path_yaml = joinpath(@__DIR__, "petab_select", "0004", "petab_select_problem.yaml")
    path_exepected = joinpath(@__DIR__, "petab_select", "0004", "expected.yaml")
    path_save = petab_select(path_yaml, Fides(nothing; verbose=false), gradient_method=:ForwardEquations, hessian_method=:GaussNewton, reuse_sensitivities=true, sensealg=:ForwardDiff, nmultistarts=10)
    expected_res = YAML.load_file(path_exepected)
    data_res = YAML.load_file(path_save)
    @test expected_res["criteria"]["NLLH"] ≈ data_res["criteria"]["NLLH"] atol=1e-3
    rm(path_save)

    path_yaml = joinpath(@__DIR__, "petab_select", "0005", "petab_select_problem.yaml")
    path_exepected = joinpath(@__DIR__, "petab_select", "0005", "expected.yaml")
    path_save = petab_select(path_yaml, Fides(:BFGS), gradient_method=:ForwardDiff, nmultistarts=10)
    expected_res = YAML.load_file(path_exepected)
    data_res = YAML.load_file(path_save)
    @test expected_res["criteria"]["NLLH"] ≈ data_res["criteria"]["NLLH"] atol=1e-3
    rm(path_save)

    path_yaml = joinpath(@__DIR__, "petab_select", "0006", "petab_select_problem.yaml")
    path_exepected = joinpath(@__DIR__, "petab_select", "0006", "expected.yaml")
    path_save = petab_select(path_yaml, IPNewton(), nmultistarts=20)
    expected_res = YAML.load_file(path_exepected)
    data_res = YAML.load_file(path_save)
    @test expected_res["criteria"]["NLLH"] ≈ data_res["criteria"]["NLLH"] atol=1e-3
    rm(path_save)

    path_yaml = joinpath(@__DIR__, "petab_select", "0007", "petab_select_problem.yaml")
    path_exepected = joinpath(@__DIR__, "petab_select", "0007", "expected.yaml")
    path_save = petab_select(path_yaml, Fides(nothing), gradient_method=:ForwardEquations,
                             hessian_method=:GaussNewton, reuse_sensitivities=true,
                             sensealg=:ForwardDiff, nmultistarts=10)
    expected_res = YAML.load_file(path_exepected)
    data_res = YAML.load_file(path_save)
    @test expected_res["criteria"]["NLLH"] ≈ data_res["criteria"]["NLLH"] atol=1e-3
    rm(path_save)

    path_yaml = joinpath(@__DIR__, "petab_select", "0008", "petab_select_problem.yaml")
    path_exepected = joinpath(@__DIR__, "petab_select", "0008", "expected.yaml")
    path_save = petab_select(path_yaml, Fides(nothing), gradient_method=:ForwardEquations,
                             hessian_method=:GaussNewton, reuse_sensitivities=true,
                             sensealg=:ForwardDiff, nmultistarts=10)
    expected_res = YAML.load_file(path_exepected)
    data_res = YAML.load_file(path_save)
    @test expected_res["criteria"]["NLLH"] ≈ data_res["criteria"]["NLLH"] atol=1e-3
    rm(path_save)
end
