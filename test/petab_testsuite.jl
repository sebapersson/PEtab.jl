using PEtab, YAML, OrdinaryDiffEq, Sundials, ForwardDiff, LinearAlgebra, FiniteDifferences,
      CSV, DataFrames, Test

function test_case(case::String; test_grad::Bool=true)
    @info "Test case $case"
    path_yaml = joinpath(@__DIR__, "petab_testsuite", case, "_$(case).yaml")
    path_ref = joinpath(@__DIR__, "petab_testsuite", case, "_$(case)_solution.yaml")

    model = PEtabModel(path_yaml, verbose=false, build_julia_files=true, write_to_file = false)
    prob = PEtabODEProblem(model; verbose=false,
                           ss_solver = SteadyStateSolver(:Simulate, abstol=1e-12, reltol=1e-10))
    x = prob.xnominal_transformed

    nllh = prob.nllh(x)
    χ₂ = prob.chi2(x)
    simvals = prob.simulated_values(x)

    reference_yaml = YAML.load_file(path_ref)
    path_simulations = joinpath(dirname(path_ref), reference_yaml["simulation_files"][1])
    nllh_ref, nllh_tol = -1 * reference_yaml["llh"], reference_yaml["tol_llh"]
    χ₂_ref, χ₂_tol = reference_yaml["chi2"], reference_yaml["tol_chi2"]
    simvals_ref, simvals_tol = CSV.read(path_simulations, DataFrame)[!, :simulation], reference_yaml["tol_simulations"]

    @test nllh ≈ nllh_ref atol = nllh_tol
    @test χ₂ ≈ χ₂_ref atol = χ₂_tol
    @test all(.≈(simvals, simvals_ref; atol = simvals_tol))
    if test_grad == true
        grad_ref = FiniteDifferences.grad(central_fdm(5, 1), prob.nllh, x)[1]
        g = prob.grad(x)
        @test all(.≈(g, grad_ref, atol = 1e-3))
    end
end

@testset "PEtab test-suite" begin
    test_case("0001")
    test_case("0002")
    test_case("0003")
    test_case("0004")
    test_case("0005")
    test_case("0006")
    test_case("0007")
    test_case("0008")
    test_case("0009")
    test_case("0010")
    test_case("0011")
    test_case("0012")
    test_case("0013")
    test_case("0014")
    test_case("0015")
    test_case("0016")
    test_case("0017")
    test_case("0018")
end
