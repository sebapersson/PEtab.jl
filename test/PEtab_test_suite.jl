using PEtab
using YAML
using Test
using OrdinaryDiffEq
using Sundials
using CSV
using ForwardDiff
using LinearAlgebra
using FiniteDifferences


function test_reference_case(test_case::String; testGradient::Bool=true)

    @info "Test case $test_case"
    path_yaml = joinpath(@__DIR__, "PEtab_test_suite", test_case, "_" * test_case * ".yaml")
    path_reference_values = joinpath(@__DIR__, "PEtab_test_suite", test_case, "_" * test_case * "_solution.yaml")

    petab_model = PEtabModel(path_yaml, verbose=false, build_julia_files=true)
    petab_problem = PEtabODEProblem(petab_model, verbose=false)

    cost = petab_problem.compute_cost(petab_problem.θ_nominalT)
    χ2 = petab_problem.compute_chi2(petab_problem.θ_nominalT)
    simulated_values = petab_problem.compute_simulated_values(petab_problem.θ_nominalT)

    reference_yaml = YAML.load_file(path_reference_values)
    path_simulations = joinpath(dirname(path_reference_values), reference_yaml["simulation_files"][1])
    cost_ref, tol_cost = -1 * reference_yaml["llh"], reference_yaml["tol_llh"]
    χ2_ref, tol_χ2 = reference_yaml["chi2"], reference_yaml["tol_chi2"]
    simulated_values_ref, tol_simulations = CSV.File(path_simulations)[:simulation], reference_yaml["tol_simulations"]

    @test cost ≈ cost_ref atol = tol_cost
    @test χ2 ≈ χ2_ref atol = tol_χ2
    @test all(abs.(simulated_values .- simulated_values_ref) .≤ tol_simulations)

    if testGradient == true
        gradient_ref = FiniteDifferences.grad(central_fdm(5, 1), petab_problem.compute_cost, petab_problem.θ_nominalT)[1]
        g = petab_problem.compute_gradient(petab_problem.θ_nominalT)
        @test norm(g - gradient_ref) ≤ 1e-7
    end
end


@testset "PEtab test-suite" begin
    test_reference_case("0001")
    test_reference_case("0002")
    test_reference_case("0003")
    test_reference_case("0004")
    test_reference_case("0005")
    test_reference_case("0006")
    test_reference_case("0007")
    test_reference_case("0008")
    test_reference_case("0009")
    test_reference_case("0010")
    test_reference_case("0011")
    test_reference_case("0012")
    test_reference_case("0013")
    test_reference_case("0014")
    test_reference_case("0015")
    test_reference_case("0016")
    test_reference_case("0017")
    test_reference_case("0018")
end
