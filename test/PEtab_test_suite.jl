using PEtab
using YAML
using Test
using OrdinaryDiffEq
using Sundials
using CSV
using ForwardDiff
using LinearAlgebra
using FiniteDifferences


function test_reference_case(testCase::String; testGradient=true)

    @info "Test case $testCase"
    path_yaml = joinpath(@__DIR__, "PEtab_test_suite", testCase, "_" * testCase * ".yaml")
    pathReferenceValues = joinpath(@__DIR__, "PEtab_test_suite", testCase, "_" * testCase * "_solution.yaml")

    petab_model = PEtabModel(path_yaml, verbose=false, build_julia_files=true)
    petab_problem = PEtabODEProblem(petab_model, verbose=false)

    cost = petab_problem.compute_cost(petab_problem.θ_nominalT)
    chi2 = petab_problem.compute_chi2(petab_problem.θ_nominalT)
    simulated_values = petab_problem.compute_simulated_values(petab_problem.θ_nominalT)

    referenceYAML = YAML.load_file(pathReferenceValues)
    pathSimulations = joinpath(dirname(pathReferenceValues), referenceYAML["simulation_files"][1])
    costRef, tolCost = -1 * referenceYAML["llh"], referenceYAML["tol_llh"]
    chi2Ref, tolChi2 = referenceYAML["chi2"], referenceYAML["tol_chi2"]
    simulated_valuesRef, tolSimulations = CSV.File(pathSimulations)[:simulation], referenceYAML["tol_simulations"]

    @test cost ≈ costRef atol = tolCost
    @test chi2 ≈ chi2Ref atol = tolChi2
    @test all(abs.(simulated_values .- simulated_valuesRef) .≤ tolSimulations)

    if testGradient == true
        gRef = FiniteDifferences.grad(central_fdm(5, 1), petab_problem.compute_cost, petab_problem.θ_nominalT)[1]
        g = petab_problem.compute_gradient(petab_problem.θ_nominalT)
        @test norm(g - gRef) ≤ 1e-7
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
