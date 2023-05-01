using PEtab
using YAML
using Test
using OrdinaryDiffEq
using Sundials
using SciMLSensitivity
using CSV
using DataFrames
using ForwardDiff
using LinearAlgebra
using FiniteDifferences


function testReferenceCase(testCase::String; testGradient=true)

    @info "Test case $testCase"
    pathYAML = joinpath(@__DIR__, "PEtab_test_suite", testCase, "_" * testCase * ".yaml")
    pathReferenceValues = joinpath(@__DIR__, "PEtab_test_suite", testCase, "_" * testCase * "_solution.yaml")

    petabModel = readPEtabModel(pathYAML, verbose=false, forceBuildJuliaFiles=false)
    petabProblem = setupPEtabODEProblem(petabModel, getODESolverOptions(Rodas5P()), verbose=false)

    cost = petabProblem.computeCost(petabProblem.θ_nominalT)
    chi2 = petabProblem.computeChi2(petabProblem.θ_nominalT)
    simulatedValues = petabProblem.computeSimulatedValues(petabProblem.θ_nominalT)

    referenceYAML = YAML.load_file(pathReferenceValues)
    pathSimulations = joinpath(dirname(pathReferenceValues), referenceYAML["simulation_files"][1])
    costRef, tolCost = -1 * referenceYAML["llh"], referenceYAML["tol_llh"]
    chi2Ref, tolChi2 = referenceYAML["chi2"], referenceYAML["tol_chi2"]
    simulatedValuesRef, tolSimulations = CSV.read(pathSimulations, DataFrame)[!, :simulation], referenceYAML["tol_simulations"]

    @test cost ≈ costRef atol=tolCost
    @test chi2 ≈ chi2Ref atol=tolChi2
    @test all(abs.(simulatedValues .- simulatedValuesRef) .≤ tolSimulations)

    if testGradient == true
        gRef = FiniteDifferences.grad(central_fdm(5, 1), petabProblem.computeCost, petabProblem.θ_nominalT)[1]
        g = petabProblem.computeGradient(petabProblem.θ_nominalT)
        @test norm(g - gRef) ≤ 1e-7
    end
end


@testset "PEtab test-suite" begin 
    testReferenceCase("0001")
    testReferenceCase("0002")
    testReferenceCase("0003")
    testReferenceCase("0004")
    testReferenceCase("0005")
    testReferenceCase("0006")
    testReferenceCase("0007")
    testReferenceCase("0008")
    testReferenceCase("0009")
    testReferenceCase("0010")
    testReferenceCase("0011")
    testReferenceCase("0012")
    testReferenceCase("0013")
    testReferenceCase("0014")
    testReferenceCase("0015")
    testReferenceCase("0016")
    testReferenceCase("0017")
    testReferenceCase("0018")
end
