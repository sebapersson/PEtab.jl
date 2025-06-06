using CSV, DataFrames, FiniteDifferences, PEtab, Test, YAML

function check_test_case(case::String; test_grad::Bool=true)
    @info "Test case $case"
    path_yaml = joinpath(@__DIR__, "petab_testsuite", case, "_$(case).yaml")
    path_ref = joinpath(@__DIR__, "petab_testsuite", case, "_$(case)_solution.yaml")

    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model; ss_solver = SteadyStateSolver(:Simulate, abstol=1e-12, reltol=1e-10))
    x = get_x(prob)

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
    for i in 1:18
        test_case = i < 10 ? "000$(i)" : "00$(i)"
        check_test_case(test_case)
    end
end
