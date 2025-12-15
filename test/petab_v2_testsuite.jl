using CSV, DataFrames, PEtab, Test, YAML

function test_v2(test_case::String)
    @info "Test case $(test_case)"
    path_yaml = joinpath(@__DIR__, "petab_v2_testsuite", test_case, "_$(test_case).yaml")
    path_ref = joinpath(@__DIR__, "petab_v2_testsuite", test_case, "_$(test_case)_solution.yaml")

    ss_solver = SteadyStateSolver(:Simulate, abstol=1e-12, reltol=1e-10)
    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model; ss_solver = ss_solver)
    x = get_x(prob)

    nllh = prob.nllh(x; prior = false)
    nllh_prior = prob.nllh(x; prior = true)
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
    if haskey(reference_yaml, "unnorm_log_posterior")
        nllh_prior_ref = reference_yaml["unnorm_log_posterior"]
        @test nllh_prior ≈ -1 * nllh_prior_ref atol=nllh_tol
    end
end

completed_tests = ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009",
                   "0010", "0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018",
                   "0020", "0021", "0022", "0023", "0024", "0025"]
for test_case in completed_tests
    test_v2(test_case)
end
