using CSV, DataFrames, FiniteDifferences, PEtab, Test, YAML

function test_v2(test_case::String; test_gradient::Bool = true)
    @info "Test case $(test_case)"
    path_yaml = joinpath(
        @__DIR__, "petab_v2_testsuite", test_case, "_$(test_case).yaml"
    )
    path_ref = joinpath(
        @__DIR__, "petab_v2_testsuite", test_case, "_$(test_case)_solution.yaml"
    )

    ss_solver = SteadyStateSolver(:Simulate, abstol = 1.0e-12, reltol = 1.0e-10)
    model = PEtabModel(path_yaml)
    prob = PEtabODEProblem(model; ss_solver = ss_solver, gradient_method = :ForwardDiff)
    x = get_x(prob)

    nllh = prob.nllh(x; prior = false)
    nllh_prior = prob.nllh(x; prior = true)
    χ₂ = prob.chi2(x)
    simvals = prob.simulated_values(x)

    reference_yaml = YAML.load_file(path_ref)
    path_simulations = joinpath(dirname(path_ref), reference_yaml["simulation_files"][1])
    nllh_ref, nllh_tol = -1 * reference_yaml["llh"], reference_yaml["tol_llh"]
    χ₂_ref, χ₂_tol = reference_yaml["chi2"], reference_yaml["tol_chi2"]
    simvals_ref = CSV.read(path_simulations, DataFrame)[!, :simulation]
    simvals_tol = reference_yaml["tol_simulations"]

    @test nllh ≈ nllh_ref atol = nllh_tol
    @test χ₂ ≈ χ₂_ref atol = χ₂_tol
    @test all(.≈(simvals, simvals_ref; atol = simvals_tol))
    if haskey(reference_yaml, "unnorm_log_posterior")
        nllh_prior_ref = reference_yaml["unnorm_log_posterior"]
        @test nllh_prior ≈ -1 * nllh_prior_ref atol = nllh_tol
    end

    return if test_gradient && test_case != "0023"
        # Need to avoid values on the edge of the support for finite differencing to work
        if :p_log_uniform in keys(x)
            x.p_log_uniform = 4.0
        end

        prob_forward_eqs = PEtabODEProblem(
            model; ss_solver = ss_solver,
            gradient_method = :ForwardEquations, sensealg = :ForwardDiff
        )

        grad_ref = FiniteDifferences.grad(central_fdm(5, 1), prob.nllh, x)[1]
        grad1 = prob.grad(x)
        grad2 = prob_forward_eqs.grad(x)
        @test all(.≈(grad1, grad_ref, atol = 1.0e-3))
        @test all(.≈(grad2, grad_ref, atol = 1.0e-3))
    end
end


supported_tests = [
    "0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0011",
    "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0020", "0021", "0022", "0023",
    "0024", "0025", "0026", "0027", "0028", "0029", "0030", "0031",
]

@testset "V2 test suite" begin
    for test_case in supported_tests
        test_v2(test_case)
    end
end
