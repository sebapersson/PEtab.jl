using PEtab, Test, OrdinaryDiffEqRosenbrock
import SciMLLogging

# An experiment whose periods after t0 apply different conditions, at different times. The
# model (from test case 0031) has k1 = k2 = 0, so A and B only change at the period starts:
# A + 5 at t = 10 and B + 3 at t = 20.
@testset "V2 experiment with several periods" begin
    path_yaml = joinpath(
        @__DIR__, "petab_v2_regression", "multiple_periods", "problem.yaml"
    )
    model = PEtabModel(path_yaml)
    osolver = ODESolver(
        Rodas5P(), abstol = 1.0e-10, reltol = 1.0e-10, verbose = SciMLLogging.None()
    )
    prob = PEtabODEProblem(model; odesolver = osolver)
    simvals = prob.simulated_values(get_x(prob))
    # Measurement order: obs_a at t = 5, 15, 25, then obs_b at the same times
    @test all(.≈(simvals, [1.0, 6.0, 6.0, 1.0, 1.0, 4.0]; atol = 1.0e-6))
end
