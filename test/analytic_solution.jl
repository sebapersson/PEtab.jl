#=
    Check the accruacy of PEtab importer for a simple linear ODE;
        s' = alpha*s; s(0) = 8.0 -> s(t) = 8.0 * exp(alpha*t)
        d' = beta*d;  d(0) = 4.0 -> d(t) = 4.0 * exp(beta*t)
    This ODE is solved analytically, and using the analytical solution the accuracy of
    the ODE solver, cost function, gradient and hessian is tested
 =#

using PEtab, OrdinaryDiffEqRosenbrock, SciMLSensitivity, ForwardDiff, LinearAlgebra, CSV,
    DataFrames, Sundials, Test

include(joinpath(@__DIR__, "common.jl"))

function test_odesolver(model::PEtabModel, osolver::ODESolver)::Nothing
    prob = PEtabODEProblem(model; odesolver = osolver, verbose = false)
    u0 = [8.0, 4.0]
    parameters_test = reshape([2.0, 3.0,
                               1.0, 2.0,
                               1.0, 0.4,
                               4.0, 3.0,
                               0.01, 0.02], (2, 5))
    for i in 1:5
        α, β = parameters_test[:, i]
        x = [α, β, 0.7, 0.6]
        sols = solve_all_conditions(x, prob, osolver.solver; abstol = osolver.abstol,
                                    reltol = osolver.reltol, save_observed_t = true)
        sol = sols[:model1_data1]
        sqdiff = 0.0
        for (it, t) in pairs(sol.t)
            sol_num = sol[[:sebastian, :damiano]]
            sol_analytic = [u0[1]*exp(α*t), u0[2]*exp(β*t)]
            sqdiff += sum((sol_num[it] - sol_analytic).^2)
        end
        @test sqdiff ≤ 1e-6
    end
    return nothing
end

function analytic_nllh(x)
    u0 = [8.0, 4.0]
    path = joinpath(@__DIR__, "analytic_solution", "measurementData_Test_model2.tsv")
    measurements_df = CSV.read(path, DataFrame)
    nllh = 0.0
    for i in 1:nrow(measurements_df)
        obs_id = measurements_df[i, :observableId]
        noise_id = measurements_df[i, :noiseParameters]
        y_obs = measurements_df[i, :measurement]
        t = measurements_df[i, :time]
        if noise_id == "sd_sebastian_new"
            σ = x[3]
        elseif noise_id == "sd_damiano_new"
            σ = x[4]
        end

        sol = [u0[1]*exp(x[1]*t), u0[2]*exp(x[2]*t)]
        if obs_id == "sebastian_measurement"
            h = sol[1]
        elseif obs_id == "damiano_measurement"
            h = sol[2]
        end

        nllh += log(σ) + 0.5*log(2*pi) + 0.5 * ((y_obs - h) / σ)^2
    end
    return nllh
end

function test_nllh_grad_hess(model::PEtabModel, osolver::ODESolver)::Nothing
    # Testing for a random parameter vector
    x = [5.42427591070766, 6.736135149066968, 1.792669526376284, 0.46272272814894944]

    nllh_ref = analytic_nllh(x)
    grad_ref = ForwardDiff.gradient(analytic_nllh, x)
    hess_ref = ForwardDiff.hessian(analytic_nllh, x)

    nllh = _compute_nllh(x, model, osolver)
    @test nllh ≈ nllh_ref atol=1e-3

    g = _compute_grad(x, model, :ForwardDiff, osolver)
    @test all(.≈(g, grad_ref; atol = 1e-3))
    g = _compute_grad(x, model, :ForwardEquations, osolver)
    @test all(.≈(g, grad_ref; atol = 1e-3))
    tmp = osolver.solver
    osolver.solver = CVODE_BDF()
    g = _compute_grad(x, model, :ForwardEquations, osolver; sensealg = ForwardSensitivity())
    @test all(.≈(g, grad_ref; atol = 1e-3))
    osolver.solver = tmp
    @test all(.≈(g, grad_ref; atol = 1e-3))
    g = _compute_grad(x, model, :Adjoint, osolver;
                      sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(false)))
    @test all(.≈(normalize(g), normalize(grad_ref); atol = 1e-3))

    H = _compute_hess(x, model, :ForwardDiff, osolver)
    @test all(.≈(H, hess_ref; atol = 5e-3))
    H = _compute_hess(x, model, :ForwardDiff, osolver; split = true)
    @test all(.≈(H, hess_ref; atol = 5e-3))
    H = _compute_hess(x, model, :BlockForwardDiff, osolver)
    @test all(.≈(H[1:2, 1:2], hess_ref[1:2, 1:2]; atol = 5e-3))
    @test all(.≈(H[3:4, 3:4], hess_ref[3:4, 3:4]; atol = 5e-3))
    H = _compute_hess(x, model, :BlockForwardDiff, osolver; split = true)
    @test all(.≈(H[1:2, 1:2], hess_ref[1:2, 1:2]; atol = 5e-3))
    @test all(.≈(H[3:4, 3:4], hess_ref[3:4, 3:4]; atol = 5e-3))
    return nothing
end

# Test that we do not have world-problem
function create_model_inside_function()
    return PEtabModel(joinpath(@__DIR__, "analytic_solution", "Test_model2.yaml"),
                      build_julia_files=true, verbose=true, write_to_file = true)
end
model = create_model_inside_function()

@testset "ODE solver" begin
    test_odesolver(model, ODESolver(Rodas5P(), abstol=1e-9, reltol=1e-9))
end

@testset "nllh, grad, and hess" begin
    test_nllh_grad_hess(model, ODESolver(Rodas5P(), abstol=1e-15, reltol=1e-15))
end

@testset "grad residuals" begin
    test_grad_residuals(model, ODESolver(Rodas5P(), abstol=1e-9, reltol=1e-9))
end

# Test if the PEtabModel can also be read from existing model files
model = PEtabModel(joinpath(@__DIR__, "analytic_solution", "Test_model2.yaml"),
                   verbose=true, write_to_file = true)
model2 = PEtabModel(joinpath(@__DIR__, "analytic_solution", "Test_model2.yaml"),
                    build_julia_files=false, verbose=true, write_to_file = false)
prob = PEtabODEProblem(model; verbose = true)
@testset "ODE solver" begin
    test_odesolver(model, ODESolver(Rodas5P(), abstol=1e-9, reltol=1e-9))
end
rm(model.paths[:dirjulia]; recursive = true)
