#=
    Check the accruacy of the PEtab importer for a simple linear ODE;
    x' = a - bx + cy; x(0) = 0
    y' = bx - cy - dy;  y(0) = 0
    with pre-equilibrium condition, where the steady-state can be solved for analytically.
 =#

using PEtab, OrdinaryDiffEqRosenbrock, SciMLSensitivity, ForwardDiff, LinearAlgebra, Sundials,
    Test

include(joinpath(@__DIR__, "common.jl"))

function solve_algebraic_ss(
        model::PEtabModel, solver, tol::Float64, a::T1, b::T1, c::T1, d::T1
    ) where {T1 <: Real}
    ofun = ODEFunction(
        model.sys_mutated, first.(model.speciemap), first.(model.parametermap), jac = true
    )
    oprob = ODEProblem(
        ofun, last.(model.speciemap), (0.0, 9.7), last.(model.parametermap)
    )
    oprob = remake(
        oprob, p = convert.(eltype(a), oprob.p), u0 = convert.(eltype(a), oprob.u0)
    )
    sols = Array{ODESolution, 1}(undef, 2)
    oprob.p[1], oprob.p[5], oprob.p[6], oprob.p[3] = a, b, c, d
    oprob.u0[1] = a / b + (a * c) / (b * d) # x0
    oprob.u0[2] = a / d # y0

    oprob.p[2] = 2.0 # a_scale
    sols[1] = solve(oprob, solver, abstol = tol, reltol = tol)
    oprob.p[2] = 0.5 # a_scale
    sols[2] = solve(oprob, solver, abstol = tol, reltol = tol)
    return sols
end

function nllh_algebraic_ss(x, model::PEtabModel, solver, tol)
    a, b, c, d = x
    petab_tables = model.petab_tables
    petab_measurements = PEtab.PEtabMeasurements(
        petab_tables[:measurements], petab_tables[:observables]
    )

    sols = solve_algebraic_ss(model, solver, tol, a, b, c, d)
    nllh = 0.0
    for i in eachindex(petab_measurements.time)
        y_obs = petab_measurements.measurements[i]
        t = petab_measurements.time[i]
        if petab_measurements.simulation_condition_id[i] == :double
            y_model = sols[1](t)[1]
        else
            y_model = sols[2](t)[2]
        end
        nllh += log(0.04) + 0.5 * log(2 * pi) + 0.5 * ((y_obs - y_model) / 0.04)^2
    end
    return nllh
end

function test_odesolver(model::PEtabModel, osolver::ODESolver, ss_solver::SteadyStateSolver)
    prob = PEtabODEProblem(model; odesolver = osolver, ss_solver = ss_solver)
    # a, b, c, d
    parameters_test = reshape(
        [
            1.0, 2.0, 3.0, 4.0,
            0.1, 0.2, 0.3, 0.4,
            4.0, 3.0, 2.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            2.5, 7.0, 3.0, 3.0,
        ], (4, 5)
    )
    for i in 1:5
        a, b, c, d = parameters_test[:, i]
        x = [a, b, c, d]
        sols = solve_all_conditions(
            x, prob, osolver.solver; abstol = osolver.abstol, reltol = osolver.reltol,
            save_observed_t = true
        )
        algebraic_sols = solve_algebraic_ss(
            model, osolver.solver, osolver.abstol, a, b, c, d
        )
        sqdiff = 0.0
        for (i, cid) in pairs(prob.model_info.simulation_info.conditionids[:experiment])
            sol_numeric = sols[cid]
            sol_algebraic = algebraic_sols[i]
            sqdiff += sum((Array(sol_numeric) - Array(sol_algebraic(sol_numeric.t))) .^ 2)
        end
        @test sqdiff ≤ 1.0e-6
    end
    return
end

function test_nllh_grad_hess(
        model::PEtabModel, osolver::ODESolver, ss_solver::SteadyStateSolver
    )::Nothing
    algebraic_nllh = (x) -> nllh_algebraic_ss(x, model, osolver.solver, osolver.abstol)
    # Testing for a random vector
    x = [10.42427591070766, 11.736135149066968, 17.817573961855622, 3.818133980515257]

    nllh_ref = algebraic_nllh(x)
    grad_ref = ForwardDiff.gradient(algebraic_nllh, x)
    hess_ref = ForwardDiff.hessian(algebraic_nllh, x)

    nllh = _compute_nllh(x, model, osolver; ss_solver = ss_solver)
    @test nllh ≈ nllh_ref atol = 1.0e-3

    g = _compute_grad(x, model, :ForwardDiff, osolver; ss_solver = ss_solver)
    @test all(.≈(g, grad_ref; atol = 1.0e-3))
    g = _compute_grad(x, model, :ForwardEquations, osolver; ss_solver = ss_solver)
    @test all(.≈(g, grad_ref; atol = 1.0e-3))
    # Here we want to test things also run with CVODE_BDF
    tmp = osolver.solver
    osolver.solver = CVODE_BDF()
    g = _compute_grad(
        x, model, :ForwardEquations, osolver; ss_solver = ss_solver,
        sensealg = ForwardSensitivity()
    )
    @test all(.≈(g, grad_ref; atol = 1.0e-3))
    osolver.solver = tmp
    # Want to test all adjoint combinations with ss-simulations
    g = _compute_grad(
        x, model, :Adjoint, osolver; ss_solver = ss_solver,
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
    )
    @test all(.≈(g, grad_ref; atol = 1.0e-3))
    g = _compute_grad(
        x, model, :Adjoint, osolver; ss_solver = ss_solver,
        sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))
    )
    @test all(.≈(g, grad_ref; atol = 1.0e-3))
    g = _compute_grad(
        x, model, :Adjoint, osolver; ss_solver = ss_solver,
        sensealg = GaussAdjoint(autojacvec = ReverseDiffVJP(true))
    )
    g = _compute_grad(
        x, model, :Adjoint, osolver; ss_solver = ss_solver,
        sensealg = GaussAdjoint(autojacvec = ReverseDiffVJP(true))
    )
    @test all(.≈(g, grad_ref; atol = 1.0e-3))

    H = _compute_hess(x, model, :ForwardDiff, osolver; ss_solver = ss_solver)
    @test all(.≈(H, hess_ref; atol = 1.0e-3))
    return nothing
end

model = PEtabModel(joinpath(@__DIR__, "analytic_ss", "Test_model3.yaml"))

@testset "ODE solver Simulate wrms termination" begin
    ss_solver = SteadyStateSolver(
        :Simulate, termination_check = :wrms, abstol = 1.0e-12, reltol = 1.0e-10
    )
    test_odesolver(
        model, ODESolver(Rodas4P(), abstol = 1.0e-12, reltol = 1.0e-12), ss_solver
    )
end

@testset "ODE solver Simulate Newton SS termination" begin
    ss_solver = SteadyStateSolver(
        :Simulate, termination_check = :Newton, abstol = 1.0e-12, reltol = 1.0e-10
    )
    test_odesolver(
        model, ODESolver(Rodas4P(), abstol = 1.0e-12, reltol = 1.0e-12), ss_solver
    )
end

@testset "Cost gradient and hessian" begin
    ss_solver = SteadyStateSolver(:Simulate, abstol = 1.0e-12, reltol = 1.0e-10)
    osolver = ODESolver(
        Rodas5P(), abstol = 1.0e-12, reltol = 1.0e-12, maxiters = Int(1.0e5)
    )
    test_nllh_grad_hess(model, osolver, ss_solver)
end

@testset "grad residuals" begin
    test_grad_residuals(model, ODESolver(Rodas5P(), abstol = 1.0e-9, reltol = 1.0e-9))
end
