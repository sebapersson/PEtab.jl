#=
    Check the accruacy of the PeTab importer for a simple linear ODE;
    x' = a - bx + cy; x(0) = 0
    y' = bx - cy - dy;  y(0) = 0
    where the model has a pre-equilibrium condition. That is he simulated data for
    this ODE model is generated by starting from the steady state;
    x* = a / b + ( a * c ) / ( b * d )
    y* = a / d
    and when computing the cost in the PeTab importer the model is first simualted
    to a steady state, and then the mian simulation matched against data is
    performed.
    This test compares the ODE-solution, cost, gradient and hessian when
    i) solving the ODE using the SS-equations as initial condition, and ii) when
    first simulating the model to the steady state.
    Accruacy of both the hessian and gradient are strongly dependent on the tolerances
    used in the TerminateSteadyState callback.
 =#


using PEtab
using Test
using OrdinaryDiffEq
using Zygote
using SciMLSensitivity
using CSV
using ForwardDiff
using LinearAlgebra
using Sundials
using Printf

import PEtab: read_petab_files, process_measurements, process_parameters, compute_θ_indices, process_simulationinfo, set_parameters_to_file_values!
import PEtab: _change_simulation_condition!, solve_ODE_all_conditions, _get_steady_state_solver


include(joinpath(@__DIR__, "Common.jl"))


function get_algebraic_ss(petab_model::PEtabModel, solver, tol::Float64, a::T1, b::T1, c::T1, d::T1) where T1<:Real

    # ODE solution with algebraically computed initial values (instead of ss pre-simulation)
    ode_problem = ODEProblem(petab_model.system_mutated, petab_model.state_map, (0.0, 9.7), petab_model.parameter_map, jac=true)
    ode_problem = remake(ode_problem, p = convert.(eltype(a), ode_problem.p), u0 = convert.(eltype(a), ode_problem.u0))
    sol_array = Array{ODESolution, 1}(undef, 2)

    # Set model parameter values to ensure initial steady state
    ode_problem.p[4], ode_problem.p[2], ode_problem.p[1], ode_problem.p[5] = a, b, c, d
    ode_problem.u0[1] = a / b + ( a * c ) / ( b * d ) # x0
    ode_problem.u0[2] = a / d # y0

    ode_problem.p[3] = 2.0 # a_scale
    sol_array[1] = solve(ode_problem, solver, abstol=tol, reltol=tol)
    ode_problem.p[3] = 0.5 # a_scale
    sol_array[2] = solve(ode_problem, solver, abstol=tol, reltol=tol)

    return sol_array
end


function compute_costAlgebraic(param_vec, petab_model, solver, tol)

    a, b, c, d = param_vec

    experimental_conditions_file, measurements_file, parameters_file, observables_file = read_petab_files(petab_model)
    measurement_info = process_measurements(measurements_file, observables_file)

    sol_arrayAlg = get_algebraic_ss(petab_model, solver, tol, a, b, c, d)
    loglik = 0.0
    for i in eachindex(measurement_info.time)
        y_obs = measurement_info.measurement[i]
        t = measurement_info.time[i]
        if measurement_info.simulation_condition_id[i] == :double
            y_model = sol_arrayAlg[1](t)[1]
        else
            y_model = sol_arrayAlg[2](t)[2]
        end
        sigma = 0.04
        loglik += log(sigma) + 0.5*log(2*pi) + 0.5 * ((y_obs - y_model) / sigma)^2
    end

    return loglik
end


function test_ode_solver_test_model3(petab_model::PEtabModel, ode_solver::ODESolver, ss_options::SteadyStateSolver)

    # Set values to PeTab file values
    experimental_conditions_file, measurements_file, parameters_file, observables_file = read_petab_files(petab_model)
    measurement_info = process_measurements(measurements_file, observables_file)
    parameter_info = process_parameters(parameters_file)
    set_parameters_to_file_values!(petab_model.parameter_map, petab_model.state_map, parameter_info)
    θ_indices = compute_θ_indices(parameter_info, measurement_info, petab_model)

    # Extract experimental conditions for simulations
    simulation_info = process_simulationinfo(petab_model, measurement_info, sensealg=nothing)

    # Parameter values where to teast accuracy. Each column is a alpha, beta, gamma and delta
    # a, b, c, d
    parameters_test = reshape([1.0, 2.0, 3.0, 4.0,
                              0.1, 0.2, 0.3, 0.4,
                              4.0, 3.0, 2.0, 1.0,
                              1.0, 1.0, 1.0, 1.0,
                              2.5, 7.0, 3.0, 3.0,], (4, 5))

    for i in 1:5
        a, b, c, d = parameters_test[:, i]
        # Set parameter values for ODE
        petab_model.parameter_map[1] = Pair(petab_model.parameter_map[1].first, c)
        petab_model.parameter_map[2] = Pair(petab_model.parameter_map[2].first, b)
        petab_model.parameter_map[4] = Pair(petab_model.parameter_map[4].first, a)
        petab_model.parameter_map[5] = Pair(petab_model.parameter_map[5].first, d)

        prob = ODEProblem(petab_model.system_mutated, petab_model.state_map, (0.0, 9.7), petab_model.parameter_map, jac=true)
        prob = remake(prob, p = convert.(Float64, prob.p), u0 = convert.(Float64, prob.u0))
        θ_dynamic = get_file_ode_values(petab_model)[1:4]
        petab_ODESolver_cache = PEtabODESolverCache(:nothing, :nothing, petab_model, simulation_info, θ_indices, nothing)
        _ss_options = _get_steady_state_solver(ss_options, prob, ss_options.abstol, ss_options.reltol, ss_options.maxiters)

        # Solve ODE system
        ode_sols, success = solve_ODE_all_conditions(prob, petab_model, θ_dynamic, petab_ODESolver_cache, simulation_info, θ_indices, ode_solver, _ss_options)
        # Solve ODE system with algebraic intial values
        algebraicODESolutions = get_algebraic_ss(petab_model, ode_solver.solver, ode_solver.abstol, a, b, c, d)

        # Compare against analytical solution
        sqdiff = 0.0
        for i in eachindex(simulation_info.experimental_condition_id)
            sol_numeric = ode_sols[simulation_info.experimental_condition_id[i]]
            sol_algebraic = algebraicODESolutions[i]
            sqdiff += sum((Array(sol_numeric) - Array(sol_algebraic(sol_numeric.t))).^2)
        end

        @test sqdiff ≤ 1e-6
    end
end


function test_cost_gradient_hessian_test_model3(petab_model::PEtabModel, ode_solver::ODESolver, ss_options::SteadyStateSolver)

    _compute_costAlgebraic = (pArg) -> compute_costAlgebraic(pArg, petab_model, ode_solver.solver, ode_solver.abstol)

    cube = CSV.File(joinpath(@__DIR__, "Test_model3", "Julia_model_files", "CubeTest_model3.csv"))

    for i in 1:1

        p = Float64.(collect(cube[i]))

        reference_cost = _compute_costAlgebraic(p)
        reference_gradient = ForwardDiff.gradient(_compute_costAlgebraic, p)
        reference_hessian = ForwardDiff.hessian(_compute_costAlgebraic, p)

        # Test both the standard and Zygote approach to compute the cost
        cost = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_cost=true, cost_method=:Standard, ss_options=ss_options)
        @test cost ≈ reference_cost atol=1e-3
        cost_zygote = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_cost=true, cost_method=:Zygote, ss_options=ss_options)
        @test cost_zygote ≈ reference_cost atol=1e-3

        # Test all gradient combinations. Note we test sensitivity equations with and without autodiff
        gradient_forwarddiff = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_gradient=true, gradient_method=:ForwardDiff, ss_options=ss_options)
        @test norm(gradient_forwarddiff - reference_gradient) ≤ 1e-2
        gradient_zygote = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_gradient=true, gradient_method=:Zygote, sensealg=ForwardDiffSensitivity(), ss_options=ss_options)
        @test norm(gradient_zygote - reference_gradient) ≤ 1e-2
        gradient_forward1 = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_gradient=true, gradient_method=:ForwardEquations, sensealg=:ForwardDiff, ss_options=ss_options)
        @test norm(gradient_forward1 - reference_gradient) ≤ 1e-2
        gradient_forward2 = _test_cost_gradient_hessian(petab_model, ODESolver(CVODE_BDF(), abstol=1e-12, reltol=1e-12), p, compute_gradient=true, gradient_method=:ForwardEquations, sensealg=ForwardSensitivity(), ss_options=ss_options)
        @test norm(gradient_forward2 - reference_gradient) ≤ 1e-2

        # Must test different adjoints
        gradient_adjoint = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_gradient=true, gradient_method=:Adjoint, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(false)), ss_options=ss_options)
        @test norm(normalize(gradient_adjoint) - normalize((reference_gradient))) ≤ 1e-2
        gradient_adjoint = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_gradient=true, gradient_method=:Adjoint, sensealg=GaussAdjoint(autojacvec=ReverseDiffVJP(false)), ss_options=ss_options)
        @test norm(normalize(gradient_adjoint) - normalize((reference_gradient))) ≤ 1e-2
        gradient_adjoint = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_gradient=true, gradient_method=:Adjoint, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false)), ss_options=ss_options)
        @test norm(normalize(gradient_adjoint) - normalize((reference_gradient))) ≤ 1e-2
        gradient_adjoint = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_gradient=true, gradient_method=:Adjoint, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false)), ss_options=ss_options, sensealg_ss=SteadyStateAdjoint())
        @test norm(normalize(gradient_adjoint) - normalize((reference_gradient))) ≤ 1e-2

        # Testing "exact" hessian via autodiff
        hessian = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_hessian=true, hessian_method=:ForwardDiff, ss_options=ss_options)
        @test norm(hessian - reference_hessian) ≤ 1e-2
    end

    return true
end


petab_model = PEtabModel(joinpath(@__DIR__, "Test_model3", "Test_model3.yaml"), build_julia_files=true, write_to_file=true)

@testset "ODE solver Simulate wrms termination" begin
    ss_optionsTest1 = SteadyStateSolver(:Simulate, check_simulation_steady_state=:wrms, abstol=1e-12, reltol=1e-10)
    test_ode_solver_test_model3(petab_model, ODESolver(Rodas4P(), abstol=1e-12, reltol=1e-12), ss_optionsTest1)
end

@testset "ODE solver Simulate Newton SS termination" begin
    ss_optionsTest2 = SteadyStateSolver(:Simulate, check_simulation_steady_state=:Newton, abstol=1e-12, reltol=1e-10)
    test_ode_solver_test_model3(petab_model, ODESolver(Rodas4P(), abstol=1e-12, reltol=1e-12), ss_optionsTest2)
end

@testset "Cost gradient and hessian" begin
    ss_optionsTest3 = SteadyStateSolver(:Simulate, check_simulation_steady_state=:wrms, abstol=1e-12, reltol=1e-10)
    test_cost_gradient_hessian_test_model3(petab_model, ODESolver(Rodas5P(), abstol=1e-12, reltol=1e-12, maxiters=Int(1e5)), ss_optionsTest3)
end

@testset "Gradient of residuals" begin
    check_gradient_residuals(petab_model, ODESolver(Rodas4P(), abstol=1e-9, reltol=1e-9))
end

# As the model has a steady-state this can be used to test the show function
odesolver = @sprintf("%s", ODESolver(QNDF(), abstol=1e-5, reltol=1e-8))
@test odesolver == "ODESolver with ODE solver QNDF. Options (abstol, reltol, maxiters) = (1.0e-05, 1.0e-08, 1.0e+04)"
ssopt = @sprintf("%s", SteadyStateSolver(:Simulate))
model = @sprintf("%s", PEtabModel(joinpath(@__DIR__, "Test_model3", "Test_model3.yaml"), build_julia_files=true, write_to_file=false))
@test model[1:75] == "PEtabModel for model Test_model3. ODE-system has 2 states and 6 parameters."
prob = @sprintf("%s", PEtabODEProblem(petab_model, verbose=false))
@test prob == "PEtabODEProblem for Test_model3. ODE-states: 2. Parameters to estimate: 4 where 4 are dynamic.\n---------- Problem settings ----------\nGradient method : ForwardDiff\nHessian method : ForwardDiff\n--------- ODE-solver settings --------\nCost Rodas5P. Options (abstol, reltol, maxiters) = (1.0e-08, 1.0e-08, 1.0e+04)\nGradient Rodas5P. Options (abstol, reltol, maxiters) = (1.0e-08, 1.0e-08, 1.0e+04)\n--------- SS solver settings ---------\nCost Simulate. Option wrms with (abstol, reltol) = (1.0e-10, 1.0e-10)\nGradient Simulate. Options wrms with (abstol, reltol) = (1.0e-10, 1.0e-10)"
