#=
    Check the accruacy of the PeTab importer for a simple linear ODE;
        s' = alpha*s; s(0) = 8.0 -> s(t) = 8.0 * exp(alpha*t)
        d' = beta*d;  d(0) = 4.0 -> d(t) = 4.0 * exp(beta*t)
    This ODE is solved analytically, and using the analytical solution the accuracy of
    the ODE solver, cost function, gradient and hessian of the PeTab importer is checked.
    The accuracy of the optimizers is further checked.
    The measurment data is avaible in test/Test_model2/
 =#

using PEtab
using Test
using OrdinaryDiffEq
using Zygote
using SciMLSensitivity
using CSV
using ForwardDiff
using LinearAlgebra

import PEtab: read_petab_files, process_measurements, process_parameters, compute_θ_indices, process_simulationinfo, set_parameters_to_file_values!
import PEtab: _change_simulation_condition!, solve_ODE_all_conditions, PEtabODESolverCache, PEtabODESolverCache


include(joinpath(@__DIR__, "Common.jl"))


function test_ode_solver_test_model2(petab_model::PEtabModel, ode_solver::ODESolver)

    # Set values to PeTab file values
    experimental_conditions_file, measurements_file, parameters_file, observables_file = read_petab_files(petab_model)
    measurement_info = process_measurements(measurements_file, observables_file)
    parameter_info = process_parameters(parameters_file)
    θ_indices = compute_θ_indices(parameter_info, measurement_info, petab_model)
    simulation_info = process_simulationinfo(petab_model, measurement_info, sensealg=nothing)
    set_parameters_to_file_values!(petab_model.parameter_map, petab_model.state_map, parameter_info)

    # Parameter values where to teast accuracy. Each column is a alpha, beta, gamma and delta
    u0 = [8.0, 4.0]
    parameters_test = reshape([2.0, 3.0,
                              1.0, 2.0,
                              1.0, 0.4,
                              4.0, 3.0,
                              0.01, 0.02], (2, 5))

    for i in 1:5

        alpha, beta = parameters_test[:, i]
        # Set parameter values for ODE
        petab_model.parameter_map[1] = Pair(petab_model.parameter_map[1].first, alpha)
        petab_model.parameter_map[2] = Pair(petab_model.parameter_map[2].first, beta)
        prob = ODEProblem(petab_model.system_mutated, petab_model.state_map, (0.0, 5e3), petab_model.parameter_map, jac=true)
        prob = remake(prob, p = convert.(Float64, prob.p), u0 = convert.(Float64, prob.u0))
        θ_dynamic = get_file_ode_values(petab_model)[1:2]

        ss_options = SteadyStateSolver(:Simulate, abstol = ode_solver.abstol / 100.0, reltol = ode_solver.reltol / 100.0)
        # Solve ODE system
        petab_ODESolver_cache = PEtabODESolverCache(:nothing, :nothing, petab_model, simulation_info, θ_indices, nothing)
        θ_dynamic = [alpha, beta]
        ode_sols, success = solve_ODE_all_conditions(prob, petab_model, θ_dynamic, petab_ODESolver_cache, simulation_info, θ_indices, ode_solver, ss_options)
        ode_sol = ode_sols[simulation_info.experimental_condition_id[1]]

        # Compare against analytical solution
        sqdiff = 0.0
        for t in ode_sol.t
            solAnalytic = [u0[1]*exp(alpha*t), u0[2]*exp(beta*t)]
            sqdiff += sum((ode_sol(t)[1:2] - solAnalytic).^2)
        end

        @test sqdiff ≤ 1e-6
    end
end


function analytic_cost_test_model2(param_vec)

    u0 = [8.0, 4.0]
    alpha, beta = param_vec[1:2]
    measurement_info = CSV.File(joinpath(@__DIR__, "Test_model2/measurementData_Test_model2.tsv"))

    # Extract correct parameter for observation i and compute loglik
    loglik = 0.0
    for i in eachindex(measurement_info)

        # Specs for observation i
        obs_id = measurement_info[:observableId][i]
        noise_id = measurement_info[:noiseParameters][i]
        y_obs = measurement_info[:measurement][i]
        t = measurement_info[:time][i]
        # Extract correct sigma
        if noise_id == "sd_sebastian_new"
            sigma = param_vec[3]
        elseif noise_id == "sd_damiano_new"
            sigma = param_vec[4]
        end

        sol = [u0[1]*exp(alpha*t), u0[2]*exp(beta*t)]
        if obs_id == "sebastian_measurement"
            y_model = sol[1]
        elseif obs_id == "damiano_measurement"
            y_model = sol[2]
        end

        loglik += log(sigma) + 0.5*log(2*pi) + 0.5 * ((y_obs - y_model) / sigma)^2
    end

    return loglik
end


function test_cost_gradient_hessian_test_model2(petab_model::PEtabModel, ode_solver::ODESolver)

    # Cube with random parameter values for testing
    cube = CSV.File(joinpath(@__DIR__, "Test_model2", "Julia_model_files", "CubeTest_model2.csv"))

    # For testing block Hessian approach
    prob1 = PEtabODEProblem(petab_model, hessian_method=:BlockForwardDiff,
                            ode_solver=ODESolver(Vern9(), abstol=1e-9, reltol=1e-9))
    prob2 = PEtabODEProblem(petab_model, hessian_method=:ForwardDiff,
                            ode_solver=ODESolver(Vern9(), abstol=1e-9, reltol=1e-9))

    for i in 1:1

        p = Float64.(collect(cube[i]))
        reference_cost = analytic_cost_test_model2(p)
        reference_gradient = ForwardDiff.gradient(analytic_cost_test_model2, p)
        reference_hessian = ForwardDiff.hessian(analytic_cost_test_model2, p)

        # Test both the standard and Zygote approach to compute the cost
        cost = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_cost=true, cost_method=:Standard)
        @test cost ≈ reference_cost atol=1e-3
        cost_zygote = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_cost=true, cost_method=:Zygote)
        @test cost_zygote ≈ reference_cost atol=1e-3

        # Test all gradient combinations. Note we test sensitivity equations with and without autodiff
        gradient_forwarddiff = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_gradient=true, gradient_method=:ForwardDiff)
        @test norm(gradient_forwarddiff - reference_gradient) ≤ 1e-2
        gradient_zygote = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_gradient=true, gradient_method=:Zygote, sensealg=ForwardDiffSensitivity())
        @test norm(gradient_zygote - reference_gradient) ≤ 1e-2
        gradient_adjoint = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_gradient=true, gradient_method=:Adjoint, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(false)))
        @test norm(normalize(gradient_adjoint) - normalize((reference_gradient))) ≤ 1e-2
        gradient_forward1 = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_gradient=true, gradient_method=:ForwardEquations, sensealg=:ForwardDiff)
        @test norm(gradient_forward1 - reference_gradient) ≤ 1e-2
        gradient_forward2 = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_gradient=true, gradient_method=:ForwardEquations, sensealg=ForwardDiffSensitivity())
        @test norm(gradient_forward2 - reference_gradient) ≤ 1e-2

        # Testing "exact" hessian via autodiff
        hessian = _test_cost_gradient_hessian(petab_model, ode_solver, p, compute_hessian=true, gradient_method=:ForwardDiff, hessian_method=:ForwardDiff)
        @test norm(hessian - reference_hessian) ≤ 1e-2

        # Testing block-hessian
        _ = prob1.compute_cost(p)
        _ = prob2.compute_cost(p)
        H1 = prob1.compute_hessian(p)
        H2 = prob2.compute_hessian(p)
        @test norm(H1[1:2] - H2[1:2]) < 1e-2
        @test norm(normalize(diag(H1, 0)[3:end]) - normalize(diag(H2, 0)[3:end])) < 1e-2
    end
end


# Used to check against world-age problem
function create_model_inside_function()
    _petab_model = PEtabModel(joinpath(@__DIR__, "Test_model2", "Test_model2.yaml"), build_julia_files=true, verbose=true)
    return _petab_model
end
petab_model = create_model_inside_function()

@testset "ODE solver" begin
    test_ode_solver_test_model2(petab_model, ODESolver(Vern9(), abstol=1e-9, reltol=1e-9))
end

@testset "Cost gradient and hessian" begin
    test_cost_gradient_hessian_test_model2(petab_model, ODESolver(Vern9(), abstol=1e-15, reltol=1e-15))
end

@testset "Gradient of residuals" begin
    check_gradient_residuals(petab_model, ODESolver(Rodas5P(), abstol=1e-9, reltol=1e-9))
end
