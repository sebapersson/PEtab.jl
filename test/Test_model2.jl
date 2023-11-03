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


function test_ode_solver_test_model2(petab_model::PEtabModel, solverOptions)

    # Set values to PeTab file values
    experimental_conditions_file, measurementDataFile, parameterDataFile, observables_dataFile = read_petab_files(petab_model)
    measurementData = process_measurements(measurementDataFile, observables_dataFile)
    paramData = process_parameters(parameterDataFile)
    θ_indices = compute_θ_indices(paramData, measurementData, petab_model)
    simulation_info = process_simulationinfo(petab_model, measurementData, sensealg=nothing)
    set_parameters_to_file_values!(petab_model.parameter_map, petab_model.state_map, paramData)

    # Parameter values where to teast accuracy. Each column is a alpha, beta, gamma and delta
    u0 = [8.0, 4.0]
    parametersTest = reshape([2.0, 3.0,
                              1.0, 2.0,
                              1.0, 0.4,
                              4.0, 3.0,
                              0.01, 0.02], (2, 5))

    for i in 1:5

        alpha, beta = parametersTest[:, i]
        # Set parameter values for ODE
        petab_model.parameter_map[1] = Pair(petab_model.parameter_map[1].first, alpha)
        petab_model.parameter_map[2] = Pair(petab_model.parameter_map[2].first, beta)
        prob = ODEProblem(petab_model.system, petab_model.state_map, (0.0, 5e3), petab_model.parameter_map, jac=true)
        prob = remake(prob, p = convert.(Float64, prob.p), u0 = convert.(Float64, prob.u0))
        θ_dynamic = get_file_ode_values(petab_model)[1:2]

        ss_options = SteadyStateSolver(:Simulate, abstol = solverOptions.abstol / 100.0, reltol = solverOptions.reltol / 100.0)
        # Solve ODE system
        petab_ODESolver_cache = PEtabODESolverCache(:nothing, :nothing, petab_model, simulation_info, θ_indices, nothing)
        θ_dynamic = [alpha, beta]
        ode_sols, success = solve_ODE_all_conditions(prob, petab_model, θ_dynamic, petab_ODESolver_cache, simulation_info, θ_indices, solverOptions, ss_options)
        ode_sol = ode_sols[simulation_info.experimental_condition_id[1]]

        # Compare against analytical solution
        sqDiff = 0.0
        for t in ode_sol.t
            solAnalytic = [u0[1]*exp(alpha*t), u0[2]*exp(beta*t)]
            sqDiff += sum((ode_sol(t)[1:2] - solAnalytic).^2)
        end

        @test sqDiff ≤ 1e-6
    end
end


function compute_costAnalyticTestModel2(paramVec)

    u0 = [8.0, 4.0]
    alpha, beta = paramVec[1:2]
    measurementData = CSV.File(joinpath(@__DIR__, "Test_model2/measurementData_Test_model2.tsv"))

    # Extract correct parameter for observation i and compute loglik
    loglik = 0.0
    for i in eachindex(measurementData)

        # Specs for observation i
        obsID = measurementData[:observableId][i]
        noiseID = measurementData[:noiseParameters][i]
        y_obs = measurementData[:measurement][i]
        t = measurementData[:time][i]
        # Extract correct sigma
        if noiseID == "sd_sebastian_new"
            sigma = paramVec[3]
        elseif noiseID == "sd_damiano_new"
            sigma = paramVec[4]
        end

        sol = [u0[1]*exp(alpha*t), u0[2]*exp(beta*t)]
        if obsID == "sebastian_measurement"
            y_model = sol[1]
        elseif obsID == "damiano_measurement"
            y_model = sol[2]
        end

        loglik += log(sigma) + 0.5*log(2*pi) + 0.5 * ((y_obs - y_model) / sigma)^2
    end

    return loglik
end


function test_cost_gradient_hessian_test_model2(petab_model::PEtabModel, solverOptions)

    # Cube with random parameter values for testing
    cube = CSV.File(joinpath(@__DIR__, "Test_model2", "Julia_model_files", "CubeTest_model2.csv"))

    for i in 1:1

        p = Float64.(collect(cube[i]))
        referenceCost = compute_costAnalyticTestModel2(p)
        referenceGradient = ForwardDiff.gradient(compute_costAnalyticTestModel2, p)
        referenceHessian = ForwardDiff.hessian(compute_costAnalyticTestModel2, p)

        # Test both the standard and Zygote approach to compute the cost
        cost = _testCostGradientOrHessian(petab_model, solverOptions, p, compute_cost=true, cost_method=:Standard)
        @test cost ≈ referenceCost atol=1e-3
        costZygote = _testCostGradientOrHessian(petab_model, solverOptions, p, compute_cost=true, cost_method=:Zygote)
        @test costZygote ≈ referenceCost atol=1e-3

        # Test all gradient combinations. Note we test sensitivity equations with and without autodiff
        gradientForwardDiff = _testCostGradientOrHessian(petab_model, solverOptions, p, compute_gradient=true, gradient_method=:ForwardDiff)
        @test norm(gradientForwardDiff - referenceGradient) ≤ 1e-2
        gradientZygote = _testCostGradientOrHessian(petab_model, solverOptions, p, compute_gradient=true, gradient_method=:Zygote, sensealg=ForwardDiffSensitivity())
        @test norm(gradientZygote - referenceGradient) ≤ 1e-2
        gradient_adjoint = _testCostGradientOrHessian(petab_model, solverOptions, p, compute_gradient=true, gradient_method=:Adjoint, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(false)))
        @test norm(normalize(gradient_adjoint) - normalize((referenceGradient))) ≤ 1e-2
        gradientForward1 = _testCostGradientOrHessian(petab_model, solverOptions, p, compute_gradient=true, gradient_method=:ForwardEquations, sensealg=:ForwardDiff)
        @test norm(gradientForward1 - referenceGradient) ≤ 1e-2
        gradientForward2 = _testCostGradientOrHessian(petab_model, solverOptions, p, compute_gradient=true, gradient_method=:ForwardEquations, sensealg=ForwardDiffSensitivity())
        @test norm(gradientForward2 - referenceGradient) ≤ 1e-2

        # Testing "exact" hessian via autodiff
        hessian = _testCostGradientOrHessian(petab_model, solverOptions, p, compute_hessian=true, gradient_method=:ForwardDiff, hessian_method=:ForwardDiff)
        @test norm(hessian - referenceHessian) ≤ 1e-2
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
