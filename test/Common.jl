

import PEtab: process_priors, change_ode_parameters!, PriorInfo, transformθ, solve_ode_all_conditions!, compute_GaussNewton_hessian!, PEtabODESolverCache, PEtabODESolverCache, PEtabODEProblemCache


function _test_cost_gradient_hessian(petab_model::PEtabModel,
                                    ode_solver::ODESolver,
                                    p::Vector{Float64};
                                    solverGradientOptions=nothing,
                                    compute_cost::Bool=false,
                                    compute_gradient::Bool=false,
                                    compute_hessian::Bool=false,
                                    cost_method::Symbol=:Standard,
                                    gradient_method::Symbol=:ForwardDiff,
                                    hessian_method::Symbol=:ForwardDiff,
                                    ss_options=nothing,
                                    sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false)),
                                    sensealg_ss=nothing)

    if isnothing(solverGradientOptions)
        solverGradientOptions = deepcopy(ode_solver)
    end

    petab_problem = PEtabODEProblem(petab_model,
                                         ode_solver=ode_solver;
                                         ode_solver_gradient=solverGradientOptions,
                                         cost_method=cost_method,
                                         gradient_method=gradient_method,
                                         hessian_method=hessian_method,
                                         ss_solver=ss_options,
                                         sensealg=sensealg,
                                         sensealg_ss=sensealg_ss,
                                         specialize_level=SciMLBase.FullSpecialize,
                                         verbose=false)

    if compute_cost == true
        return petab_problem.compute_cost(p)
    end

    if compute_gradient == true
        _gradient = zeros(length(p))
        petab_problem.compute_gradient!(_gradient, p)
        return _gradient
    end

    if compute_hessian == true
        _hessian = zeros(length(p), length(p))
        petab_problem.compute_cost(p)
        petab_problem.compute_hessian!(_hessian, p)
        return _hessian
    end
end


function check_gradient_residuals(petab_model::PEtabModel, ode_solver::ODESolver; verbose::Bool=true, custom_parameter_values=nothing)

    # Process PeTab files into type-stable Julia structs
    experimental_conditions_file, measurements_file, parameters_file, observables_file = read_petab_files(petab_model)
    parameter_info = process_parameters(parameters_file, custom_parameter_values=custom_parameter_values)
    measurement_info = process_measurements(measurements_file, observables_file)
    simulation_info = process_simulationinfo(petab_model, measurement_info, sensealg=nothing)

    # Indices for mapping parameter-estimation vector to dynamic, observable and sd parameters correctly when calculating cost
    θ_indices = compute_θ_indices(parameter_info, measurement_info, petab_model)

    # Set model parameter values to those in the PeTab parameter data ensuring correct value of constant parameters
    set_parameters_to_file_values!(petab_model.parameter_map, petab_model.state_map, parameter_info)
    prior_info::PriorInfo = process_priors(θ_indices, parameters_file)

    petab_ODE_cache = PEtabODEProblemCache(:ForwardEquations, :GaussNewton, :ForwardDiff, petab_model, :ForwardDiff, measurement_info, simulation_info, θ_indices, nothing)
    petab_ODESolver_cache = PEtabODESolverCache(:ForwardEquations, :GaussNewton, petab_model, simulation_info, θ_indices, nothing)

    # The time-span 5e3 is overwritten when performing actual forward simulations
    ode_problem = ODEProblem(petab_model.system_mutated, petab_model.state_map, (0.0, 5e3), petab_model.parameter_map, jac=true, sparse=false)
    ode_problem = remake(ode_problem, p = convert.(Float64, ode_problem.p), u0 = convert.(Float64, ode_problem.u0))
    ss_options = SteadyStateSolver(:Simulate, abstol=ode_solver.abstol / 100.0, reltol = ode_solver.reltol / 100.0)
    _ss_options = PEtab._get_steady_state_solver(ss_options, ode_problem, ss_options.abstol, ss_options.reltol, ss_options.maxiters)
    compute_Jacobian!, _ = PEtab.create_hessian_function(:GaussNewton, ode_problem, ode_solver, _ss_options, petab_ODE_cache, petab_ODESolver_cache,
                                         petab_model, simulation_info, θ_indices, measurement_info,
                                         parameter_info, prior_info, nothing, return_jacobian=true)
    compute_sum_residuals = PEtab.create_cost_function(:Standard, ode_problem, ode_solver, _ss_options, petab_ODE_cache, petab_ODESolver_cache,
                                         petab_model, simulation_info, θ_indices, measurement_info,
                                         parameter_info, prior_info, nothing, true)

    # Extract parameter vector
    θ_names = θ_indices.θ_names
    param_vecNominal = [parameter_info.nominal_value[findfirst(x -> x == θ_names[i], parameter_info.parameter_id)] for i in eachindex(θ_names)]
    param_vec = transformθ(param_vecNominal, θ_names, θ_indices, reverse_transform=true)

    jac_out = zeros(length(param_vec), length(measurement_info.time))
    residual_grad = ForwardDiff.gradient(compute_sum_residuals, param_vec)
    compute_Jacobian!(jac_out, param_vec)
    sqdiffResidual = sum((sum(jac_out, dims=2) - residual_grad).^2)
    @test sqdiffResidual ≤ 1e-5
end


function get_file_ode_values(petab_model::PEtabModel)

    # Change model parameters
    experimental_conditions_file, measurements_file, parameters_file, observables_file = read_petab_files(petab_model)
    parameter_info = process_parameters(parameters_file)
    measurement_info = process_measurements(measurements_file, observables_file)
    θ_indices = compute_θ_indices(parameter_info, measurement_info, petab_model)

    θ_names = θ_indices.θ_names
    θ_est = parameter_info.nominal_value[findall(x -> x ∈ θ_names, parameter_info.parameter_id)]

    return θ_est[θ_indices.iθ_dynamic]
end

function test_split_conditions(petab_model)
    prob1 = PEtabODEProblem(petab_model, hessian_method=:ForwardDiff, split_over_conditions=false,
                            verbose=false)
    prob2 = PEtabODEProblem(petab_model, gradient_method=:ForwardEquations, sensealg=:ForwardDiff,
                            hessian_method=:ForwardDiff, split_over_conditions=true,
                            verbose=false)
    prob3 = PEtabODEProblem(petab_model, hessian_method=:BlockForwardDiff, split_over_conditions=false,
                            verbose=false)
    prob4 = PEtabODEProblem(petab_model, hessian_method=:BlockForwardDiff, split_over_conditions=true,
                            verbose=false)
    x = prob1.θ_nominalT
    _ = prob1.compute_cost(x)
    _ = prob2.compute_cost(x)
    _ = prob3.compute_cost(x)
    _ = prob4.compute_cost(x)
    # Test split for full-hessian and Forward-equations
    g1 = prob1.compute_gradient(x)
    g2 = prob2.compute_gradient(x)
    h1 = prob1.compute_hessian(x)
    h2 = prob2.compute_hessian(x)
    @test norm(g1 - g2) ≤ 1e-10
    @test norm(h1 - h2) ≤ 1e-10

    # Test split for block-hessian
    h3 = prob3.compute_hessian(x)
    h4 = prob4.compute_hessian(x)
    @test norm(h3 - h4) ≤ 1e-10

    return nothing
end
