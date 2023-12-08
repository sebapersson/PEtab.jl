

import PEtab: process_priors, change_ode_parameters!, PriorInfo, transformθ, solve_ode_all_conditions!, compute_GaussNewton_hessian!, PEtabODESolverCache, PEtabODESolverCache, PEtabODEProblemCache


function _testCostGradientOrHessian(petab_model::PEtabModel,
                                    solverOptions::ODESolver,
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
                                    sensealg_ss=SteadyStateAdjoint())

    if isnothing(solverGradientOptions)
        solverGradientOptions = deepcopy(solverOptions)
    end

    petab_problem = PEtabODEProblem(petab_model,
                                         ode_solver=solverOptions;
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


function check_gradient_residuals(petab_model::PEtabModel, solverOptions::ODESolver; verbose::Bool=true, custom_parameter_values=nothing)

    # Process PeTab files into type-stable Julia structs
    experimental_conditions_file, measurementDataFile, parameterDataFile, observables_dataFile = read_petab_files(petab_model)
    parameterData = process_parameters(parameterDataFile, custom_parameter_values=custom_parameter_values)
    measurementData = process_measurements(measurementDataFile, observables_dataFile)
    simulation_info = process_simulationinfo(petab_model, measurementData, sensealg=nothing)

    # Indices for mapping parameter-estimation vector to dynamic, observable and sd parameters correctly when calculating cost
    paramEstIndices = compute_θ_indices(parameterData, measurementData, petab_model)

    # Set model parameter values to those in the PeTab parameter data ensuring correct value of constant parameters
    set_parameters_to_file_values!(petab_model.parameter_map, petab_model.state_map, parameterData)
    prior_info::PriorInfo = process_priors(paramEstIndices, parameterDataFile)

    petab_ODE_cache = PEtabODEProblemCache(:ForwardEquations, :GaussNewton, :ForwardDiff, petab_model, :ForwardDiff, measurementData, simulation_info, paramEstIndices, nothing)
    petab_ODESolver_cache = PEtabODESolverCache(:ForwardEquations, :GaussNewton, petab_model, simulation_info, paramEstIndices, nothing)

    # The time-span 5e3 is overwritten when performing actual forward simulations
    odeProb = ODEProblem(petab_model.system, petab_model.state_map, (0.0, 5e3), petab_model.parameter_map, jac=true, sparse=false)
    odeProb = remake(odeProb, p = convert.(Float64, odeProb.p), u0 = convert.(Float64, odeProb.u0))
    ss_options = SteadyStateSolver(:Simulate, abstol=solverOptions.abstol / 100.0, reltol = solverOptions.reltol / 100.0)
    _ss_options = PEtab._get_steady_state_solver(ss_options, odeProb, ss_options.abstol, ss_options.reltol, ss_options.maxiters)
    computeJacobian = PEtab.create_hessian_function(:GaussNewton, odeProb, solverOptions, _ss_options, petab_ODE_cache, petab_ODESolver_cache,
                                         petab_model, simulation_info, paramEstIndices, measurementData,
                                         parameterData, prior_info, nothing, return_jacobian=true)
    computeSumResiduals = PEtab.create_cost_function(:Standard, odeProb, solverOptions, _ss_options, petab_ODE_cache, petab_ODESolver_cache,
                                         petab_model, simulation_info, paramEstIndices, measurementData,
                                         parameterData, prior_info, nothing, 1, nothing, nothing, true)

    # Extract parameter vector
    namesParamEst = paramEstIndices.θ_names
    paramVecNominal = [parameterData.nominal_value[findfirst(x -> x == namesParamEst[i], parameterData.parameter_id)] for i in eachindex(namesParamEst)]
    paramVec = transformθ(paramVecNominal, namesParamEst, paramEstIndices, reverse_transform=true)

    jacOut = zeros(length(paramVec), length(measurementData.time))
    residualGrad = ForwardDiff.gradient(computeSumResiduals, paramVec)
    computeJacobian(jacOut, paramVec)
    sqDiffResidual = sum((sum(jacOut, dims=2) - residualGrad).^2)
    @test sqDiffResidual ≤ 1e-5
end


function get_file_ode_values(petab_model::PEtabModel)

    # Change model parameters
    experimental_conditions_file, measurementDataFile, parameterDataFile, observables_dataFile = read_petab_files(petab_model)
    parameter_info = process_parameters(parameterDataFile)
    measurement_info = process_measurements(measurementDataFile, observables_dataFile)
    θ_indices = compute_θ_indices(parameter_info, measurement_info, petab_model)

    θ_names = θ_indices.θ_names
    θ_est = parameter_info.nominal_value[findall(x -> x ∈ θ_names, parameter_info.parameter_id)]

    return θ_est[θ_indices.iθ_dynamic]
end