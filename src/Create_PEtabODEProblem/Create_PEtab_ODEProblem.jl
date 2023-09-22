"""
    PEtabODEProblem(petab_model::PEtabModel; <keyword arguments>)

Given a `PEtabModel` creates a `PEtabODEProblem` with potential user specified options.

The keyword arguments (described below) allows to choose cost, gradient, and Hessian methods, ODE solver options, 
and other tuneable options that can potentially make computations more efficient for some "edge-case" models. Please 
refer to the documentation for guidance on selecting the most efficient options for different types of models. If a 
keyword argument is not set, a suitable default option is chosen based on the number of model parameters.

Once created, a `PEtabODEProblem` contains everything needed to perform parameter estimtimation (see above)

!!! note
    Every problem is unique, so even though the default settings often work well they might not be optimal.

# Keyword arguments
- `ode_solver::ODESolver`: Options for the ODE solver when computing the cost, such as solver and tolerances.
- `ode_solver_gradient::ODESolver`: Options for the ODE solver when computing the gradient, such as the ODE solver options used in adjoint sensitivity analysis. Defaults to `ode_solver` if not set to nothing.
- `ss_solver::SteadyStateSolver`: Options for finding steady-state for models with pre-equilibrium. Steady-state can be found via simulation or rootfinding, which can be set using `SteadyStateSolver` (see documentation). If not set, defaults to simulation with `wrms < 1` termination.
- `ss_solver_gradient::SteadyStateSolver`: Options for finding steady-state for models with pre-equilibrium when computing gradients. Defaults to `ss_solver` value if not set.
- `cost_method::Symbol=:Standard`: Method for computing the cost (objective). Two options are available: `:Standard`, which is the most efficient, and `:Zygote`, which is less efficient but compatible with the Zygote automatic differentiation library.
- `gradient_method=nothing`: Method for computing the gradient of the objective. Four options are available:
    * `:ForwardDiff`: Compute the gradient via forward-mode automatic differentiation using ForwardDiff.jl. Most efficient for models with ≤50 parameters. The number of chunks can be optionally set using `chunksize`.
    * `:ForwardEquations`: Compute the gradient via the model sensitivities, where `sensealg` specifies how to solve for the sensitivities. Most efficient when the Hessian is approximated using the Gauss-Newton method and when the optimizer can reuse the sensitivities (`reuse_sensitivities`) from gradient computations in Hessian computations (e.g., when the optimizer always computes the gradient before the Hessian).
    * `:Adjoint`: Compute the gradient via adjoint sensitivity analysis, where `sensealg` specifies which algorithm to use. Most efficient for large models (≥75 parameters).
    * `:Zygote`: Compute the gradient via the Zygote package, where `sensealg` specifies which sensitivity algorithm to use when solving the ODE model. This is the most inefficient option and not recommended.
- `hessian_method=nothing`: method for computing the Hessian of the cost. There are three available options:
    * `:ForwardDiff`: Compute the Hessian via forward-mode automatic differentiation using ForwardDiff.jl. This is often only computationally feasible for models with ≤20 parameters but can greatly improve optimizer convergence.
    * `:BlockForwardDiff`: Compute the Hessian block approximation via forward-mode automatic differentiation using ForwardDiff.jl. The approximation consists of two block matrices: the first is the Hessian for only the dynamic parameters (parameter part of the ODE system), and the second is for the non-dynamic parameters (e.g., noise parameters). This is computationally feasible for models with ≤20 dynamic parameters and often performs better than BFGS methods.
    * `:GaussNewton`: Approximate the Hessian via the Gauss-Newton method, which often performs better than the BFGS method. If we can reuse the sensitivities from the gradient in the optimizer (see `reuse_sensitivities`), this method is best paired with `gradient_method=:ForwardEquations`.
- `sparse_jacobian::Bool=false`: When solving the ODE du/dt=f(u, p, t), whether implicit solvers use a sparse Jacobian. Sparse Jacobian often performs best for large models (≥100 states).
- `specialize_level=SciMLBase.FullSpecialize`: Specialization level when building the ODE problem. It is not recommended to change this parameter (see https://docs.sciml.ai/SciMLBase/stable/interfaces/Problems/).
- `sensealg`: Sensitivity algorithm for gradient computations. The available options for each gradient method are:
    * `:ForwardDiff`: None (as ForwardDiff takes care of all computation steps).
    * `:ForwardEquations`: `:ForwardDiff` (uses ForwardDiff.jl and typicaly performs best) or `ForwardDiffSensitivity()` and `ForwardSensitivity()` from SciMLSensitivity.jl (https://github.com/SciML/SciMLSensitivity.jl).
    * `:Adjoint`: `InterpolatingAdjoint()` and `QuadratureAdjoint()` from SciMLSensitivity.jl.
    * `:Zygote`: All sensealg in SciMLSensitivity.jl.
- `sensealg_ss=nothing`: Sensitivity algorithm for adjoint gradient computations for steady-state simulations. The available options are `SteadyStateAdjoint()`, `InterpolatingAdjoint()`, and `QuadratureAdjoint()` from SciMLSensitivity.jl. `SteadyStateAdjoint()` is the most efficient but requires a non-singular Jacobian, and in the case of a non-singular Jacobian, the code automatically switches to `InterpolatingAdjoint()`.
- `chunksize=nothing`: Chunk-size for ForwardDiff.jl when computing the gradient and Hessian via forward-mode automatic differentiation. If nothing is provided, the default value is used. Tuning `chunksize` is non-trivial, and we plan to add automatic functionality for this.
- `split_over_conditions::Bool=false`: For gradient and Hessian via ForwardDiff.jl, whether or not to split calls to ForwardDiff across experimental (simulation) conditions. This parameter should only be set to true if the model has many parameters specific to an experimental condition; otherwise, the overhead from the calls will increase run time. See the Beer example for a case where this is needed.
- `reuse_sensitivities::Bool=false` : If set to `true`, reuse the sensitivities computed during gradient computations for the Gauss-Newton Hessian approximation. This option is only applicable when using `hessian_method=:GaussNewton` and `gradient_method=:ForwardEquations`. Note that it should only be used when the optimizer always computes the gradient before the Hessian.
- `verbose::Bool=true` : If set to `true`, print progress messages while setting up the PEtabODEProblem.
"""
function PEtabODEProblem(petab_model::PEtabModel;
                         ode_solver::Union{Nothing, ODESolver}=nothing,
                         ode_solver_gradient::Union{Nothing, ODESolver}=nothing,
                         ss_solver::Union{Nothing, SteadyStateSolver}=nothing,
                         ss_solver_gradient::Union{Nothing, SteadyStateSolver}=nothing,
                         cost_method::Union{Nothing, Symbol}=:Standard,
                         gradient_method::Union{Nothing, Symbol}=nothing,
                         hessian_method::Union{Nothing, Symbol}=nothing,
                         sparse_jacobian::Union{Nothing, Bool}=nothing,
                         specialize_level=SciMLBase.FullSpecialize,
                         sensealg=nothing,
                         sensealg_ss=nothing,
                         chunksize::Union{Nothing, Int64}=nothing,
                         split_over_conditions::Bool=false,
                         n_processes::Signed=1,
                         reuse_sensitivities::Bool=false,
                         verbose::Bool=true,
                         custom_parameter_values::Union{Nothing, Dict}=nothing)::PEtabODEProblem

    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && @printf(" Building PEtabODEProblem for %s\n", petab_model.model_name)

    # Sanity check user provided methods
    allowed_cost_methods = [:Standard, :Zygote]
    allowed_gradient_methods = [nothing, :ForwardDiff, :ForwardEquations, :Adjoint, :Zygote]
    allowed_hessian_methods = [nothing, :ForwardDiff, :BlockForwardDiff, :GaussNewton]
    @assert cost_method ∈ allowed_cost_methods "Allowed cost methods are " * string(allowed_cost_methods) * " not " * string(cost_method)
    @assert gradient_method ∈ allowed_gradient_methods "Allowed gradient methods are " * string(allowed_gradient_methods) * " not " * string(gradient_method)
    @assert hessian_method ∈ allowed_hessian_methods "Allowed hessian methods are " * string(allowed_hessian_methods) * " not " * string(hessian_method)

    if gradient_method === :Adjoint
        @assert "SciMLSensitivity" ∈ string.(values(Base.loaded_modules)) "To use adjoint sensitivity analysis SciMLSensitivity must be loaded"
    end
    if gradient_method === :Zygote
        @assert "Zygote" ∈ string.(values(Base.loaded_modules)) "To use Zygote automatic differantiation Zygote must be loaded"
        @assert "SciMLSensitivity" ∈ string.(values(Base.loaded_modules)) "To use Zygote automatic differantiation SciMLSensitivity must be loaded"
    end

    # Structs to bookep parameters, measurements, observations etc...
    experimental_conditions, measurements_data, parameters_data, observables_data = read_petab_files(petab_model)
    parameter_info = process_parameters(parameters_data, custom_parameter_values=custom_parameter_values)
    measurement_info = process_measurements(measurements_data, observables_data)
    θ_indices = compute_θ_indices(parameter_info, measurement_info, petab_model)
    prior_info = process_priors(θ_indices, parameters_data)

    # In case not specified by the user set ODE, gradient and Hessian options
    nODEs = length(states(petab_model.system))
    if nODEs ≤ 15 && length(θ_indices.θ_dynamic_names) ≤ 20
        model_size = :Small
    elseif nODEs ≤ 50 && length(θ_indices.θ_dynamic_names) ≤ 69
        model_size = :Medium
    else
        model_size = :Large
    end
    _gradient_method = set_gradient_method(gradient_method, model_size, reuse_sensitivities)
    _hessian_method = set_hessian_method(hessian_method, model_size)
    _sensealg = set_sensealg(sensealg, Val(_gradient_method))
    _ode_solver = set_ODESolver(ode_solver, model_size, _gradient_method)
    _ode_solver_gradient = isnothing(ode_solver_gradient) ? deepcopy(_ode_solver) : ode_solver_gradient
    __ss_solver = set_SteadyStateSolver(ss_solver, _ode_solver)
    __ss_solver_gradient = isnothing(ss_solver_gradient) ? deepcopy(__ss_solver) : ss_solver_gradient
    _sparse_jacobian = !isnothing(sparse_jacobian) ? sparse_jacobian : (model_size === :Large ? true : false)

    simulation_info = process_simulationinfo(petab_model, measurement_info, sensealg=_sensealg)

    # The time-span 5e3 is overwritten when performing forward simulations. As we solve an expanded system with the forward
    # equations, we need a seperate problem for it
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && @printf(" Building ODEProblem from ODESystem ...")
    timeTake = @elapsed begin
    # Set model parameter values to those in the PeTab parameter to ensure correct constant parameters
    set_parameters_to_file_values!(petab_model.parameter_map, petab_model.state_map, parameter_info)
    if petab_model.system isa ODESystem && petab_model.defined_in_julia == false
        __ode_problem = ODEProblem{true, specialize_level}(petab_model.system, petab_model.state_map, [0.0, 5e3], petab_model.parameter_map, jac=true, sparse=_sparse_jacobian)
    else
        # For reaction systems this bugs out if I try to set specialize_level (specifially state-map and parameter-map are not 
        # made into vectors)
        __ode_problem = ODEProblem(petab_model.system, zeros(Float64, length(petab_model.state_map)), [0.0, 5e3], petab_model.parameter_map, jac=true, sparse=_sparse_jacobian)
    end
    _ode_problem = remake(__ode_problem, p = convert.(Float64, __ode_problem.p), u0 = convert.(Float64, __ode_problem.u0))
    end
    verbose == true && @printf(" done. Time = %.1e\n", timeTake)

    # Needed to properly initalise steady-state solver options with model Jacobian etc...
    _ss_solver = _get_steady_state_solver(__ss_solver, _ode_problem, _ode_solver.abstol*100, _ode_solver.reltol*100, _ode_solver.maxiters)
    _ss_solver_gradient = _get_steady_state_solver(__ss_solver_gradient, _ode_problem, _ode_solver_gradient.abstol*100, _ode_solver_gradient.reltol*100, _ode_solver_gradient.maxiters)

    # If we are computing the cost, gradient and hessians accross several processes we need to send ODEProblem, and
    # PEtab structs to each process
    if n_processes > 1
        jobs, results = setUpProcesses(petab_model, ode_solver, solverAbsTol, solverRelTol, odeSolverAdjoint, sensealgAdjoint,
                                       sensealg_ss, solverAdjointAbsTol, solverAdjointRelTol, odeSolverForwardEquations,
                                       sensealg_forward_equations, parameter_info, measurement_info, simulation_info, θ_indices,
                                       prior_info, ode_problem, chunksize)
    else
        jobs, results = nothing, nothing
    end

    petab_ODE_cache = PEtabODEProblemCache(_gradient_method, _hessian_method, petab_model, _sensealg, measurement_info, simulation_info, θ_indices, chunksize)
    petab_ODESolver_cache = PEtabODESolverCache(_gradient_method, _hessian_method, petab_model, simulation_info, θ_indices, chunksize)

    # To get multiple dispatch to work correctly 
    _cost_method = cost_method === :Zygote ? Val(:Zygote) : cost_method
    __gradient_method = _gradient_method === :Zygote ? Val(:Zygote) : _gradient_method

    # The cost (likelihood) can either be computed in the standard way or the Zygote way. The second consumes more
    # memory as in-place mutations are not compatible with Zygote
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building cost function for method ", string(cost_method), " ...")
    b_build = @elapsed compute_cost = create_cost_function(_cost_method, _ode_problem, _ode_solver, _ss_solver, 
        petab_ODE_cache, petab_ODESolver_cache, petab_model, simulation_info, θ_indices, measurement_info, parameter_info, 
        prior_info, _sensealg, n_processes, jobs, results, false)

    compute_chi2 = (θ; as_array=false) -> begin
        _ = compute_cost(θ)
        if as_array == false
            return sum(measurement_info.chi2_values)
        else
            return measurement_info.chi2_values
        end
    end
    compute_residuals = (θ; as_array=false) -> begin
        _ = compute_cost(θ)
        if as_array == false
            return sum(measurement_info.residuals)
        else
            return measurement_info.residuals
        end
    end
    compute_simulated_values = (θ) -> begin
        _ = compute_cost(θ)
        return measurement_info.simulated_values
    end
    verbose == true && @printf(" done. Time = %.1e\n", b_build)

    # The gradient can either be computed via autodiff, forward sensitivity equations, adjoint sensitivity equations
    # and Zygote
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building gradient function for method ", string(_gradient_method), " ...")
    _ode_problemGradient = gradient_method === :ForwardEquations ? get_ODE_forward_equations(_ode_problem, sensealg) : get_ODE_forward_equations(_ode_problem, :NoSpecialProblem)
    b_build = @elapsed compute_gradient! = create_gradient_function(__gradient_method, _ode_problemGradient, 
        _ode_solver_gradient, _ss_solver_gradient, petab_ODE_cache, petab_ODESolver_cache, petab_model, simulation_info, θ_indices,
        measurement_info, parameter_info, _sensealg, prior_info, chunksize=chunksize, n_processes=n_processes,
        jobs=jobs, results=results, split_over_conditions=split_over_conditions, sensealg_ss=sensealg_ss)
    # Non in-place gradient
    compute_gradient = (θ) -> begin
        gradient = zeros(Float64, length(θ))
        compute_gradient!(gradient, θ)
        return gradient
    end
    verbose == true && @printf(" done. Time = %.1e\n", b_build)

    # The Hessian can either be computed via automatic differentation, or approximated via a block approximation or the
    # Gauss Newton method
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building hessian function for method ", string(_hessian_method), " ...")
    b_build = @elapsed compute_hessian! = create_hessian_function(_hessian_method, _ode_problem, _ode_solver, _ss_solver,
        petab_ODE_cache, petab_ODESolver_cache, petab_model, simulation_info, θ_indices, measurement_info, parameter_info,
        prior_info, chunksize, n_processes=n_processes, jobs=jobs, results=results,
        split_over_conditions=split_over_conditions, reuse_sensitivities=reuse_sensitivities)
    # Non-inplace Hessian
    compute_hessian = (θ) -> begin
                                hessian = zeros(Float64, length(θ), length(θ))
                                compute_hessian!(hessian, θ)
                                return hessian
                            end
    verbose == true && @printf(" done. Time = %.1e\n", b_build)

    # Nominal parameter values + parameter bounds on parameter-scale (transformed)
    θ_names = θ_indices.θ_names
    lower_bounds = [parameter_info.lower_bounds[findfirst(x -> x == θ_names[i], parameter_info.parameter_id)] for i in eachindex(θ_names)]
    upper_bounds = [parameter_info.upper_bounds[findfirst(x -> x == θ_names[i], parameter_info.parameter_id)] for i in eachindex(θ_names)]
    θ_nominal = [parameter_info.nominal_value[findfirst(x -> x == θ_names[i], parameter_info.parameter_id)] for i in eachindex(θ_names)]
    transformθ!(lower_bounds, θ_names, θ_indices, reverse_transform=true)
    transformθ!(upper_bounds, θ_names, θ_indices, reverse_transform=true)
    θ_nominalT = transformθ(θ_nominal, θ_names, θ_indices, reverse_transform=true)

    petab_problem = PEtabODEProblem(compute_cost,
                                    compute_chi2,
                                    compute_gradient!,
                                    compute_gradient,
                                    compute_hessian!,
                                    compute_hessian,
                                    compute_simulated_values,
                                    compute_residuals,
                                    cost_method,
                                    _gradient_method,
                                    Symbol(_hessian_method),
                                    Int64(length(θ_names)),
                                    θ_names,
                                    θ_nominal,
                                    θ_nominalT,
                                    lower_bounds,
                                    upper_bounds,
                                    petab_model,
                                    _ode_solver,
                                    _ode_solver_gradient,
                                    _ss_solver,
                                    _ss_solver_gradient,
                                    θ_indices,
                                    simulation_info,
                                    split_over_conditions)
    return petab_problem
end


function create_cost_function(which_method::Symbol,
                              ode_problem::ODEProblem,
                              ode_solver::ODESolver,
                              ss_solver::SteadyStateSolver,
                              petab_ODE_cache::PEtabODEProblemCache,
                              petab_ODESolver_cache::PEtabODESolverCache,
                              petab_model::PEtabModel,
                              simulation_info::SimulationInfo,
                              θ_indices::ParameterIndices,
                              measurement_info::MeasurementsInfo,
                              parameter_info::ParametersInfo,
                              prior_info::PriorInfo,              
                              sensealg,  
                              n_processes,
                              jobs,
                              results,
                              compute_residuals)

    # Functions needed for mapping θ_est to the ODE problem, and then for solving said ODE-system
    if which_method == :Standard && n_processes == 1

        __compute_cost = let θ_indices=θ_indices, ode_solver=ode_solver, prior_info=prior_info
                            (θ_est) -> compute_cost(θ_est,
                                                    ode_problem,
                                                    ode_solver,
                                                    ss_solver,
                                                    petab_model,
                                                    simulation_info,
                                                    θ_indices,
                                                    measurement_info,
                                                    parameter_info,
                                                    prior_info,
                                                    petab_ODE_cache,
                                                    petab_ODESolver_cache,
                                                    [:all],
                                                    true,
                                                    false,
                                                    compute_residuals)
                        end
    end


    if false
        __compute_cost = (θ_est) ->  begin
                                            costTot::Float64 = 0.0
                                            @inbounds for i in n_processes:-1:1
                                                @async put!(jobs[i], tuple(θ_est, :Cost))
                                            end
                                            @inbounds for i in n_processes:-1:1
                                                status::Symbol, cost::Float64 = take!(results[i])
                                                if status != :Done
                                                    println("Error : Could not send ODE problem to proces ", procs()[i])
                                                end
                                                costTot += cost
                                            end
                                            return costTot
                                        end

    end

    return __compute_cost
end


function create_gradient_function(which_method::Symbol,
                                  ode_problem::ODEProblem,
                                  ode_solver::ODESolver,
                                  ss_solver::SteadyStateSolver,
                                  petab_ODE_cache::PEtabODEProblemCache,
                                  petab_ODESolver_cache::PEtabODESolverCache,
                                  petab_model::PEtabModel,
                                  simulation_info::SimulationInfo,
                                  θ_indices::ParameterIndices,
                                  measurement_info::MeasurementsInfo,
                                  parameter_info::ParametersInfo,
                                  sensealg,
                                  prior_info::PriorInfo;
                                  chunksize::Union{Nothing, Int64}=nothing,
                                  sensealg_ss=nothing,
                                  n_processes::Int64=1,
                                  jobs=nothing,
                                  results=nothing,
                                  split_over_conditions::Bool=false)

    θ_dynamic = petab_ODE_cache.θ_dynamic
    θ_sd = petab_ODE_cache.θ_sd
    θ_observable = petab_ODE_cache.θ_observable
    θ_non_dynamic = petab_ODE_cache.θ_non_dynamic

    if which_method == :ForwardDiff && n_processes == 1

        iθ_sd, iθ_observable, iθ_non_dynamic, iθ_not_ode = get_index_parameters_not_ODE(θ_indices)
        compute_cost_θ_not_ODE = (x) -> compute_cost_not_solve_ODE(x[iθ_sd], x[iθ_observable], x[iθ_non_dynamic],
                                                                 petab_model, simulation_info, θ_indices, measurement_info,
                                                                 parameter_info, petab_ODE_cache, exp_id_solve=[:all],
                                                                 compute_gradient_not_solve_autodiff=true)

        if split_over_conditions == false
            # Compute gradient for parameters which are a part of the ODE-system (dynamic parameters)
            compute_cost_θ_dynamic = (x) -> compute_cost_solve_ODE(x, θ_sd, θ_observable,  θ_non_dynamic, ode_problem,
                ode_solver, ss_solver, petab_model, simulation_info, θ_indices, measurement_info,
                parameter_info, petab_ODE_cache, petab_ODESolver_cache, compute_gradient_θ_dynamic=true, exp_id_solve=[:all])

            _chunksize = isnothing(chunksize) ? ForwardDiff.Chunk(θ_dynamic) : ForwardDiff.Chunk(chunksize)
            cfg = ForwardDiff.GradientConfig(compute_cost_θ_dynamic, θ_dynamic, _chunksize)

            _compute_gradient! = (gradient, θ_est; isremade=false) -> compute_gradient_autodiff!(gradient,
                                                                                                 θ_est,
                                                                                                 compute_cost_θ_not_ODE,
                                                                                                 compute_cost_θ_dynamic,
                                                                                                 petab_ODE_cache,
                                                                                                 cfg,
                                                                                                 simulation_info,
                                                                                                 θ_indices,
                                                                                                 prior_info;
                                                                                                 isremade=isremade)
        end

        if split_over_conditions == true

            compute_cost_θ_dynamic = (x, _exp_id_solve) -> compute_cost_solve_ODE(x, θ_sd, θ_observable,  θ_non_dynamic, ode_problem,
                ode_solver, ss_solver, petab_model, simulation_info, θ_indices, measurement_info, parameter_info,
                petab_ODE_cache, petab_ODESolver_cache, compute_gradient_θ_dynamic=true, exp_id_solve=_exp_id_solve)

            _compute_gradient! = (gradient, θ_est) -> compute_gradient_autodiff_split!(gradient,
                                                                                       θ_est,
                                                                                       compute_cost_θ_not_ODE,
                                                                                       compute_cost_θ_dynamic,
                                                                                       petab_ODE_cache,
                                                                                       simulation_info,
                                                                                       θ_indices,
                                                                                       prior_info)
        end
    end

    if which_method === :ForwardEquations && n_processes == 1

        _chunksize = isnothing(chunksize) ? ForwardDiff.Chunk(θ_dynamic) : ForwardDiff.Chunk(chunksize)
        if sensealg === :ForwardDiff && split_over_conditions == false
            _solve_ode_all_conditions! = (sol_values, θ) -> begin
                solve_ode_all_conditions!(sol_values, θ, petab_ODESolver_cache,
                    simulation_info.ode_sols_derivatives, ode_problem, petab_model, simulation_info, ode_solver,
                    ss_solver, θ_indices, petab_ODE_cache, save_at_observed_t=true, exp_id_solve=[:all],
                    compute_forward_sensitivites_ad=true)
            end
            cfg = ForwardDiff.JacobianConfig(_solve_ode_all_conditions!, petab_ODE_cache.sol_values,
                petab_ODE_cache.θ_dynamic, _chunksize)
        end

        if sensealg === :ForwardDiff && split_over_conditions == true

            _solve_ode_all_conditions! = (sol_values, θ, _exp_id_solve) -> begin
                solve_ode_all_conditions!(sol_values, θ, petab_ODESolver_cache,
                    simulation_info.ode_sols_derivatives, ode_problem, petab_model, simulation_info, ode_solver,
                    ss_solver, θ_indices, petab_ODE_cache, save_at_observed_t=true, exp_id_solve=_exp_id_solve,
                    compute_forward_sensitivites_ad=true)
            end
            cfg = ForwardDiff.JacobianConfig(_solve_ode_all_conditions!, petab_ODE_cache.sol_values,
                petab_ODE_cache.θ_dynamic, _chunksize)
        end

        if sensealg != :ForwardDiff
            _solve_ode_all_conditions! = (ode_sols, ode_problem, θ_dynamic, _exp_id_solve) -> begin
                solve_ode_all_conditions!(ode_sols, ode_problem, petab_model, θ_dynamic, petab_ODESolver_cache,
                    simulation_info, θ_indices, ode_solver, ss_solver, save_at_observed_t=true,
                    exp_id_solve=_exp_id_solve, compute_forward_sensitivites=true)
                end
            cfg = nothing
        end

        iθ_sd, iθ_observable, iθ_non_dynamic, iθ_not_ode = get_index_parameters_not_ODE(θ_indices)
        compute_cost_θ_not_ODE = (x) -> compute_cost_not_solve_ODE(x[iθ_sd], x[iθ_observable], x[iθ_non_dynamic],
            petab_model, simulation_info, θ_indices, measurement_info, parameter_info, petab_ODE_cache, exp_id_solve=[:all],
            compute_gradient_not_solve_forward=true)

        _compute_gradient! = (gradient, θ_est; isremade=false) -> compute_gradient_forward_equations!(gradient,
                                                                                                      θ_est,
                                                                                                      compute_cost_θ_not_ODE,
                                                                                                      petab_model,
                                                                                                      ode_problem,
                                                                                                      sensealg,
                                                                                                      simulation_info,
                                                                                                      θ_indices,
                                                                                                      measurement_info,
                                                                                                      parameter_info,
                                                                                                      _solve_ode_all_conditions!,
                                                                                                      prior_info,
                                                                                                      cfg,
                                                                                                      petab_ODE_cache,
                                                                                                      exp_id_solve=[:all],
                                                                                                      split_over_conditions=split_over_conditions,
                                                                                                      isremade=isremade)
    end

    if false

        _compute_gradient! = (gradient, θ_est) -> begin
                                                    gradient .= 0.0
                                                    @inbounds for i in n_processes:-1:1
                                                        @async put!(jobs[i], tuple(θ_est, which_method))
                                                    end
                                                    @inbounds for i in n_processes:-1:1
                                                        status::Symbol, gradientPart::Vector{Float64} = take!(results[i])
                                                        if status != :Done
                                                            println("Error : Could not compute gradient for ", procs()[i])
                                                        end
                                                        gradient .+= gradientPart
                                                    end
                                                end


    end

    return _compute_gradient!
end


function create_hessian_function(which_method::Symbol,
                                 ode_problem::ODEProblem,
                                 ode_solver::ODESolver,
                                 ss_solver::SteadyStateSolver,
                                 petab_ODE_cache::PEtabODEProblemCache,
                                 petab_ODESolver_cache::PEtabODESolverCache,
                                 petab_model::PEtabModel,
                                 simulation_info::SimulationInfo,
                                 θ_indices::ParameterIndices,
                                 measurement_info::MeasurementsInfo,
                                 parameter_info::ParametersInfo,
                                 prior_info::PriorInfo,
                                 chunksize::Union{Nothing, Int64};
                                 reuse_sensitivities::Bool=false,
                                 n_processes::Int64=1,
                                 jobs=nothing,
                                 results=nothing,
                                 split_over_conditions::Bool=false,
                                 return_jacobian::Bool=false)

    θ_dynamic = petab_ODE_cache.θ_dynamic
    θ_sd = petab_ODE_cache.θ_sd
    θ_observable = petab_ODE_cache.θ_observable
    θ_non_dynamic = petab_ODE_cache.θ_non_dynamic

    if which_method === :ForwardDiff

        if split_over_conditions == false

            _eval_hessian = (θ_est) -> compute_cost(θ_est, ode_problem, ode_solver, ss_solver, petab_model,
                simulation_info, θ_indices, measurement_info, parameter_info, prior_info, petab_ODE_cache,
                petab_ODESolver_cache, [:all], false, true, false)

            _chunksize = isnothing(chunksize) ? ForwardDiff.Chunk(zeros(length(θ_indices.θ_names))) : ForwardDiff.Chunk(chunksize)
            cfg = ForwardDiff.HessianConfig(_eval_hessian, zeros(length(θ_indices.θ_names)), _chunksize)

            _compute_hessian = (hessian, θ_est) -> compute_hessian!(hessian,
                                                                    θ_est,
                                                                    _eval_hessian,
                                                                    cfg,
                                                                    simulation_info,
                                                                    θ_indices,
                                                                    prior_info)
        end

        if split_over_conditions == true
            _eval_hessian = (θ_est, _condition_id) -> compute_cost(θ_est, ode_problem, ode_solver, ss_solver, petab_model,
                                                    simulation_info, θ_indices, measurement_info, parameter_info, prior_info,
                                                    petab_ODE_cache, petab_ODESolver_cache, _condition_id, false, true, false)
            _compute_hessian = (hessian, θ_est) -> compute_hessian_split!(hessian,
                                                                          θ_est,
                                                                          _eval_hessian,
                                                                          simulation_info,
                                                                          θ_indices,
                                                                          prior_info)
        end

    end

    # Functions needed for mapping θ_est to the ODE problem, and then for solving said ODE-system
    if which_method === :BlockForwardDiff

        iθ_sd, iθ_observable, iθ_non_dynamic, iθ_not_ode = get_index_parameters_not_ODE(θ_indices)
        compute_cost_θ_not_ODE = (x) -> compute_cost_not_solve_ODE(x[iθ_sd], x[iθ_observable], x[iθ_non_dynamic],
                                                                 petab_model, simulation_info, θ_indices, measurement_info,
                                                                 parameter_info, petab_ODE_cache, exp_id_solve=[:all],
                                                                 compute_gradient_not_solve_autodiff=true)

        if split_over_conditions == false

            compute_cost_θ_dynamic = (x) -> compute_cost_solve_ODE(x, θ_sd, θ_observable,  θ_non_dynamic, ode_problem,
                ode_solver, ss_solver, petab_model, simulation_info, θ_indices, measurement_info,
                parameter_info, petab_ODE_cache, petab_ODESolver_cache, compute_gradient_θ_dynamic=true, exp_id_solve=[:all])

            _chunksize = isnothing(chunksize) ? ForwardDiff.Chunk(θ_dynamic) : ForwardDiff.Chunk(chunksize)
            cfg = ForwardDiff.HessianConfig(compute_cost_θ_dynamic, θ_dynamic, ForwardDiff.Chunk(chunksize))

            _compute_hessian = (hessian, θ_est) -> compute_hessian_block!(hessian,
                                                                          θ_est,
                                                                          compute_cost_θ_not_ODE,
                                                                          compute_cost_θ_dynamic,
                                                                          petab_ODE_cache,
                                                                          cfg,
                                                                          simulation_info,
                                                                          θ_indices,
                                                                          prior_info,
                                                                          exp_id_solve=[:all])
        end

        if split_over_conditions == true

            compute_cost_θ_dynamic = (x, _exp_id_solve) -> compute_cost_solve_ODE(x, θ_sd, θ_observable,  θ_non_dynamic, ode_problem, ode_solver,
                ss_solver, petab_model, simulation_info, θ_indices, measurement_info, parameter_info, petab_ODE_cache,
                petab_ODESolver_cache, compute_gradient_θ_dynamic=true, exp_id_solve=_exp_id_solve)

            _compute_hessian = (hessian, θ_est) -> compute_hessian_block_split!(hessian,
                                                                                θ_est,
                                                                                compute_cost_θ_not_ODE,
                                                                                compute_cost_θ_dynamic,
                                                                                petab_ODE_cache,
                                                                                simulation_info,
                                                                                θ_indices,
                                                                                prior_info,
                                                                                exp_id_solve=[:all])
        end
    end

    if which_method == :GaussNewton && n_processes == 1

        change_simulation_condition! = (p_ode_problem, u0, conditionId, θ_dynamic) -> _change_simulation_condition!(p_ode_problem, u0, conditionId, θ_dynamic, petab_model, θ_indices)
        change_simulation_condition! = (p_ode_problem, u0, conditionId, θ_dynamic) -> _change_simulation_condition!(p_ode_problem, u0, conditionId, θ_dynamic, petab_model, θ_indices)

        if split_over_conditions == false
            _solve_ode_all_conditions! = (sol_values, θ) -> solve_ode_all_conditions!(sol_values, θ, petab_ODESolver_cache, simulation_info.ode_sols_derivatives, ode_problem, petab_model, simulation_info, ode_solver, ss_solver, θ_indices, petab_ODE_cache, save_at_observed_t=true, exp_id_solve=[:all], compute_forward_sensitivites_ad=true)
        else
            _solve_ode_all_conditions! = (sol_values, θ, _exp_id_solve) -> solve_ode_all_conditions!(sol_values, θ, petab_ODESolver_cache, simulation_info.ode_sols_derivatives, ode_problem, petab_model, simulation_info, ode_solver, ss_solver, θ_indices, petab_ODE_cache, save_at_observed_t=true, exp_id_solve=_exp_id_solve, compute_forward_sensitivites_ad=true)
        end

        _chunksize = isnothing(chunksize) ? ForwardDiff.Chunk(petab_ODE_cache.θ_dynamic) : ForwardDiff.Chunk(chunksize)
        cfg = cfg = ForwardDiff.JacobianConfig(_solve_ode_all_conditions!, petab_ODE_cache.sol_values, petab_ODE_cache.θ_dynamic, _chunksize)

        iθ_sd, iθ_observable, iθ_non_dynamic, iθ_not_ode = get_index_parameters_not_ODE(θ_indices)
        _compute_residuals_not_solve_ode! = (residuals, θ_not_ode) ->  begin
            θ_sd = @view θ_not_ode[iθ_sd]
            θ_observable = @view θ_not_ode[iθ_observable]
            θ_non_dynamic = @view θ_not_ode[iθ_non_dynamic]
            compute_residuals_not_solve_ode!(residuals, θ_sd, θ_observable,  θ_non_dynamic, petab_model, simulation_info,
                                         θ_indices, measurement_info, parameter_info, petab_ODE_cache;
                                         exp_id_solve=[:all])
                                                                        end

        _θ_not_ode = zeros(eltype(petab_ODE_cache.θ_dynamic), length(iθ_not_ode))
        cfg_not_solve_ode = ForwardDiff.JacobianConfig(_compute_residuals_not_solve_ode!, petab_ODE_cache.residuals_gn, _θ_not_ode, ForwardDiff.Chunk(_θ_not_ode))

        _compute_hessian = (hessian, θ_est; isremade=false) -> compute_GaussNewton_hessian!(hessian,
                                                                                                      θ_est,
                                                                                                      ode_problem,
                                                                                                      _compute_residuals_not_solve_ode!,
                                                                                                      petab_model,
                                                                                                      simulation_info,
                                                                                                      θ_indices,
                                                                                                      measurement_info,
                                                                                                      parameter_info,
                                                                                                      _solve_ode_all_conditions!,
                                                                                                      prior_info,
                                                                                                      cfg,
                                                                                                      cfg_not_solve_ode,
                                                                                                      petab_ODE_cache,
                                                                                                      exp_id_solve=[:all],
                                                                                                      reuse_sensitivities=reuse_sensitivities,
                                                                                                      return_jacobian=return_jacobian,
                                                                                                      split_over_conditions=split_over_conditions,
                                                                                                      isremade=isremade)
    end

    if false

        _compute_hessian = (hessian, θ_est) ->   begin
                                                    hessian .= 0.0
                                                    @inbounds for i in n_processes:-1:1
                                                        @async put!(jobs[i], tuple(θ_est, which_method))
                                                    end
                                                    @inbounds for i in n_processes:-1:1
                                                        status::Symbol, hessianPart::Matrix{Float64} = take!(results[i])
                                                        if status != :Done
                                                            println("Error : Could not send ODE problem to process ", procs()[i])
                                                        end
                                                        hessian .+= hessianPart
                                                    end
                                                end

    end

    return _compute_hessian
end


function get_ODE_forward_equations(ode_problem::ODEProblem,
                                   sensealg_forward_equations)::ODEProblem
    return ode_problem
end


function PEtabODEProblemCache(gradient_method::Symbol,
                              hessian_method::Union{Symbol, Nothing},
                              petab_model::PEtabModel,
                              sensealg,
                              measurement_info::MeasurementsInfo,
                              simulation_info::SimulationInfo,
                              θ_indices::ParameterIndices,
                              _chunksize)::PEtabODEProblemCache

    θ_dynamic = zeros(Float64, length(θ_indices.iθ_dynamic))
    θ_observable = zeros(Float64, length(θ_indices.iθ_observable))
    θ_sd = zeros(Float64, length(θ_indices.iθ_sd))
    θ_non_dynamic = zeros(Float64, length(θ_indices.iθ_non_dynamic))

    level_cache = 0
    if hessian_method ∈ [:ForwardDiff, :BlockForwardDiff, :GaussNewton]
        level_cache = 2
    elseif gradient_method ∈ [:ForwardDiff, :ForwardEquations]
        level_cache = 1
    else
        level_cache = 0
    end

    if isnothing(_chunksize)
        chunksize = length(θ_indices.iθ_dynamic) > 10 ? 10 : length(θ_indices.iθ_dynamic)
    else
        chunksize = _chunksize
    end

    _θ_dynamicT = zeros(Float64, length(θ_indices.iθ_dynamic))
    _θ_observableT = zeros(Float64, length(θ_indices.iθ_observable))
    _θ_sdT = zeros(Float64, length(θ_indices.iθ_sd))
    _θ_non_dynamicT = zeros(Float64, length(θ_indices.iθ_non_dynamic))
    θ_dynamicT = DiffCache(_θ_dynamicT, chunksize, levels=level_cache)
    θ_observableT = DiffCache(_θ_observableT, chunksize, levels=level_cache)
    θ_sdT = DiffCache(_θ_sdT, chunksize, levels=level_cache)
    θ_non_dynamicT = DiffCache(_θ_non_dynamicT, chunksize, levels=level_cache)

    gradient_θ_dyanmic = zeros(Float64, length(θ_dynamic))
    gradient_θ_not_ode = zeros(Float64, length(θ_indices.iθ_not_ode))

    # For forward sensitivity equations and adjoint sensitivity analysis we need to
    # compute partial derivatives symbolically. Here the helping vectors are pre-allocated
    if gradient_method ∈ [:Adjoint, :ForwardEquations] || hessian_method ∈ [:GaussNewton]
        n_model_states = length(states(petab_model.system))
        n_model_parameters = length(parameters(petab_model.system))
        ∂h∂u = zeros(Float64, n_model_states)
        ∂σ∂u = zeros(Float64, n_model_states)
        ∂h∂p = zeros(Float64, n_model_parameters)
        ∂σ∂p = zeros(Float64, n_model_parameters)
        ∂G∂p = zeros(Float64, n_model_parameters)
        ∂G∂p_ = zeros(Float64, n_model_parameters)
        ∂G∂u = zeros(Float64, n_model_states)
        p = zeros(Float64, n_model_parameters)
        u = zeros(Float64, n_model_states)
    else
        ∂h∂u = zeros(Float64, 0)
        ∂σ∂u = zeros(Float64, 0)
        ∂h∂p = zeros(Float64, 0)
        ∂σ∂p = zeros(Float64, 0)
        ∂G∂p = zeros(Float64, 0)
        ∂G∂p_ = zeros(Float64, 0)
        ∂G∂u =  zeros(Float64, 0)
        p = zeros(Float64, 0)
        u = zeros(Float64, 0)
    end

    # In case the sensitivites are computed via automatic differentitation we need to pre-allocate an
    # sensitivity matrix all experimental conditions (to efficiently levarage autodiff and handle scenarios are
    # pre-equlibrita model). Here we pre-allocate said matrix and the output matrix from the forward senstivity
    # code
    if (gradient_method === :ForwardEquations && sensealg === :ForwardDiff) || hessian_method === :GaussNewton
        n_model_states = length(states(petab_model.system))
        n_timepoints_save = sum(length(simulation_info.time_observed[experimental_condition_id]) for experimental_condition_id in simulation_info.experimental_condition_id)
        S = zeros(Float64, (n_timepoints_save*n_model_states, length(θ_indices.θ_dynamic_names)))
        sol_values = zeros(Float64, n_model_states, n_timepoints_save)
    else
        S = zeros(Float64, (0, 0))
        sol_values = zeros(Float64, (0, 0))
    end

    if hessian_method === :GaussNewton
        jacobian_gn = zeros(Float64, length(θ_indices.θ_names), length(measurement_info.time))
        residuals_gn = zeros(Float64, length(measurement_info.time))
    else
        jacobian_gn = zeros(Float64, (0, 0))
        residuals_gn = zeros(Float64, 0)
    end

    if gradient_method === :ForwardEquations || hessian_method === :GaussNewton
        _gradient = zeros(Float64, length(θ_indices.iθ_dynamic))
    else
        _gradient = zeros(Float64, 0)
    end

    if gradient_method === :Adjoint
        n_model_states = length(states(petab_model.system))
        n_model_parameters = length(parameters(petab_model.system))
        du = zeros(Float64, n_model_states)
        dp = zeros(Float64, n_model_parameters)
        _gradient_adjoint = zeros(Float64, n_model_parameters)
        S_t0 = zeros(Float64, (n_model_states, n_model_parameters))
    else
        du = zeros(Float64, 0)
        dp = zeros(Float64, 0)
        _gradient_adjoint = zeros(Float64, 0)
        S_t0 = zeros(Float64, (0, 0))
    end

    # Allocate arrays to track if θ_dynamic should be permuted prior and post gradient compuations. This feature
    # is used if PEtabODEProblem is remade (via remake) to compute the gradient of a problem with reduced number
    # of parameters where to run fewer chunks with ForwardDiff.jl we only run enough chunks to reach nθ_dynamic
    θ_dynamic_input_order::Vector{Int64} = collect(1:length(θ_dynamic))
    θ_dynamic_output_order::Vector{Int64} = collect(1:length(θ_dynamic))
    nθ_dynamic::Vector{Int64} = Int64[length(θ_dynamic)]

    petab_ODE_cache = PEtabODEProblemCache(θ_dynamic,
                                            θ_sd,
                                            θ_observable,
                                            θ_non_dynamic,
                                            θ_dynamicT,
                                            θ_sdT,
                                            θ_observableT,
                                            θ_non_dynamicT,
                                            gradient_θ_dyanmic,
                                            gradient_θ_not_ode,
                                            jacobian_gn,
                                            residuals_gn,
                                            _gradient,
                                            _gradient_adjoint,
                                            S_t0,
                                            ∂h∂u,
                                            ∂σ∂u,
                                            ∂h∂p,
                                            ∂σ∂p,
                                            ∂G∂p,
                                            ∂G∂p_,
                                            ∂G∂u,
                                            dp,
                                            du,
                                            p,
                                            u,
                                            S,
                                            sol_values,
                                            θ_dynamic_input_order,
                                            θ_dynamic_output_order,
                                            nθ_dynamic)

    return petab_ODE_cache
end


function PEtabODESolverCache(gradient_method::Symbol,
                             hessian_method::Union{Symbol, Nothing},
                             petab_model::PEtabModel,
                             simulation_info::SimulationInfo,
                             θ_indices::ParameterIndices,
                             _chunksize)::PEtabODESolverCache

    n_model_states = length(states(petab_model.system))
    n_model_parameters = length(parameters(petab_model.system))

    level_cache = 0
    if hessian_method ∈ [:ForwardDiff, :BlockForwardDiff, :GaussNewton]
        level_cache = 2
    elseif gradient_method ∈ [:ForwardDiff, :ForwardEquations]
        level_cache = 1
    else
        level_cache = 0
    end

    if isnothing(_chunksize)
        chunksize = length(θ_indices.iθ_dynamic) > 10 ? 10 : length(θ_indices.iθ_dynamic)
    else
        chunksize = _chunksize
    end

    if simulation_info.has_pre_equilibration_condition_id == true
        conditions_simulate_over = unique(vcat(simulation_info.pre_equilibration_condition_id, simulation_info.experimental_condition_id))
    else
        conditions_simulate_over = unique(simulation_info.experimental_condition_id)
    end

    _p_ode_problem_cache = Tuple(DiffCache(zeros(Float64, n_model_parameters), chunksize, levels=level_cache) for i in eachindex(conditions_simulate_over))
    _u0_cache = Tuple(DiffCache(zeros(Float64, n_model_states), chunksize, levels=level_cache) for i in eachindex(conditions_simulate_over))
    p_ode_problem_cache::Dict = Dict([(conditions_simulate_over[i], _p_ode_problem_cache[i]) for i in eachindex(_p_ode_problem_cache)])
    u0_cache::Dict = Dict([(conditions_simulate_over[i], _u0_cache[i]) for i in eachindex(_u0_cache)])

    return PEtabODESolverCache(p_ode_problem_cache, u0_cache)
end
