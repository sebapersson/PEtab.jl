function PEtabODEProblem(petab_model::PEtabModel;
                         ode_solver::Union{Nothing, ODESolver}=nothing,
                         ode_solver_gradient::Union{Nothing, ODESolver}=nothing,
                         ss_solver::Union{Nothing, SteadyStateSolver}=nothing,
                         ss_solver_gradient::Union{Nothing, SteadyStateSolver}=nothing,
                         cost_method::Union{Nothing, Symbol}=:Standard,
                         gradient_method::Union{Nothing, Symbol}=nothing,
                         hessian_method::Union{Nothing, Symbol}=nothing,
                         FIM_method::Union{Nothing, Symbol}=nothing,
                         sparse_jacobian::Union{Nothing, Bool}=nothing,
                         specialize_level=SciMLBase.FullSpecialize,
                         sensealg=nothing,
                         sensealg_ss=nothing,
                         chunksize::Union{Nothing, Int64}=nothing,
                         split_over_conditions::Bool=false,
                         reuse_sensitivities::Bool=false,
                         verbose::Bool=true,
                         custom_parameter_values::Union{Nothing, Dict}=nothing)::PEtabODEProblem

    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && @printf(" Building PEtabODEProblem for %s\n", petab_model.model_name)

    # Sanity check user provided methods
    allowed_cost_methods = [:Standard, :Zygote]
    allowed_gradient_methods = [nothing, :ForwardDiff, :ForwardEquations, :Adjoint, :Zygote]
    allowed_hessian_methods = [nothing, :ForwardDiff, :BlockForwardDiff, :GaussNewton]
    allowed_FIM_methods = [nothing, :ForwardDiff, :GaussNewton]
    @assert cost_method ∈ allowed_cost_methods "Allowed cost methods are " * string(allowed_cost_methods) * " not " * string(cost_method)
    @assert gradient_method ∈ allowed_gradient_methods "Allowed gradient methods are " * string(allowed_gradient_methods) * " not " * string(gradient_method)
    @assert hessian_method ∈ allowed_hessian_methods "Allowed hessian methods are " * string(allowed_hessian_methods) * " not " * string(hessian_method)
    @assert FIM_method ∈ allowed_FIM_methods "Allowed FIM methods are " * string(allowed_FIM_methods) * " not " * string(FIM_method)

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

    # Select methods for computing Fisher-Information-Matrix (FIM)
    if isnothing(FIM_method) && length(θ_indices.θ_names) ≤ 100
        _FIM_method = :ForwardDiff
    elseif isnothing(FIM_method)
        _FIM_method = :GaussNewton
    else
        _FIM_method = FIM_method
    end

    # In case the user has not provided input, set default values 
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
    time_take = @elapsed begin
        # Set model parameter values to those in the PeTab parameter to ensure correct constant parameters
        set_parameters_to_file_values!(petab_model.parameter_map, petab_model.state_map, parameter_info)
        if petab_model.system isa ODESystem && petab_model.defined_in_julia == false
            __ode_problem = ODEProblem{true, specialize_level}(petab_model.system, petab_model.state_map, 
                                                               [0.0, 5e3], petab_model.parameter_map, jac=true, 
                                                               sparse=_sparse_jacobian)
        else
            # For reaction systems this bugs out if I try to set specialize_level (specifially state-map and parameter-map are not
            # made into vectors)
            __ode_problem = ODEProblem(petab_model.system, zeros(Float64, length(petab_model.state_map)), 
                                       [0.0, 5e3], petab_model.parameter_map, jac=true, sparse=_sparse_jacobian)
        end
        _ode_problem = remake(__ode_problem, p = convert.(Float64, __ode_problem.p), 
                              u0 = convert.(Float64, __ode_problem.u0))
    end
    verbose == true && @printf(" done. Time = %.1e\n", time_take)

    # Needed to properly initalise steady-state solver options with model Jacobian etc...
    _ss_solver = _get_steady_state_solver(__ss_solver, _ode_problem, _ode_solver.abstol*100, _ode_solver.reltol*100, _ode_solver.maxiters)
    _ss_solver_gradient = _get_steady_state_solver(__ss_solver_gradient, _ode_problem, _ode_solver_gradient.abstol*100, _ode_solver_gradient.reltol*100, _ode_solver_gradient.maxiters)

    # Cache to avoid to many allocations 
    petab_ODE_cache = PEtabODEProblemCache(_gradient_method, _hessian_method, _FIM_method, petab_model, 
                                           _sensealg, measurement_info, simulation_info, θ_indices, chunksize)
    petab_ODESolver_cache = PEtabODESolverCache(_gradient_method, _hessian_method, petab_model, simulation_info, 
                                                θ_indices, chunksize)

    # To get multiple dispatch to work correctly when choosing cost and or gradient methods with extensions
    _cost_method = cost_method === :Zygote ? Val(:Zygote) : cost_method
    __gradient_method = _gradient_method === :Zygote ? Val(:Zygote) : _gradient_method

    # The cost (likelihood) can either be computed in the standard way or the Zygote way. The second consumes more
    # memory as in-place mutations are not compatible with Zygote
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building cost function for method ", string(cost_method), " ...")
    b_build = @elapsed begin
        compute_cost = create_cost_function(_cost_method, _ode_problem, _ode_solver, _ss_solver,
                                            petab_ODE_cache, petab_ODESolver_cache, petab_model,
                                            simulation_info, θ_indices, measurement_info, parameter_info,
                                            prior_info, _sensealg, false)
                        end
    verbose == true && @printf(" done. Time = %.1e\n", b_build)                        

    # The gradient can either be computed via autodiff, forward sensitivity equations, adjoint
    # sensitivity equations and Zygote
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building gradient function for method ", string(_gradient_method), " ...")
    _ode_problemGradient = gradient_method === :ForwardEquations ? get_ODE_forward_equations(_ode_problem, sensealg) : get_ODE_forward_equations(_ode_problem, :NoSpecialProblem)
    b_build = @elapsed begin
        compute_gradient!, compute_gradient = create_gradient_function(__gradient_method, _ode_problemGradient,
                                                                       _ode_solver_gradient, _ss_solver_gradient,
                                                                       petab_ODE_cache, petab_ODESolver_cache,
                                                                       petab_model, simulation_info, θ_indices,
                                                                       measurement_info, parameter_info, _sensealg,
                                                                       prior_info, chunksize=chunksize,
                                                                       split_over_conditions=split_over_conditions,
                                                                       sensealg_ss=sensealg_ss)
                        end
    verbose == true && @printf(" done. Time = %.1e\n", b_build)

    # The Hessian can either be computed via automatic differentation, or approximated via a block approximation or the
    # Gauss Newton method
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building hessian function for method ", string(_hessian_method), " ...")
    b_build = @elapsed  begin
        compute_hessian!, compute_hessian = create_hessian_function(_hessian_method, _ode_problem,
                                                                    _ode_solver, _ss_solver,
                                                                    petab_ODE_cache, petab_ODESolver_cache,
                                                                    petab_model, simulation_info,
                                                                    θ_indices, measurement_info, parameter_info,
                                                                    prior_info, chunksize,
                                                                    split_over_conditions=split_over_conditions,
                                                                    reuse_sensitivities=reuse_sensitivities)
                        end
    verbose == true && @printf(" done. Time = %.1e\n", b_build)

    # Fisher-Information-Matrix to use for practical identifiabillity analysis
    compute_FIM!, compute_FIM = create_hessian_function(_FIM_method, _ode_problem, _ode_solver, _ss_solver,
                                                        petab_ODE_cache, petab_ODESolver_cache, petab_model,
                                                        simulation_info, θ_indices, measurement_info, parameter_info,
                                                        prior_info, chunksize, split_over_conditions=false,
                                                        reuse_sensitivities=false)

    # Additional functions useful for analysing parameter estimation performance
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

    # Extract bounds and nominal parameter values
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
                                    compute_FIM!,
                                    compute_FIM,
                                    compute_simulated_values,
                                    compute_residuals,
                                    cost_method,
                                    _gradient_method,
                                    Symbol(_hessian_method),
                                    _FIM_method,
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
                                    _ode_problem,
                                    split_over_conditions,
                                    prior_info, 
                                    parameter_info, 
                                    petab_ODE_cache, 
                                    measurement_info)
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
                              compute_residuals)

    __compute_cost = let ode_problem=ode_problem, ode_solver=ode_solver, ss_solver=ss_solver,
                         petab_model=petab_model, simulation_info=simulation_info, θ_indices=θ_indices,
                         measurement_info=measurement_info, parameter_info=parameter_info, prior_info=prior_info,
                         petab_ODE_cache=petab_ODE_cache, petab_ODESolver_cache=petab_ODESolver_cache,
                         compute_residuals=compute_residuals

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
                                  split_over_conditions::Bool=false)

    @unpack θ_dynamic, θ_sd, θ_observable, θ_non_dynamic = petab_ODE_cache

    if which_method == :ForwardDiff

        iθ_sd, iθ_observable, iθ_non_dynamic, iθ_not_ode = get_index_parameters_not_ODE(θ_indices)
        compute_cost_θ_not_ODE =    let petab_model=petab_model, simulation_info=simulation_info, θ_indices=θ_indices,
                                        measurement_info=measurement_info, parameter_info=parameter_info,
                                        petab_ODE_cache=petab_ODE_cache, iθ_sd=iθ_sd, iθ_observable=iθ_observable,
                                        iθ_non_dynamic=iθ_non_dynamic

                                        (x) -> compute_cost_not_solve_ODE(x[iθ_sd],
                                                                          x[iθ_observable],
                                                                          x[iθ_non_dynamic],
                                                                          petab_model,
                                                                          simulation_info,
                                                                          θ_indices,
                                                                          measurement_info,
                                                                          parameter_info,
                                                                          petab_ODE_cache,
                                                                          exp_id_solve=[:all],
                                                                          compute_gradient_not_solve_autodiff=true)
                                    end

        if split_over_conditions == false

            # Compute gradient for parameters which are a part of the ODE-system (dynamic parameters)
            compute_cost_θ_dynamic = let θ_sd=θ_sd, θ_observable=θ_observable, θ_non_dynamic=θ_non_dynamic,
                                         ode_problem=ode_problem, ss_solver=ss_solver, petab_model=petab_model,
                                         simulation_info=simulation_info, θ_indices=θ_indices,
                                         measurement_info=measurement_info, parameter_info=parameter_info,
                                         petab_ODE_cache=petab_ODE_cache, petab_ODESolver_cache=petab_ODESolver_cache

                                        (x) -> compute_cost_solve_ODE(x,
                                                                      θ_sd,
                                                                      θ_observable,
                                                                      θ_non_dynamic,
                                                                      ode_problem,
                                                                      ode_solver,
                                                                      ss_solver,
                                                                      petab_model,
                                                                      simulation_info,
                                                                      θ_indices,
                                                                      measurement_info,
                                                                      parameter_info,
                                                                      petab_ODE_cache,
                                                                      petab_ODESolver_cache,
                                                                      compute_gradient_θ_dynamic=true,
                                                                      exp_id_solve=[:all])

                                     end

            _chunksize = isnothing(chunksize) ? ForwardDiff.Chunk(θ_dynamic) : ForwardDiff.Chunk(chunksize)
            cfg = ForwardDiff.GradientConfig(compute_cost_θ_dynamic, θ_dynamic, _chunksize)
            _compute_gradient! = let compute_cost_θ_not_ODE=compute_cost_θ_not_ODE, compute_cost_θ_dynamic=compute_cost_θ_dynamic,
                                     petab_ODE_cache=petab_ODE_cache, cfg=cfg, simulation_info=simulation_info,
                                     θ_indices=θ_indices, prior_info=prior_info

                                    (gradient, θ; isremade=false) -> compute_gradient_autodiff!(gradient,
                                                                                                θ,
                                                                                                compute_cost_θ_not_ODE,
                                                                                                compute_cost_θ_dynamic,
                                                                                                petab_ODE_cache,
                                                                                                cfg,
                                                                                                simulation_info,
                                                                                                θ_indices,
                                                                                                prior_info;
                                                                                                isremade=isremade)
                                  end

        end

        if split_over_conditions == true

            compute_cost_θ_dynamic = let θ_sd=θ_sd, θ_observable=θ_observable,  θ_non_dynamic=θ_non_dynamic,
                                         ode_problem=ode_problem, ode_solver=ode_solver, ss_solver=ss_solver,
                                         petab_model=petab_model, simulation_info=simulation_info, θ_indices=θ_indices,
                                         measurement_info=measurement_info, parameter_info=parameter_info,
                                         petab_ODE_cache=petab_ODE_cache, petab_ODESolver_cache=petab_ODESolver_cache

                                        (x, _exp_id_solve) -> compute_cost_solve_ODE(x,
                                                                                     θ_sd,
                                                                                     θ_observable,
                                                                                     θ_non_dynamic,
                                                                                     ode_problem,
                                                                                     ode_solver,
                                                                                     ss_solver,
                                                                                     petab_model,
                                                                                     simulation_info,
                                                                                     θ_indices,
                                                                                     measurement_info,
                                                                                     parameter_info,
                                                                                     petab_ODE_cache,
                                                                                     petab_ODESolver_cache,
                                                                                     compute_gradient_θ_dynamic=true,
                                                                                     exp_id_solve=_exp_id_solve)

                                      end

            _compute_gradient! = let compute_cost_θ_not_ODE=compute_cost_θ_not_ODE, compute_cost_θ_dynamic=compute_cost_θ_dynamic,
                                     petab_ODE_cache=petab_ODE_cache, simulation_info=simulation_info,
                                     θ_indices=θ_indices, prior_info=prior_info

                                     (gradient, θ) -> compute_gradient_autodiff_split!(gradient,
                                                                                       θ,
                                                                                       compute_cost_θ_not_ODE,
                                                                                       compute_cost_θ_dynamic,
                                                                                       petab_ODE_cache,
                                                                                       simulation_info,
                                                                                       θ_indices,
                                                                                       prior_info)
                                 end
        end
    end

    if which_method === :ForwardEquations

        _chunksize = isnothing(chunksize) ? ForwardDiff.Chunk(θ_dynamic) : ForwardDiff.Chunk(chunksize)
        if sensealg === :ForwardDiff && split_over_conditions == false

            _solve_ode_all_conditions! = let petab_ODESolver_cache=petab_ODESolver_cache,
                                             sol_derivative=simulation_info.ode_sols_derivatives,
                                             ode_problem=ode_problem, petab_model=petab_model,
                                             simulation_info=simulation_info, ode_solver=ode_solver,
                                             ss_solver=ss_solver, θ_indices=θ_indices, petab_ODE_cache=petab_ODE_cache

                                            (sols, θ) -> solve_ode_all_conditions!(sols,
                                                                                  θ,
                                                                                  petab_ODESolver_cache,
                                                                                  sol_derivative,
                                                                                  ode_problem,
                                                                                  petab_model,
                                                                                  simulation_info,
                                                                                  ode_solver,
                                                                                  ss_solver,
                                                                                  θ_indices,
                                                                                  petab_ODE_cache,
                                                                                  save_at_observed_t=true,
                                                                                  exp_id_solve=[:all],
                                                                                  compute_forward_sensitivites_ad=true)
                                          end
            cfg = ForwardDiff.JacobianConfig(_solve_ode_all_conditions!, petab_ODE_cache.sol_values,
                                             petab_ODE_cache.θ_dynamic, _chunksize)
        end

        if sensealg === :ForwardDiff && split_over_conditions == true

            _solve_ode_all_conditions! = let petab_ODESolver_cache=petab_ODESolver_cache,
                                             sol_derivative=simulation_info.ode_sols_derivatives,
                                             ode_problem=ode_problem, petab_model=petab_model,
                                             simulation_info=simulation_info, ode_solver=ode_solver,
                                             ss_solver=ss_solver, θ_indices=θ_indices, petab_ODE_cache=petab_ODE_cache

                                            (sols, θ, _id) -> solve_ode_all_conditions!(sols,
                                                                                        θ,
                                                                                        petab_ODESolver_cache,
                                                                                        sol_derivative,
                                                                                        ode_problem,
                                                                                        petab_model,
                                                                                        simulation_info,
                                                                                        ode_solver,
                                                                                        ss_solver,
                                                                                        θ_indices,
                                                                                        petab_ODE_cache,
                                                                                        save_at_observed_t=true,
                                                                                        exp_id_solve=_id,
                                                                                        compute_forward_sensitivites_ad=true)
                                         end
            cfg = ForwardDiff.JacobianConfig(_solve_ode_all_conditions!, petab_ODE_cache.sol_values,
                                             petab_ODE_cache.θ_dynamic, _chunksize)
        end

        if sensealg != :ForwardDiff
            _solve_ode_all_conditions! = let petab_ODESolver_cache=petab_ODESolver_cache, simulation_info=simulation_info,
                                             θ_indices=θ_indices, ode_solver=ode_solver, ss_solver=ss_solver

                                             (sols, oprob, θ_dyn, _id) -> solve_ode_all_conditions!(sols,
                                                                                                    oprob,
                                                                                                    petab_model,
                                                                                                    θ_dyn,
                                                                                                    petab_ODESolver_cache,
                                                                                                    simulation_info,
                                                                                                    θ_indices,
                                                                                                    ode_solver,
                                                                                                    ss_solver,
                                                                                                    save_at_observed_t=true,
                                                                                                    exp_id_solve=_id,
                                                                                                    compute_forward_sensitivites=true)
                                         end
            cfg = nothing
        end

        iθ_sd, iθ_observable, iθ_non_dynamic, iθ_not_ode = get_index_parameters_not_ODE(θ_indices)
        compute_cost_θ_not_ODE =    let petab_model=petab_model, simulation_info=simulation_info, θ_indices=θ_indices,
                                        measurement_info=measurement_info, parameter_info=parameter_info,
                                        petab_ODE_cache=petab_ODE_cache, iθ_sd=iθ_sd, iθ_observable=iθ_observable,
                                        iθ_non_dynamic=iθ_non_dynamic

                                        (x) -> compute_cost_not_solve_ODE(x[iθ_sd],
                                                                          x[iθ_observable],
                                                                          x[iθ_non_dynamic],
                                                                          petab_model,
                                                                          simulation_info,
                                                                          θ_indices,
                                                                          measurement_info,
                                                                          parameter_info,
                                                                          petab_ODE_cache,
                                                                          exp_id_solve=[:all],
                                                                          compute_gradient_not_solve_forward=true)
                                    end

        _compute_gradient! = let petab_model=petab_model, ode_problem=ode_problem, sensealg=sensealg, compute_cost_θ_not_ODE=compute_cost_θ_not_ODE,
                                 simulation_info=simulation_info, θ_indices=θ_indices, measurement_info=measurement_info,
                                 parameter_info=parameter_info, _solve_ode_all_conditions! =_solve_ode_all_conditions!,
                                 prior_info=prior_info, cfg=cfg, petab_ODE_cache=petab_ODE_cache, split_over_conditions=split_over_conditions

                                 (g, θ; isremade=false) -> compute_gradient_forward_equations!(g,
                                                                                               θ,
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
    end


    compute_gradient = let _compute_gradient! =_compute_gradient!
        (θ) -> begin
            gradient = zeros(Float64, length(θ))
            _compute_gradient!(gradient, θ)
            return gradient
        end
    end

    return _compute_gradient!, compute_gradient
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
                                 split_over_conditions::Bool=false,
                                 return_jacobian::Bool=false)

    @unpack θ_dynamic, θ_sd, θ_observable, θ_non_dynamic = petab_ODE_cache

    if which_method === :ForwardDiff

        if split_over_conditions == false

            _eval_hessian = let ode_problem=ode_problem, ode_solver=ode_solver, ss_solver=ss_solver,
                                petab_model=petab_model, simulation_info=simulation_info, θ_indices=θ_indices,
                                measurement_info=measurement_info, parameter_info=parameter_info, prior_info=prior_info,
                                petab_ODE_cache=petab_ODE_cache, petab_ODESolver_cache=petab_ODESolver_cache

                                (θ) -> compute_cost(θ,
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
                                                    false,
                                                    true,
                                                    false)
                            end


            _chunksize = isnothing(chunksize) ? ForwardDiff.Chunk(zeros(length(θ_indices.θ_names))) : ForwardDiff.Chunk(chunksize)
            cfg = ForwardDiff.HessianConfig(_eval_hessian, zeros(length(θ_indices.θ_names)), _chunksize)

            _compute_hessian! = let _eval_hessian=_eval_hessian, cfg=cfg, simulation_info=simulation_info,
                                   θ_indices=θ_indices, prior_info=prior_info

                                   (hessian, θ) -> compute_hessian!(hessian,
                                                                    θ,
                                                                    _eval_hessian,
                                                                    cfg,
                                                                    simulation_info,
                                                                    θ_indices,
                                                                    prior_info)
                               end
        end

        if split_over_conditions == true

            _eval_hessian = let ode_problem=ode_problem, ode_solver=ode_solver, ss_solver=ss_solver, petab_model=petab_model,
                                simulation_info=simulation_info, θ_indices=θ_indices, measurement_info=measurement_info,
                                parameter_info=parameter_info, prior_info, petab_ODE_cache=petab_ODE_cache,
                                petab_ODESolver_cache=petab_ODESolver_cache, petab_ODE_cache=petab_ODE_cache

                                (θ, _id) -> compute_cost(θ,
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
                                                         _id,
                                                         false,
                                                         true,
                                                         false)
                            end

            _compute_hessian! = let _eval_hessian=_eval_hessian, simulation_info=simulation_info,
                                   θ_indices=θ_indices, prior_info=prior_info

                                   (hessian, θ_est) -> compute_hessian_split!(hessian,
                                                                              θ,
                                                                              _eval_hessian,
                                                                              simulation_info,
                                                                              θ_indices,
                                                                              prior_info)
                                end
        end
    end

    # Functions needed for mapping θ_est to the ODE problem, and then for solving said ODE-system
    if which_method === :BlockForwardDiff

        iθ_sd, iθ_observable, iθ_non_dynamic, iθ_not_ode = get_index_parameters_not_ODE(θ_indices)
        compute_cost_θ_not_ODE = let iθ_sd=iθ_sd, iθ_observable=iθ_observable, iθ_non_dynamic=iθ_non_dynamic,
                                     petab_model=petab_model, simulation_info=simulation_info, θ_indices=θ_indices,
                                     measurement_info=measurement_info, parameter_info=parameter_info, petab_ODE_cache=petab_ODE_cache

                                    (x) -> compute_cost_not_solve_ODE(x[iθ_sd],
                                                                      x[iθ_observable],
                                                                      x[iθ_non_dynamic],
                                                                      petab_model,
                                                                      simulation_info,
                                                                      θ_indices,
                                                                      measurement_info,
                                                                      parameter_info,
                                                                      petab_ODE_cache,
                                                                      exp_id_solve=[:all],
                                                                      compute_gradient_not_solve_autodiff=true)
                                 end

        if split_over_conditions == false

            compute_cost_θ_dynamic = let θ_sd=θ_sd, θ_observable=θ_observable, θ_non_dynamic=θ_non_dynamic,
                                         ode_problem=ode_problem, ode_solver=ode_solver, ss_solver=ss_solver,
                                         petab_model=petab_model, simulation_info=simulation_info, θ_indices=θ_indices,
                                         measurement_info=measurement_info, parameter_info=parameter_info,
                                         petab_ODE_cache=petab_ODE_cache, petab_ODESolver_cache=petab_ODESolver_cache

                                        (x) -> compute_cost_solve_ODE(x,
                                                                      θ_sd,
                                                                      θ_observable,
                                                                      θ_non_dynamic,
                                                                      ode_problem,
                                                                      ode_solver,
                                                                      ss_solver,
                                                                      petab_model,
                                                                      simulation_info,
                                                                      θ_indices,
                                                                      measurement_info,
                                                                      parameter_info,
                                                                      petab_ODE_cache,
                                                                      petab_ODESolver_cache,
                                                                      compute_gradient_θ_dynamic=true,
                                                                      exp_id_solve=[:all])
                                     end

            _chunksize = isnothing(chunksize) ? ForwardDiff.Chunk(θ_dynamic) : ForwardDiff.Chunk(chunksize)
            cfg = ForwardDiff.HessianConfig(compute_cost_θ_dynamic, θ_dynamic, ForwardDiff.Chunk(chunksize))

            _compute_hessian! = let compute_cost_θ_not_ODE=compute_cost_θ_not_ODE, compute_cost_θ_dynamic=compute_cost_θ_dynamic,
                                   petab_ODE_cache=petab_ODE_cache, cfg=cfg, simulation_info=simulation_info,
                                   θ_indices=θ_indices, prior_info=prior_info

                                  (hessian, θ_est) -> compute_hessian_block!(hessian,
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
        end

        if split_over_conditions == true

            compute_cost_θ_dynamic = let θ_sd=θ_sd, θ_observable=θ_observable, θ_non_dynamic=θ_non_dynamic,
                                         ode_problem=ode_problem, ode_solver=ode_solver, ss_solver=ss_solver,
                                         petab_model=petab_model, simulation_info=simulation_info, θ_indices=θ_indices,
                                         measurement_info=measurement_info, parameter_info=parameter_info,
                                         petab_ODE_cache=petab_ODE_cache, petab_ODESolver_cache=petab_ODESolver_cache

                                         (x, _id) -> compute_cost_solve_ODE(x,
                                                                            θ_sd,
                                                                            θ_observable,
                                                                            θ_non_dynamic,
                                                                            ode_problem,
                                                                            ode_solver,
                                                                            ss_solver,
                                                                            petab_model,
                                                                            simulation_info,
                                                                            θ_indices,
                                                                            measurement_info,
                                                                            parameter_info,
                                                                            petab_ODE_cache,
                                                                            petab_ODESolver_cache,
                                                                            compute_gradient_θ_dynamic=true,
                                                                            exp_id_solve=_id)
                                      end

            _compute_hessian! = let compute_cost_θ_not_ODE=compute_cost_θ_not_ODE, compute_cost_θ_dynamic=compute_cost_θ_dynamic,
                                   petab_ODE_cache=petab_ODE_cache, simulation_info=simulation_info, θ_indices, prior_info=prior_info

                                    (hessian, θ_est) -> compute_hessian_block_split!(hessian,
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
    end

    if which_method == :GaussNewton

        if split_over_conditions == false
            _solve_ode_all_conditions! = let petab_ODESolver_cache=petab_ODESolver_cache, sols_derivatives=simulation_info.ode_sols_derivatives,
                                             ode_problem=ode_problem, simulation_info=simulation_info, ode_solver=ode_solver,
                                             ss_solver=ss_solver, θ_indices=θ_indices, petab_ODE_cache=petab_ODE_cache

                                            (sols, θ) -> solve_ode_all_conditions!(sols,
                                                                                   θ,
                                                                                   petab_ODESolver_cache,
                                                                                   sols_derivatives,
                                                                                   ode_problem,
                                                                                   petab_model,
                                                                                   simulation_info,
                                                                                   ode_solver,
                                                                                   ss_solver,
                                                                                   θ_indices,
                                                                                   petab_ODE_cache,
                                                                                   save_at_observed_t=true,
                                                                                   exp_id_solve=[:all],
                                                                                   compute_forward_sensitivites_ad=true)
                                         end
        else

            _solve_ode_all_conditions! = let petab_ODESolver_cache=petab_ODESolver_cache, sols_derivatives=simulation_info.ode_sols_derivatives,
                                             ode_problem=ode_problem, simulation_info=simulation_info, ode_solver=ode_solver,
                                             ss_solver=ss_solver, θ_indices=θ_indices, petab_ODE_cache=petab_ODE_cache

                                            (sols, θ, _id) -> solve_ode_all_conditions!(sols,
                                                                                        θ,
                                                                                        petab_ODESolver_cache,
                                                                                        sols_derivatives,
                                                                                        ode_problem,
                                                                                        petab_model,
                                                                                        simulation_info,
                                                                                        ode_solver,
                                                                                        ss_solver,
                                                                                        θ_indices,
                                                                                        petab_ODE_cache,
                                                                                        save_at_observed_t=true,
                                                                                        exp_id_solve=_id,
                                                                                        compute_forward_sensitivites_ad=true)
                                         end
        end

        _chunksize = isnothing(chunksize) ? ForwardDiff.Chunk(petab_ODE_cache.θ_dynamic) : ForwardDiff.Chunk(chunksize)
        cfg = cfg = ForwardDiff.JacobianConfig(_solve_ode_all_conditions!, petab_ODE_cache.sol_values, petab_ODE_cache.θ_dynamic, _chunksize)

        iθ_sd, iθ_observable, iθ_non_dynamic, iθ_not_ode = get_index_parameters_not_ODE(θ_indices)
        _compute_residuals_not_solve_ode! = let θ_sd=θ_sd, θ_observable=θ_observable, θ_non_dynamic=θ_non_dynamic,
                                                petab_model=petab_model, simulation_info=simulation_info,
                                                θ_indices=θ_indices, measurement_info=measurement_info,
                                                parameter_info=parameter_info, petab_ODE_cache=petab_ODE_cache, iθ_observable=iθ_observable,
                                                iθ_non_dynamic=iθ_non_dynamic, iθ_sd=iθ_sd

                                            (residuals, θ_not_ode) ->  begin
                                                            θ_sd = @view θ_not_ode[iθ_sd]
                                                            θ_observable = @view θ_not_ode[iθ_observable]
                                                            θ_non_dynamic = @view θ_not_ode[iθ_non_dynamic]
                                                            compute_residuals_not_solve_ode!(residuals,
                                                                                             θ_sd,
                                                                                             θ_observable,
                                                                                             θ_non_dynamic,
                                                                                             petab_model,
                                                                                             simulation_info,
                                                                                             θ_indices,
                                                                                             measurement_info,
                                                                                             parameter_info,
                                                                                             petab_ODE_cache;
                                                                                             exp_id_solve=[:all])
                                                                         end
                                                end

        _θ_not_ode = zeros(eltype(petab_ODE_cache.θ_dynamic), length(iθ_not_ode))
        cfg_not_solve_ode = ForwardDiff.JacobianConfig(_compute_residuals_not_solve_ode!, petab_ODE_cache.residuals_gn, _θ_not_ode, ForwardDiff.Chunk(_θ_not_ode))

        _compute_hessian! = let _compute_residuals_not_solve_ode! =_compute_residuals_not_solve_ode!,
                               petab_model=petab_model, simulation_info=simulation_info, θ_indices=θ_indices,
                               measurement_info=measurement_info, parameter_info=parameter_info,
                               _solve_ode_all_conditions! =_solve_ode_all_conditions!, prior_info=prior_info, cfg=cfg,
                               cfg_not_solve_ode=cfg_not_solve_ode, reuse_sensitivities=reuse_sensitivities,
                               return_jacobian=return_jacobian, split_over_conditions=split_over_conditions

                              (hessian, θ; isremade=false) -> compute_GaussNewton_hessian!(hessian,
                                                                                           θ,
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
    end

        compute_hessian = (θ) -> begin
                                hessian = zeros(Float64, length(θ), length(θ))
                                _compute_hessian!(hessian, θ)
                                return hessian
                            end

    return _compute_hessian!, compute_hessian
end


function get_ODE_forward_equations(ode_problem::ODEProblem,
                                   sensealg_forward_equations)::ODEProblem
    return ode_problem
end
