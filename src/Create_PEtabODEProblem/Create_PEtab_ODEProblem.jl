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
    experimentalConditions, measurementsData, parametersData, observablesData = readPEtabFiles(petab_model)
    parameterInfo = processParameters(parametersData, custom_parameter_values=custom_parameter_values)
    measurementInfo = processMeasurements(measurementsData, observablesData)
    θ_indices = computeIndicesθ(parameterInfo, measurementInfo, petab_model)
    priorInfo = processPriors(θ_indices, parametersData)

    # In case not specified by the user set ODE, gradient and Hessian options
    nODEs = length(states(petab_model.system))
    if nODEs ≤ 15 && length(θ_indices.θ_dynamicNames) ≤ 20
        modelSize = :Small
    elseif nODEs ≤ 50 && length(θ_indices.θ_dynamicNames) ≤ 69
        modelSize = :Medium
    else
        modelSize = :Large
    end
    _gradient_method = setGradientMethod(gradient_method, modelSize, reuse_sensitivities)
    _hessian_method = setHessianMethod(hessian_method, modelSize)
    _sensealg = setSensealg(sensealg, Val(_gradient_method))
    _ode_solver = setODESolver(ode_solver, modelSize, _gradient_method)
    _ode_solver_gradient = isnothing(ode_solver_gradient) ? deepcopy(_ode_solver) : ode_solver_gradient
    _ss_solver = setSteadyStateSolver(ss_solver, _ode_solver)
    _ss_solver_gradient = isnothing(ss_solver_gradient) ? deepcopy(_ss_solver) : ss_solver_gradient
    _sparse_jacobian = !isnothing(sparse_jacobian) ? sparse_jacobian : (modelSize === :Large ? true : false)

    simulation_info = processSimulationInfo(petab_model, measurementInfo, sensealg=_sensealg)

    # The time-span 5e3 is overwritten when performing forward simulations. As we solve an expanded system with the forward
    # equations, we need a seperate problem for it
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && @printf(" Building ODEProblem from ODESystem ...")
    timeTake = @elapsed begin
    # Set model parameter values to those in the PeTab parameter to ensure correct constant parameters
    setParamToFileValues!(petab_model.parameter_map, petab_model.state_map, parameterInfo)
    if petab_model.system isa ODESystem
        __odeProblem = ODEProblem{true, specialize_level}(petab_model.system, petab_model.state_map, [0.0, 5e3], petab_model.parameter_map, jac=true, sparse=_sparse_jacobian)
    else
        # For reaction systems this bugs out if I try to set specialize_level (specifially state-map and parameter-map are not 
        # made into vectors)
        __odeProblem = ODEProblem(petab_model.system, zeros(Float64, length(petab_model.state_map)), [0.0, 5e3], petab_model.parameter_map, jac=true, sparse=_sparse_jacobian)
    end
    _odeProblem = remake(__odeProblem, p = convert.(Float64, __odeProblem.p), u0 = convert.(Float64, __odeProblem.u0))
    end
    verbose == true && @printf(" done. Time = %.1e\n", timeTake)

    # Needed to properly initalise steady-state solver options with model Jacobian etc...
    _ss_solver = _get_steady_state_solver(_ss_solver, _odeProblem, _ss_solver.abstol, _ss_solver.reltol, _ss_solver.maxiters)
    _ss_solver_gradient = _get_steady_state_solver(_ss_solver_gradient, _odeProblem, _ss_solver_gradient.abstol, _ss_solver_gradient.reltol, _ss_solver_gradient.maxiters)

    # If we are computing the cost, gradient and hessians accross several processes we need to send ODEProblem, and
    # PEtab structs to each process
    if n_processes > 1
        jobs, results = setUpProcesses(petab_model, ode_solver, solverAbsTol, solverRelTol, odeSolverAdjoint, sensealgAdjoint,
                                       sensealg_ss, solverAdjointAbsTol, solverAdjointRelTol, odeSolverForwardEquations,
                                       sensealgForwardEquations, parameterInfo, measurementInfo, simulation_info, θ_indices,
                                       priorInfo, odeProblem, chunksize)
    else
        jobs, results = nothing, nothing
    end

    petabODECache = PEtabODEProblemCache(_gradient_method, _hessian_method, petab_model, _sensealg, measurementInfo, simulation_info, θ_indices, chunksize)
    petabODESolverCache = createPEtabODESolverCache(_gradient_method, _hessian_method, petab_model, simulation_info, θ_indices, chunksize)

    # To get multiple dispatch to work correctly 
    _cost_method = cost_method === :Zygote ? Val(:Zygote) : cost_method
    __gradient_method = _gradient_method === :Zygote ? Val(:Zygote) : _gradient_method

    # The cost (likelihood) can either be computed in the standard way or the Zygote way. The second consumes more
    # memory as in-place mutations are not compatible with Zygote
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building cost function for method ", string(cost_method), " ...")
    bBuild = @elapsed compute_cost = setUpCost(_cost_method, _odeProblem, _ode_solver, _ss_solver, petabODECache, petabODESolverCache,
                                              petab_model, simulation_info, θ_indices, measurementInfo, parameterInfo, priorInfo,
                                              _sensealg, n_processes, jobs, results, false)

    compute_chi2 = (θ; asArray=false) -> begin
        _ = compute_cost(θ)
        if asArray == false
            return sum(measurementInfo.chi2Values)
        else
            return measurementInfo.chi2Values
        end
    end
    compute_residuals = (θ; asArray=false) -> begin
        _ = compute_cost(θ)
        if asArray == false
            return sum(measurementInfo.residuals)
        else
            return measurementInfo.residuals
        end
    end
    compute_simulated_values = (θ) -> begin
        _ = compute_cost(θ)
        return measurementInfo.simulatedValues
    end
    verbose == true && @printf(" done. Time = %.1e\n", bBuild)

    # The gradient can either be computed via autodiff, forward sensitivity equations, adjoint sensitivity equations
    # and Zygote
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building gradient function for method ", string(_gradient_method), " ...")
    _odeProblemGradient = gradient_method === :ForwardEquations ? getODEProblemForwardEquations(_odeProblem, sensealg) : getODEProblemForwardEquations(_odeProblem, :NoSpecialProblem)
    bBuild = @elapsed compute_gradient! = setUpGradient(__gradient_method, _odeProblemGradient, _ode_solver_gradient,
        _ss_solver_gradient, petabODECache, petabODESolverCache, petab_model, simulation_info, θ_indices,
        measurementInfo, parameterInfo, _sensealg, priorInfo, chunksize=chunksize, n_processes=n_processes,
        jobs=jobs, results=results, split_over_conditions=split_over_conditions, sensealg_ss=sensealg_ss)
    # Non in-place gradient
    compute_gradient = (θ) -> begin
        gradient = zeros(Float64, length(θ))
        compute_gradient!(gradient, θ)
        return gradient
    end
    verbose == true && @printf(" done. Time = %.1e\n", bBuild)

    # The Hessian can either be computed via automatic differentation, or approximated via a block approximation or the
    # Gauss Newton method
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building hessian function for method ", string(_hessian_method), " ...")
    bBuild = @elapsed compute_hessian! = setUpHessian(_hessian_method, _odeProblem, _ode_solver, _ss_solver,
        petabODECache, petabODESolverCache, petab_model, simulation_info, θ_indices, measurementInfo, parameterInfo,
        priorInfo, chunksize, n_processes=n_processes, jobs=jobs, results=results,
        split_over_conditions=split_over_conditions, reuse_sensitivities=reuse_sensitivities)
    # Non-inplace Hessian
    compute_hessian = (θ) -> begin
                                hessian = zeros(Float64, length(θ), length(θ))
                                compute_hessian!(hessian, θ)
                                return hessian
                            end
    verbose == true && @printf(" done. Time = %.1e\n", bBuild)

    # Nominal parameter values + parameter bounds on parameter-scale (transformed)
    θ_names = θ_indices.θ_names
    lower_bounds = [parameterInfo.lowerBound[findfirst(x -> x == θ_names[i], parameterInfo.parameterId)] for i in eachindex(θ_names)]
    upper_bounds = [parameterInfo.upperBound[findfirst(x -> x == θ_names[i], parameterInfo.parameterId)] for i in eachindex(θ_names)]
    θ_nominal = [parameterInfo.nominalValue[findfirst(x -> x == θ_names[i], parameterInfo.parameterId)] for i in eachindex(θ_names)]
    transformθ!(lower_bounds, θ_names, θ_indices, reverseTransform=true)
    transformθ!(upper_bounds, θ_names, θ_indices, reverseTransform=true)
    θ_nominalT = transformθ(θ_nominal, θ_names, θ_indices, reverseTransform=true)

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


function setUpCost(whichMethod::Symbol,
                   odeProblem::ODEProblem,
                   ode_solver::ODESolver,
                   ss_solver::SteadyStateSolver,
                   petabODECache::PEtabODEProblemCache,
                   petabODESolverCache::PEtabODESolverCache,
                   petab_model::PEtabModel,
                   simulation_info::SimulationInfo,
                   θ_indices::ParameterIndices,
                   measurementInfo::MeasurementsInfo,
                   parameterInfo::ParametersInfo,
                   priorInfo::PriorInfo,
                   sensealg,
                   n_processes,
                   jobs,
                   results,
                   compute_residuals)

    # Functions needed for mapping θ_est to the ODE problem, and then for solving said ODE-system
    if whichMethod == :Standard && n_processes == 1

        __compute_cost = let θ_indices=θ_indices, ode_solver=ode_solver, priorInfo=priorInfo
                            (θ_est) -> compute_cost(θ_est,
                                               odeProblem,
                                               ode_solver,
                                               ss_solver,
                                               petab_model,
                                               simulation_info,
                                               θ_indices,
                                               measurementInfo,
                                               parameterInfo,
                                               priorInfo,
                                               petabODECache,
                                               petabODESolverCache,
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


function setUpGradient(whichMethod::Symbol,
                       odeProblem::ODEProblem,
                       ode_solver::ODESolver,
                       ss_solver::SteadyStateSolver,
                       petabODECache::PEtabODEProblemCache,
                       petabODESolverCache::PEtabODESolverCache,
                       petab_model::PEtabModel,
                       simulation_info::SimulationInfo,
                       θ_indices::ParameterIndices,
                       measurementInfo::MeasurementsInfo,
                       parameterInfo::ParametersInfo,
                       sensealg,
                       priorInfo::PriorInfo;
                       chunksize::Union{Nothing, Int64}=nothing,
                       sensealg_ss=nothing,
                       n_processes::Int64=1,
                       jobs=nothing,
                       results=nothing,
                       split_over_conditions::Bool=false)

    θ_dynamic = petabODECache.θ_dynamic
    θ_sd = petabODECache.θ_sd
    θ_observable = petabODECache.θ_observable
    θ_nonDynamic = petabODECache.θ_nonDynamic

    if whichMethod == :ForwardDiff && n_processes == 1

        iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = getIndicesParametersNotInODESystem(θ_indices)
        compute_costNotODESystemθ = (x) -> compute_costNotSolveODE(x[iθ_sd], x[iθ_observable], x[iθ_nonDynamic],
                                                                 petab_model, simulation_info, θ_indices, measurementInfo,
                                                                 parameterInfo, petabODECache, expIDSolve=[:all],
                                                                 compute_gradientNotSolveAutoDiff=true)

        if split_over_conditions == false
            # Compute gradient for parameters which are a part of the ODE-system (dynamic parameters)
            compute_costDynamicθ = (x) -> compute_costSolveODE(x, θ_sd, θ_observable, θ_nonDynamic, odeProblem,
                ode_solver, ss_solver, petab_model, simulation_info, θ_indices, measurementInfo,
                parameterInfo, petabODECache, petabODESolverCache, compute_gradientDynamicθ=true, expIDSolve=[:all])

            _chunksize = isnothing(chunksize) ? ForwardDiff.Chunk(θ_dynamic) : ForwardDiff.Chunk(chunksize)
            cfg = ForwardDiff.GradientConfig(compute_costDynamicθ, θ_dynamic, _chunksize)

            _compute_gradient! = (gradient, θ_est; isRemade=false) -> compute_gradientAutoDiff!(gradient,
                                                                                              θ_est,
                                                                                              compute_costNotODESystemθ,
                                                                                              compute_costDynamicθ,
                                                                                              petabODECache,
                                                                                              cfg,
                                                                                              simulation_info,
                                                                                              θ_indices,
                                                                                              priorInfo;
                                                                                              isRemade=isRemade)
        end

        if split_over_conditions == true

            compute_costDynamicθ = (x, _expIdSolve) -> compute_costSolveODE(x, θ_sd, θ_observable, θ_nonDynamic, odeProblem,
                ode_solver, ss_solver, petab_model, simulation_info, θ_indices, measurementInfo, parameterInfo,
                petabODECache, petabODESolverCache, compute_gradientDynamicθ=true, expIDSolve=_expIdSolve)

            _compute_gradient! = (gradient, θ_est) -> compute_gradientAutoDiffSplitOverConditions!(gradient,
                                                                                                 θ_est,
                                                                                                 compute_costNotODESystemθ,
                                                                                                 compute_costDynamicθ,
                                                                                                 petabODECache,
                                                                                                 simulation_info,
                                                                                                 θ_indices,
                                                                                                 priorInfo)
        end
    end

    if whichMethod === :ForwardEquations && n_processes == 1

        _chunksize = isnothing(chunksize) ? ForwardDiff.Chunk(θ_dynamic) : ForwardDiff.Chunk(chunksize)
        if sensealg === :ForwardDiff && split_over_conditions == false
            _solveODEAllExperimentalConditions! = (odeSolutionValues, θ) -> begin
                solveODEAllExperimentalConditions!(odeSolutionValues, θ, petabODESolverCache,
                    simulation_info.odeSolutionsDerivatives, odeProblem, petab_model, simulation_info, ode_solver,
                    ss_solver, θ_indices, petabODECache, onlySaveAtObservedTimes=true, expIDSolve=[:all],
                    computeForwardSensitivitesAD=true)
            end
            cfg = ForwardDiff.JacobianConfig(_solveODEAllExperimentalConditions!, petabODECache.odeSolutionValues,
                petabODECache.θ_dynamic, _chunksize)
        end

        if sensealg === :ForwardDiff && split_over_conditions == true

            _solveODEAllExperimentalConditions! = (odeSolutionValues, θ, _expIdSolve) -> begin
                solveODEAllExperimentalConditions!(odeSolutionValues, θ, petabODESolverCache,
                    simulation_info.odeSolutionsDerivatives, odeProblem, petab_model, simulation_info, ode_solver,
                    ss_solver, θ_indices, petabODECache, onlySaveAtObservedTimes=true, expIDSolve=_expIdSolve,
                    computeForwardSensitivitesAD=true)
            end
            cfg = ForwardDiff.JacobianConfig(_solveODEAllExperimentalConditions!, petabODECache.odeSolutionValues,
                petabODECache.θ_dynamic, _chunksize)
        end

        if sensealg != :ForwardDiff
            _solveODEAllExperimentalConditions! = (odeSolutions, odeProblem, θ_dynamic, _expIDSolve) -> begin
                solveODEAllExperimentalConditions!(odeSolutions, odeProblem, petab_model, θ_dynamic, petabODESolverCache,
                    simulation_info, θ_indices, ode_solver, ss_solver, onlySaveAtObservedTimes=true,
                    expIDSolve=_expIDSolve, computeForwardSensitivites=true)
                end
            cfg = nothing
        end

        iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = getIndicesParametersNotInODESystem(θ_indices)
        compute_costNotODESystemθ = (x) -> compute_costNotSolveODE(x[iθ_sd], x[iθ_observable], x[iθ_nonDynamic],
            petab_model, simulation_info, θ_indices, measurementInfo, parameterInfo, petabODECache, expIDSolve=[:all],
            compute_gradientNotSolveForward=true)

        _compute_gradient! = (gradient, θ_est; isRemade=false) -> compute_gradientForwardEquations!(gradient,
                                                                                                 θ_est,
                                                                                                 compute_costNotODESystemθ,
                                                                                                 petab_model,
                                                                                                 odeProblem,
                                                                                                 sensealg,
                                                                                                 simulation_info,
                                                                                                 θ_indices,
                                                                                                 measurementInfo,
                                                                                                 parameterInfo,
                                                                                                 _solveODEAllExperimentalConditions!,
                                                                                                 priorInfo,
                                                                                                 cfg,
                                                                                                 petabODECache,
                                                                                                 expIDSolve=[:all],
                                                                                                 split_over_conditions=split_over_conditions,
                                                                                                 isRemade=isRemade)
    end

    if false

        _compute_gradient! = (gradient, θ_est) -> begin
                                                    gradient .= 0.0
                                                    @inbounds for i in n_processes:-1:1
                                                        @async put!(jobs[i], tuple(θ_est, whichMethod))
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


function setUpHessian(whichMethod::Symbol,
                      odeProblem::ODEProblem,
                      ode_solver::ODESolver,
                      ss_solver::SteadyStateSolver,
                      petabODECache::PEtabODEProblemCache,
                      petabODESolverCache::PEtabODESolverCache,
                      petab_model::PEtabModel,
                      simulation_info::SimulationInfo,
                      θ_indices::ParameterIndices,
                      measurementInfo::MeasurementsInfo,
                      parameterInfo::ParametersInfo,
                      priorInfo::PriorInfo,
                      chunksize::Union{Nothing, Int64};
                      reuse_sensitivities::Bool=false,
                      n_processes::Int64=1,
                      jobs=nothing,
                      results=nothing,
                      split_over_conditions::Bool=false,
                      returnJacobian::Bool=false)

    θ_dynamic = petabODECache.θ_dynamic
    θ_sd = petabODECache.θ_sd
    θ_observable = petabODECache.θ_observable
    θ_nonDynamic = petabODECache.θ_nonDynamic

    if whichMethod === :ForwardDiff

        if split_over_conditions == false

            _evalHessian = (θ_est) -> compute_cost(θ_est, odeProblem, ode_solver, ss_solver, petab_model,
                simulation_info, θ_indices, measurementInfo, parameterInfo, priorInfo, petabODECache,
                petabODESolverCache, [:all], false, true, false)

            _chunksize = isnothing(chunksize) ? ForwardDiff.Chunk(zeros(length(θ_indices.θ_names))) : ForwardDiff.Chunk(chunksize)
            cfg = ForwardDiff.HessianConfig(_evalHessian, zeros(length(θ_indices.θ_names)), _chunksize)

            _compute_hessian = (hessian, θ_est) -> compute_hessian!(hessian,
                                                                  θ_est,
                                                                  _evalHessian,
                                                                  cfg,
                                                                  simulation_info,
                                                                  θ_indices,
                                                                  priorInfo)
        end

        if split_over_conditions == true
            _evalHessian = (θ_est) -> compute_cost(θ_est, odeProblem, ode_solver, ss_solver, petab_model,
                                                  simulation_info, θ_indices, measurementInfo, parameterInfo, priorInfo,
                                                  petabODECache, petabODESolverCache, [:all], false, true, false)
            _compute_hessian = (hessian, θ_est) -> compute_hessianSplitOverConditions!(hessian,
                                                                                     θ_est,
                                                                                     _evalHessian,
                                                                                     simulation_info,
                                                                                     θ_indices,
                                                                                     priorInfo)
        end

    end

    # Functions needed for mapping θ_est to the ODE problem, and then for solving said ODE-system
    if whichMethod === :BlockForwardDiff

        iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = getIndicesParametersNotInODESystem(θ_indices)
        compute_costNotODESystemθ = (x) -> compute_costNotSolveODE(x[iθ_sd], x[iθ_observable], x[iθ_nonDynamic],
                                                                 petab_model, simulation_info, θ_indices, measurementInfo,
                                                                 parameterInfo, petabODECache, expIDSolve=[:all],
                                                                 compute_gradientNotSolveAutoDiff=true)

        if split_over_conditions == false

            compute_costDynamicθ = (x) -> compute_costSolveODE(x, θ_sd, θ_observable, θ_nonDynamic, odeProblem,
                ode_solver, ss_solver, petab_model, simulation_info, θ_indices, measurementInfo,
                parameterInfo, petabODECache, petabODESolverCache, compute_gradientDynamicθ=true, expIDSolve=[:all])

            _chunksize = isnothing(chunksize) ? ForwardDiff.Chunk(θ_dynamic) : ForwardDiff.Chunk(chunksize)
            cfg = ForwardDiff.HessianConfig(compute_costDynamicθ, θ_dynamic, ForwardDiff.Chunk(chunksize))

            _compute_hessian = (hessian, θ_est) -> compute_hessianBlockApproximation!(hessian,
                                                                                        θ_est,
                                                                                        compute_costNotODESystemθ,
                                                                                        compute_costDynamicθ,
                                                                                        petabODECache,
                                                                                        cfg,
                                                                                        simulation_info,
                                                                                        θ_indices,
                                                                                        priorInfo,
                                                                                        expIDSolve=[:all])
        end

        if split_over_conditions == true

            compute_costDynamicθ = (x, _expIDSolve) -> compute_costSolveODE(x, θ_sd, θ_observable, θ_nonDynamic, odeProblem, ode_solver,
                ss_solver, petab_model, simulation_info, θ_indices, measurementInfo, parameterInfo, petabODECache,
                petabODESolverCache, compute_gradientDynamicθ=true, expIDSolve=_expIDSolve)

            _compute_hessian = (hessian, θ_est) -> compute_hessianBlockApproximationSplitOverConditions!(hessian,
                                                                                                       θ_est,
                                                                                                       compute_costNotODESystemθ,
                                                                                                       compute_costDynamicθ,
                                                                                                       petabODECache,
                                                                                                       simulation_info,
                                                                                                       θ_indices,
                                                                                                       priorInfo,
                                                                                                       expIDSolve=[:all])
        end
    end

    if whichMethod == :GaussNewton && n_processes == 1

        changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petab_model, θ_indices)
        changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petab_model, θ_indices)

        if split_over_conditions == false
            _solveODEAllExperimentalConditions! = (odeSolutionValues, θ) -> solveODEAllExperimentalConditions!(odeSolutionValues, θ, petabODESolverCache, simulation_info.odeSolutionsDerivatives, odeProblem, petab_model, simulation_info, ode_solver, ss_solver, θ_indices, petabODECache, onlySaveAtObservedTimes=true, expIDSolve=[:all], computeForwardSensitivitesAD=true)
        else
            _solveODEAllExperimentalConditions! = (odeSolutionValues, θ, _expIdSolve) -> solveODEAllExperimentalConditions!(odeSolutionValues, θ, petabODESolverCache, simulation_info.odeSolutionsDerivatives, odeProblem, petab_model, simulation_info, ode_solver, ss_solver, θ_indices, petabODECache, onlySaveAtObservedTimes=true, expIDSolve=_expIdSolve, computeForwardSensitivitesAD=true)
        end

        _chunksize = isnothing(chunksize) ? ForwardDiff.Chunk(petabODECache.θ_dynamic) : ForwardDiff.Chunk(chunksize)
        cfg = cfg = ForwardDiff.JacobianConfig(_solveODEAllExperimentalConditions!, petabODECache.odeSolutionValues, petabODECache.θ_dynamic, _chunksize)

        iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = getIndicesParametersNotInODESystem(θ_indices)
        _compute_residualsNotSolveODE! = (residuals, θ_notOdeSystem) ->  begin
            θ_sd = @view θ_notOdeSystem[iθ_sd]
            θ_observable = @view θ_notOdeSystem[iθ_observable]
            θ_nonDynamic = @view θ_notOdeSystem[iθ_nonDynamic]
            compute_residualsNotSolveODE!(residuals, θ_sd, θ_observable, θ_nonDynamic, petab_model, simulation_info,
                                         θ_indices, measurementInfo, parameterInfo, petabODECache;
                                         expIDSolve=[:all])
                                                                        end

        _θ_notOdeSystem = zeros(eltype(petabODECache.θ_dynamic), length(iθ_notOdeSystem))
        cfgNotSolveODE = ForwardDiff.JacobianConfig(_compute_residualsNotSolveODE!, petabODECache.residualsGN, _θ_notOdeSystem, ForwardDiff.Chunk(_θ_notOdeSystem))

        _compute_hessian = (hessian, θ_est; isRemade=false) -> computeGaussNewtonHessianApproximation!(hessian,
                                                                                                      θ_est,
                                                                                                      odeProblem,
                                                                                                      _compute_residualsNotSolveODE!,
                                                                                                      petab_model,
                                                                                                      simulation_info,
                                                                                                      θ_indices,
                                                                                                      measurementInfo,
                                                                                                      parameterInfo,
                                                                                                      _solveODEAllExperimentalConditions!,
                                                                                                      priorInfo,
                                                                                                      cfg,
                                                                                                      cfgNotSolveODE,
                                                                                                      petabODECache,
                                                                                                      expIDSolve=[:all],
                                                                                                      reuse_sensitivities=reuse_sensitivities,
                                                                                                      returnJacobian=returnJacobian,
                                                                                                      split_over_conditions=split_over_conditions,
                                                                                                      isRemade=isRemade)
    end

    if false

        _compute_hessian = (hessian, θ_est) ->   begin
                                                    hessian .= 0.0
                                                    @inbounds for i in n_processes:-1:1
                                                        @async put!(jobs[i], tuple(θ_est, whichMethod))
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


function getODEProblemForwardEquations(odeProblem::ODEProblem,
                                       sensealgForwardEquations)::ODEProblem
    return odeProblem
end


function PEtabODEProblemCache(gradient_method::Symbol,
                                    hessian_method::Union{Symbol, Nothing},
                                    petab_model::PEtabModel,
                                    sensealg,
                                    measurementInfo::MeasurementsInfo,
                                    simulation_info::SimulationInfo,
                                    θ_indices::ParameterIndices,
                                    _chunksize)::PEtabODEProblemCache

    θ_dynamic = zeros(Float64, length(θ_indices.iθ_dynamic))
    θ_observable = zeros(Float64, length(θ_indices.iθ_observable))
    θ_sd = zeros(Float64, length(θ_indices.iθ_sd))
    θ_nonDynamic = zeros(Float64, length(θ_indices.iθ_nonDynamic))

    levelCache = 0
    if hessian_method ∈ [:ForwardDiff, :BlockForwardDiff, :GaussNewton]
        levelCache = 2
    elseif gradient_method ∈ [:ForwardDiff, :ForwardEquations]
        levelCache = 1
    else
        levelCache = 0
    end

    if isnothing(_chunksize)
        chunksize = length(θ_indices.iθ_dynamic) > 10 ? 10 : length(θ_indices.iθ_dynamic)
    else
        chunksize = _chunksize
    end

    _θ_dynamicT = zeros(Float64, length(θ_indices.iθ_dynamic))
    _θ_observableT = zeros(Float64, length(θ_indices.iθ_observable))
    _θ_sdT = zeros(Float64, length(θ_indices.iθ_sd))
    _θ_nonDynamicT = zeros(Float64, length(θ_indices.iθ_nonDynamic))
    θ_dynamicT = DiffCache(_θ_dynamicT, chunksize, levels=levelCache)
    θ_observableT = DiffCache(_θ_observableT, chunksize, levels=levelCache)
    θ_sdT = DiffCache(_θ_sdT, chunksize, levels=levelCache)
    θ_nonDynamicT = DiffCache(_θ_nonDynamicT, chunksize, levels=levelCache)

    gradientDyanmicθ = zeros(Float64, length(θ_dynamic))
    gradientNotODESystemθ = zeros(Float64, length(θ_indices.iθ_notOdeSystem))

    # For forward sensitivity equations and adjoint sensitivity analysis we need to
    # compute partial derivatives symbolically. Here the helping vectors are pre-allocated
    if gradient_method ∈ [:Adjoint, :ForwardEquations] || hessian_method ∈ [:GaussNewton]
        nModelStates = length(states(petab_model.system))
        nModelParameters = length(parameters(petab_model.system))
        ∂h∂u = zeros(Float64, nModelStates)
        ∂σ∂u = zeros(Float64, nModelStates)
        ∂h∂p = zeros(Float64, nModelParameters)
        ∂σ∂p = zeros(Float64, nModelParameters)
        ∂G∂p = zeros(Float64, nModelParameters)
        ∂G∂p_ = zeros(Float64, nModelParameters)
        ∂G∂u = zeros(Float64, nModelStates)
        p = zeros(Float64, nModelParameters)
        u = zeros(Float64, nModelStates)
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
        nModelStates = length(states(petab_model.system))
        nTimePointsSaveAt = sum(length(simulation_info.timeObserved[experimentalConditionId]) for experimentalConditionId in simulation_info.experimentalConditionId)
        S = zeros(Float64, (nTimePointsSaveAt*nModelStates, length(θ_indices.θ_dynamicNames)))
        odeSolutionValues = zeros(Float64, nModelStates, nTimePointsSaveAt)
    else
        S = zeros(Float64, (0, 0))
        odeSolutionValues = zeros(Float64, (0, 0))
    end

    if hessian_method === :GaussNewton
        jacobianGN = zeros(Float64, length(θ_indices.θ_names), length(measurementInfo.time))
        residualsGN = zeros(Float64, length(measurementInfo.time))
    else
        jacobianGN = zeros(Float64, (0, 0))
        residualsGN = zeros(Float64, 0)
    end

    if gradient_method === :ForwardEquations || hessian_method === :GaussNewton
        _gradient = zeros(Float64, length(θ_indices.iθ_dynamic))
    else
        _gradient = zeros(Float64, 0)
    end

    if gradient_method === :Adjoint
        nModelStates = length(states(petab_model.system))
        nModelParameters = length(parameters(petab_model.system))
        du = zeros(Float64, nModelStates)
        dp = zeros(Float64, nModelParameters)
        _gradientAdjoint = zeros(Float64, nModelParameters)
        St0 = zeros(Float64, (nModelStates, nModelParameters))
    else
        du = zeros(Float64, 0)
        dp = zeros(Float64, 0)
        _gradientAdjoint = zeros(Float64, 0)
        St0 = zeros(Float64, (0, 0))
    end

    # Allocate arrays to track if θ_dynamic should be permuted prior and post gradient compuations. This feature
    # is used if PEtabODEProblem is remade (via remake) to compute the gradient of a problem with reduced number
    # of parameters where to run fewer chunks with ForwardDiff.jl we only run enough chunks to reach nθ_dynamicEst
    θ_dynamicInputOrder::Vector{Int64} = collect(1:length(θ_dynamic))
    θ_dynamicOutputOrder::Vector{Int64} = collect(1:length(θ_dynamic))
    nθ_dynamicEst::Vector{Int64} = Int64[length(θ_dynamic)]

    petabODECache = PEtabODEProblemCache(θ_dynamic,
                                         θ_sd,
                                         θ_observable,
                                         θ_nonDynamic,
                                         θ_dynamicT,
                                         θ_sdT,
                                         θ_observableT,
                                         θ_nonDynamicT,
                                         gradientDyanmicθ,
                                         gradientNotODESystemθ,
                                         jacobianGN,
                                         residualsGN,
                                         _gradient,
                                         _gradientAdjoint,
                                         St0,
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
                                         odeSolutionValues,
                                         θ_dynamicInputOrder,
                                         θ_dynamicOutputOrder,
                                         nθ_dynamicEst)

    return petabODECache
end


function createPEtabODESolverCache(gradient_method::Symbol,
                                   hessian_method::Union{Symbol, Nothing},
                                   petab_model::PEtabModel,
                                   simulation_info::SimulationInfo,
                                   θ_indices::ParameterIndices,
                                   _chunksize)::PEtabODESolverCache

    nModelStates = length(states(petab_model.system))
    nModelParameters = length(parameters(petab_model.system))

    levelCache = 0
    if hessian_method ∈ [:ForwardDiff, :BlockForwardDiff, :GaussNewton]
        levelCache = 2
    elseif gradient_method ∈ [:ForwardDiff, :ForwardEquations]
        levelCache = 1
    else
        levelCache = 0
    end

    if isnothing(_chunksize)
        chunksize = length(θ_indices.iθ_dynamic) > 10 ? 10 : length(θ_indices.iθ_dynamic)
    else
        chunksize = _chunksize
    end

    if simulation_info.haspreEquilibrationConditionId == true
        conditionsToSimulateOver = unique(vcat(simulation_info.preEquilibrationConditionId, simulation_info.experimentalConditionId))
    else
        conditionsToSimulateOver = unique(simulation_info.experimentalConditionId)
    end

    _pODEProblemCache = Tuple(DiffCache(zeros(Float64, nModelParameters), chunksize, levels=levelCache) for i in eachindex(conditionsToSimulateOver))
    _u0Cache = Tuple(DiffCache(zeros(Float64, nModelStates), chunksize, levels=levelCache) for i in eachindex(conditionsToSimulateOver))
    pODEProblemCache::Dict = Dict([(conditionsToSimulateOver[i], _pODEProblemCache[i]) for i in eachindex(_pODEProblemCache)])
    u0Cache::Dict = Dict([(conditionsToSimulateOver[i], _u0Cache[i]) for i in eachindex(_u0Cache)])

    return PEtabODESolverCache(pODEProblemCache, u0Cache)
end
