"""
    createPEtabODEProblem(petabModel::PEtabModel; <keyword arguments>)

Given a `PEtabModel` creates a `PEtabODEProblem` with potential user specified options.

The keyword arguments (described below) allow the user to choose cost, gradient, and Hessian methods, ODE solver options, and other tuneable options that can potentially make computations more efficient for some "edge-case" models. Please refer to the documentation for guidance on selecting the most efficient options for different types of models.

If a keyword argument is not set, a suitable default option is chosen based on the number of model parameters.
!!! note
    Every problem is unique, so even though the default settings often work well they might not be optimal.

# Keyword arguments
- `odeSolverOptions::ODESolverOptions`: Options for the ODE solver when computing the cost, such as solver and tolerances.
- `odeSolverGradientOptions::ODESolverOptions`: Options for the ODE solver when computing the gradient, such as the ODE solver options used in adjoint sensitivity analysis. Defaults to `odeSolverOptions` if not set to nothing.
- `ssSolverOptions::SteadyStateSolverOptions`: Options for finding steady-state for models with pre-equilibrium. Steady-state can be found via simulation or rootfinding, which can be set using `SteadyStateSolverOptions` (see documentation). If not set, defaults to simulation with `wrms < 1` termination.
- `ssSolverGradientOptions::SteadyStateSolverOptions`: Options for finding steady-state for models with pre-equilibrium when computing gradients. Defaults to `ssSolverOptions` value if not set.
- `costMethod::Symbol=:Standard`: Method for computing the cost (objective). Two options are available: `:Standard`, which is the most efficient, and `:Zygote`, which is less efficient but compatible with the Zygote automatic differentiation library.
- `gradientMethod=nothing`: Method for computing the gradient of the objective. Four options are available:
    * `:ForwardDiff`: Compute the gradient via forward-mode automatic differentiation using ForwardDiff.jl. Most efficient for models with ≤50 parameters. The number of chunks can be optionally set using `chunkSize`.
    * `:ForwardEquations`: Compute the gradient via the model sensitivities, where `sensealg` specifies how to solve for the sensitivities. Most efficient when the Hessian is approximated using the Gauss-Newton method and when the optimizer can reuse the sensitivities (`reuseS`) from gradient computations in Hessian computations (e.g., when the optimizer always computes the gradient before the Hessian).
    * `:Adjoint`: Compute the gradient via adjoint sensitivity analysis, where `sensealg` specifies which algorithm to use. Most efficient for large models (≥75 parameters).
    * `:Zygote`: Compute the gradient via the Zygote package, where `sensealg` specifies which sensitivity algorithm to use when solving the ODE model. This is the most inefficient option and not recommended.
- `hessianMethod=nothing`: method for computing the Hessian of the cost. There are three available options:
    * `:ForwardDiff`: Compute the Hessian via forward-mode automatic differentiation using ForwardDiff.jl. This is often only computationally feasible for models with ≤20 parameters but can greatly improve optimizer convergence.
    * `:BlockForwardDiff`: Compute the Hessian block approximation via forward-mode automatic differentiation using ForwardDiff.jl. The approximation consists of two block matrices: the first is the Hessian for only the dynamic parameters (parameter part of the ODE system), and the second is for the non-dynamic parameters (e.g., noise parameters). This is computationally feasible for models with ≤20 dynamic parameters and often performs better than BFGS methods.
    * `:GaussNewton`: Approximate the Hessian via the Gauss-Newton method, which often performs better than the BFGS method. If we can reuse the sensitivities from the gradient in the optimizer (see `reuseS`), this method is best paired with `gradientMethod=:ForwardEquations`.
- `sparseJacobian::Bool=false`: When solving the ODE du/dt=f(u, p, t), whether implicit solvers use a sparse Jacobian. Sparse Jacobian often performs best for large models (≥100 states).
- `specializeLevel=SciMLBase.FullSpecialize`: Specialization level when building the ODE problem. It is not recommended to change this parameter (see https://docs.sciml.ai/SciMLBase/stable/interfaces/Problems/).
- `sensealg`: Sensitivity algorithm for gradient computations. The available options for each gradient method are:
    * `:ForwardDiff`: None (as ForwardDiff takes care of all computation steps).
    * `:ForwardEquations`: `:ForwardDiff` (uses ForwardDiff.jl and typicaly performs best) or `ForwardDiffSensitivity()` and `ForwardSensitivity()` from SciMLSensitivity.jl (https://github.com/SciML/SciMLSensitivity.jl).
    * `:Adjoint`: `InterpolatingAdjoint()` and `QuadratureAdjoint()` from SciMLSensitivity.jl.
    * `:Zygote`: All sensealg in SciMLSensitivity.jl.
- `sensealgSS=nothing`: Sensitivity algorithm for adjoint gradient computations for steady-state simulations. The available options are `SteadyStateAdjoint()`, `InterpolatingAdjoint()`, and `QuadratureAdjoint()` from SciMLSensitivity.jl. `SteadyStateAdjoint()` is the most efficient but requires a non-singular Jacobian, and in the case of a non-singular Jacobian, the code automatically switches to `InterpolatingAdjoint()`.
- `chunkSize=nothing`: Chunk-size for ForwardDiff.jl when computing the gradient and Hessian via forward-mode automatic differentiation. If nothing is provided, the default value is used. Tuning `chunkSize` is non-trivial, and we plan to add automatic functionality for this.
- `splitOverConditions::Bool=false`: For gradient and Hessian via ForwardDiff.jl, whether or not to split calls to ForwardDiff across experimental (simulation) conditions. This parameter should only be set to true if the model has many parameters specific to an experimental condition; otherwise, the overhead from the calls will increase run time. See the Beer example for a case where this is needed.
- `reuseS::Bool=false` : If set to `true`, reuse the sensitivities computed during gradient computations for the Gauss-Newton Hessian approximation. This option is only applicable when using `hessianMethod=:GaussNewton` and `gradientMethod=:ForwardEquations`. Note that it should only be used when the optimizer always computes the gradient before the Hessian.
- `verbose::Bool=true` : If set to `true`, print progress messages while setting up the PEtabODEProblem.
"""
function createPEtabODEProblem(petabModel::PEtabModel;
                               odeSolverOptions::Union{Nothing, ODESolverOptions}=nothing,
                               odeSolverGradientOptions::Union{Nothing, ODESolverOptions}=nothing,
                               ssSolverOptions::Union{Nothing, SteadyStateSolverOptions}=nothing,
                               ssSolverGradientOptions::Union{Nothing, SteadyStateSolverOptions}=nothing,
                               costMethod::Union{Nothing, Symbol}=:Standard,
                               gradientMethod::Union{Nothing, Symbol}=nothing,
                               hessianMethod::Union{Nothing, Symbol}=nothing,
                               sparseJacobian::Union{Nothing, Bool}=nothing,
                               specializeLevel=SciMLBase.FullSpecialize,
                               sensealg=nothing,
                               sensealgSS=nothing,
                               chunkSize::Union{Nothing, Int64}=nothing,
                               splitOverConditions::Bool=false,
                               numberOfprocesses::Signed=1,
                               reuseS::Bool=false,
                               verbose::Bool=true,
                               customParameterValues::Union{Nothing, Dict}=nothing)::PEtabODEProblem

    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && @printf(" Building PEtabODEProblem for %s\n", petabModel.modelName)

    # Sanity check user provided methods
    allowedCostMethods = [:Standard, :Zygote]
    allowedGradientMethods = [nothing, :ForwardDiff, :ForwardEquations, :Adjoint, :Zygote]
    allowedHessianMethods = [nothing, :ForwardDiff, :BlockForwardDiff, :GaussNewton]
    @assert costMethod ∈ allowedCostMethods "Allowed cost methods are " * string(allowedCostMethods) * " not " * string(costMethod)
    @assert gradientMethod ∈ allowedGradientMethods "Allowed gradient methods are " * string(allowedGradientMethods) * " not " * string(gradientMethod)
    @assert hessianMethod ∈ allowedHessianMethods "Allowed hessian methods are " * string(allowedHessianMethods) * " not " * string(hessianMethod)

    # Structs to bookep parameters, measurements, observations etc...
    experimentalConditions, measurementsData, parametersData, observablesData = readPEtabFiles(petabModel)
    parameterInfo = processParameters(parametersData, customParameterValues=customParameterValues)
    measurementInfo = processMeasurements(measurementsData, observablesData)
    θ_indices = computeIndicesθ(parameterInfo, measurementInfo, petabModel)
    priorInfo = processPriors(θ_indices, parametersData)

    # In case not specified by the user set ODE, gradient and Hessian options
    nODEs = length(states(petabModel.odeSystem))
    if nODEs ≤ 15 && length(θ_indices.θ_dynamicNames) ≤ 20
        modelSize = :Small
    elseif nODEs ≤ 50 && length(θ_indices.θ_dynamicNames) ≤ 69
        modelSize = :Medium
    else
        modelSize = :Large
    end
    _gradientMethod = setGradientMethod(gradientMethod, modelSize, reuseS)
    _hessianMethod = setHessianMethod(hessianMethod, modelSize)
    _sensealg = setSensealg(sensealg, Val(_gradientMethod))
    _odeSolverOptions = setODESolverOptions(odeSolverOptions, modelSize, _gradientMethod)
    _odeSolverGradientOptions = isnothing(odeSolverGradientOptions) ? deepcopy(_odeSolverOptions) : odeSolverGradientOptions
    _ssSolverOptions = setSteadyStateSolverOptions(ssSolverOptions, _odeSolverOptions)
    _ssSolverGradientOptions = isnothing(ssSolverGradientOptions) ? deepcopy(_ssSolverOptions) : ssSolverGradientOptions
    _sparseJacobian = !isnothing(sparseJacobian) ? sparseJacobian : (modelSize === :Large ? true : false)

    simulationInfo = processSimulationInfo(petabModel, measurementInfo, sensealg=_sensealg)

    println("_sensealg = ", _sensealg)

    # The time-span 5e3 is overwritten when performing forward simulations. As we solve an expanded system with the forward
    # equations, we need a seperate problem for it
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && @printf(" Building ODEProblem from ODESystem ...")
    timeTake = @elapsed begin
    # Set model parameter values to those in the PeTab parameter to ensure correct constant parameters
    setParamToFileValues!(petabModel.parameterMap, petabModel.stateMap, parameterInfo)
    __odeProblem = ODEProblem{true, specializeLevel}(petabModel.odeSystem, petabModel.stateMap, [0.0, 5e3], petabModel.parameterMap, jac=true, sparse=_sparseJacobian)
    _odeProblem = remake(__odeProblem, p = convert.(Float64, __odeProblem.p), u0 = convert.(Float64, __odeProblem.u0))
    end
    verbose == true && @printf(" done. Time = %.1e\n", timeTake)

    # Needed to properly initalise steady-state solver options with model Jacobian etc...
    _ssSolverOptions = _getSteadyStateSolverOptions(_ssSolverOptions, _odeProblem, _ssSolverOptions.abstol, _ssSolverOptions.reltol, _ssSolverOptions.maxiters)
    _ssSolverGradientOptions = _getSteadyStateSolverOptions(_ssSolverGradientOptions, _odeProblem, _ssSolverGradientOptions.abstol, _ssSolverGradientOptions.reltol, _ssSolverGradientOptions.maxiters)

    # If we are computing the cost, gradient and hessians accross several processes we need to send ODEProblem, and
    # PEtab structs to each process
    if numberOfprocesses > 1
        jobs, results = setUpProcesses(petabModel, odeSolverOptions, solverAbsTol, solverRelTol, odeSolverAdjoint, sensealgAdjoint,
                                       sensealgSS, solverAdjointAbsTol, solverAdjointRelTol, odeSolverForwardEquations,
                                       sensealgForwardEquations, parameterInfo, measurementInfo, simulationInfo, θ_indices,
                                       priorInfo, odeProblem, chunkSize)
    else
        jobs, results = nothing, nothing
    end

    petabODECache = createPEtabODEProblemCache(_gradientMethod, _hessianMethod, petabModel, _sensealg, measurementInfo, simulationInfo, θ_indices, chunkSize)
    petabODESolverCache = createPEtabODESolverCache(_gradientMethod, _hessianMethod, petabModel, simulationInfo, θ_indices, chunkSize)

    # The cost (likelihood) can either be computed in the standard way or the Zygote way. The second consumes more
    # memory as in-place mutations are not compatible with Zygote
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building cost function for method ", string(costMethod), " ...")
    bBuild = @elapsed computeCost = setUpCost(costMethod, _odeProblem, _odeSolverOptions, _ssSolverOptions, petabODECache, petabODESolverCache,
                                              petabModel, simulationInfo, θ_indices, measurementInfo, parameterInfo, priorInfo,
                                              sensealg, numberOfprocesses, jobs, results, false)

    computeChi2 = (θ; asArray=false) -> begin
        _ = computeCost(θ)
        if asArray == false
            return sum(measurementInfo.chi2Values)
        else
            return measurementInfo.chi2Values
        end
    end
    computeResiduals = (θ; asArray=false) -> begin
        _ = computeCost(θ)
        if asArray == false
            return sum(measurementInfo.residuals)
        else
            return measurementInfo.residuals
        end
    end
    computeSimulatedValues = (θ) -> begin
        _ = computeCost(θ)
        return measurementInfo.simulatedValues
    end
    verbose == true && @printf(" done. Time = %.1e\n", bBuild)

    # The gradient can either be computed via autodiff, forward sensitivity equations, adjoint sensitivity equations
    # and Zygote
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building gradient function for method ", string(_gradientMethod), " ...")
    _odeProblemGradient = gradientMethod === :ForwardEquations ? getODEProblemForwardEquations(_odeProblem, sensealg) : getODEProblemForwardEquations(_odeProblem, :NoSpecialProblem)
    bBuild = @elapsed computeGradient! = setUpGradient(_gradientMethod, _odeProblemGradient, _odeSolverGradientOptions,
        _ssSolverGradientOptions, petabODECache, petabODESolverCache, petabModel, simulationInfo, θ_indices,
        measurementInfo, parameterInfo, _sensealg, priorInfo, chunkSize=chunkSize, numberOfprocesses=numberOfprocesses,
        jobs=jobs, results=results, splitOverConditions=splitOverConditions, sensealgSS=sensealgSS)
    # Non in-place gradient
    computeGradient = (θ) -> begin
        gradient = zeros(Float64, length(θ))
        computeGradient!(gradient, θ)
        return gradient
    end
    verbose == true && @printf(" done. Time = %.1e\n", bBuild)

    # The Hessian can either be computed via automatic differentation, or approximated via a block approximation or the
    # Gauss Newton method
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building hessian function for method ", string(_hessianMethod), " ...")
    bBuild = @elapsed computeHessian! = setUpHessian(_hessianMethod, _odeProblem, _odeSolverOptions, _ssSolverOptions,
        petabODECache, petabODESolverCache, petabModel, simulationInfo, θ_indices, measurementInfo, parameterInfo,
        priorInfo, chunkSize, numberOfprocesses=numberOfprocesses, jobs=jobs, results=results,
        splitOverConditions=splitOverConditions, reuseS=reuseS)
    # Non-inplace Hessian
    computeHessian = (θ) -> begin
                                hessian = zeros(Float64, length(θ), length(θ))
                                computeHessian!(hessian, θ)
                                return hessian
                            end
    verbose == true && @printf(" done. Time = %.1e\n", bBuild)

    # Nominal parameter values + parameter bounds on parameter-scale (transformed)
    θ_estNames = θ_indices.θ_estNames
    lowerBounds = [parameterInfo.lowerBound[findfirst(x -> x == θ_estNames[i], parameterInfo.parameterId)] for i in eachindex(θ_estNames)]
    upperBounds = [parameterInfo.upperBound[findfirst(x -> x == θ_estNames[i], parameterInfo.parameterId)] for i in eachindex(θ_estNames)]
    θ_nominal = [parameterInfo.nominalValue[findfirst(x -> x == θ_estNames[i], parameterInfo.parameterId)] for i in eachindex(θ_estNames)]
    transformθ!(lowerBounds, θ_estNames, θ_indices, reverseTransform=true)
    transformθ!(upperBounds, θ_estNames, θ_indices, reverseTransform=true)
    θ_nominalT = transformθ(θ_nominal, θ_estNames, θ_indices, reverseTransform=true)

    petabProblem = PEtabODEProblem(computeCost,
                                   computeChi2,
                                   computeGradient!,
                                   computeGradient,
                                   computeHessian!,
                                   computeHessian,
                                   computeSimulatedValues,
                                   computeResiduals,
                                   costMethod,
                                   _gradientMethod,
                                   Symbol(_hessianMethod),
                                   Int64(length(θ_estNames)),
                                   θ_estNames,
                                   θ_nominal,
                                   θ_nominalT,
                                   lowerBounds,
                                   upperBounds,
                                   joinpath(petabModel.dirJulia, "Cube" * petabModel.modelName * ".csv"),
                                   petabModel,
                                   _odeSolverOptions,
                                   _odeSolverGradientOptions,
                                   _ssSolverOptions,
                                   _ssSolverGradientOptions,
                                   θ_indices,
                                   simulationInfo,
                                   splitOverConditions)
    return petabProblem
end


function setUpCost(whichMethod::Symbol,
                   odeProblem::ODEProblem,
                   odeSolverOptions::ODESolverOptions,
                   ssSolverOptions::SteadyStateSolverOptions,
                   petabODECache::PEtabODEProblemCache,
                   petabODESolverCache::PEtabODESolverCache,
                   petabModel::PEtabModel,
                   simulationInfo::SimulationInfo,
                   θ_indices::ParameterIndices,
                   measurementInfo::MeasurementsInfo,
                   parameterInfo::ParametersInfo,
                   priorInfo::PriorInfo,
                   sensealg,
                   numberOfprocesses,
                   jobs,
                   results,
                   computeResiduals)

    # Functions needed for mapping θ_est to the ODE problem, and then for solving said ODE-system
    if whichMethod == :Standard && numberOfprocesses == 1

        __computeCost = let θ_indices=θ_indices, odeSolverOptions=odeSolverOptions, priorInfo=priorInfo
                            (θ_est) -> computeCost(θ_est,
                                               odeProblem,
                                               odeSolverOptions,
                                               ssSolverOptions,
                                               petabModel,
                                               simulationInfo,
                                               θ_indices,
                                               measurementInfo,
                                               parameterInfo,
                                               priorInfo,
                                               petabODECache,
                                               petabODESolverCache,
                                               [:all],
                                               true,
                                               false,
                                               computeResiduals)
                        end
    end

    if whichMethod == :Zygote
        changeExperimentalCondition = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
        _changeODEProblemParameters = (pODEProblem, θ_est) -> changeODEProblemParameters(pODEProblem, θ_est, θ_indices, petabModel)
        solveODEExperimentalCondition = (odeProblem, conditionId, θ_dynamic, tMax) -> solveOdeModelAtExperimentalCondZygote(odeProblem, conditionId, θ_dynamic, tMax, changeExperimentalCondition, measurementInfo, simulationInfo, odeSolverOptions.solver, odeSolverOptions.abstol, odeSolverOptions.reltol, ssSolverOptions.abstol, ssSolverOptions.reltol, sensealg, petabModel.computeTStops)
        __computeCost = (θ_est) -> computeCostZygote(θ_est,
                                                     odeProblem,
                                                     petabModel,
                                                     simulationInfo,
                                                     θ_indices,
                                                     measurementInfo,
                                                     parameterInfo,
                                                     _changeODEProblemParameters,
                                                     solveODEExperimentalCondition,
                                                     priorInfo)
    end

    if false
        __computeCost = (θ_est) ->  begin
                                            costTot::Float64 = 0.0
                                            @inbounds for i in numberOfprocesses:-1:1
                                                @async put!(jobs[i], tuple(θ_est, :Cost))
                                            end
                                            @inbounds for i in numberOfprocesses:-1:1
                                                status::Symbol, cost::Float64 = take!(results[i])
                                                if status != :Done
                                                    println("Error : Could not send ODE problem to proces ", procs()[i])
                                                end
                                                costTot += cost
                                            end
                                            return costTot
                                        end

    end

    return __computeCost
end


function setUpGradient(whichMethod::Symbol,
                       odeProblem::ODEProblem,
                       odeSolverOptions::ODESolverOptions,
                       ssSolverOptions::SteadyStateSolverOptions,
                       petabODECache::PEtabODEProblemCache,
                       petabODESolverCache::PEtabODESolverCache,
                       petabModel::PEtabModel,
                       simulationInfo::SimulationInfo,
                       θ_indices::ParameterIndices,
                       measurementInfo::MeasurementsInfo,
                       parameterInfo::ParametersInfo,
                       sensealg,
                       priorInfo::PriorInfo;
                       chunkSize::Union{Nothing, Int64}=nothing,
                       sensealgSS=nothing,
                       numberOfprocesses::Int64=1,
                       jobs=nothing,
                       results=nothing,
                       splitOverConditions::Bool=false)

    θ_dynamic = petabODECache.θ_dynamic
    θ_sd = petabODECache.θ_sd
    θ_observable = petabODECache.θ_observable
    θ_nonDynamic = petabODECache.θ_nonDynamic

    if whichMethod == :ForwardDiff && numberOfprocesses == 1

        iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = getIndicesParametersNotInODESystem(θ_indices)
        computeCostNotODESystemθ = (x) -> computeCostNotSolveODE(x[iθ_sd], x[iθ_observable], x[iθ_nonDynamic],
                                                                 petabModel, simulationInfo, θ_indices, measurementInfo,
                                                                 parameterInfo, petabODECache, expIDSolve=[:all],
                                                                 computeGradientNotSolveAutoDiff=true)

        if splitOverConditions == false
            # Compute gradient for parameters which are a part of the ODE-system (dynamic parameters)
            computeCostDynamicθ = (x) -> computeCostSolveODE(x, θ_sd, θ_observable, θ_nonDynamic, odeProblem,
                odeSolverOptions, ssSolverOptions, petabModel, simulationInfo, θ_indices, measurementInfo,
                parameterInfo, petabODECache, petabODESolverCache, computeGradientDynamicθ=true, expIDSolve=[:all])

            _chunkSize = isnothing(chunkSize) ? ForwardDiff.Chunk(θ_dynamic) : ForwardDiff.Chunk(chunkSize)
            cfg = ForwardDiff.GradientConfig(computeCostDynamicθ, θ_dynamic, _chunkSize)

            _computeGradient! = (gradient, θ_est; isRemade=false) -> computeGradientAutoDiff!(gradient,
                                                                                              θ_est,
                                                                                              computeCostNotODESystemθ,
                                                                                              computeCostDynamicθ,
                                                                                              petabODECache,
                                                                                              cfg,
                                                                                              simulationInfo,
                                                                                              θ_indices,
                                                                                              priorInfo;
                                                                                              isRemade=isRemade)
        end

        if splitOverConditions == true

            computeCostDynamicθ = (x, _expIdSolve) -> computeCostSolveODE(x, θ_sd, θ_observable, θ_nonDynamic, odeProblem,
                odeSolverOptions, ssSolverOptions, petabModel, simulationInfo, θ_indices, measurementInfo, parameterInfo,
                petabODECache, petabODESolverCache, computeGradientDynamicθ=true, expIDSolve=_expIdSolve)

            _computeGradient! = (gradient, θ_est) -> computeGradientAutoDiffSplitOverConditions!(gradient,
                                                                                                 θ_est,
                                                                                                 computeCostNotODESystemθ,
                                                                                                 computeCostDynamicθ,
                                                                                                 petabODECache,
                                                                                                 simulationInfo,
                                                                                                 θ_indices,
                                                                                                 priorInfo)
        end
    end

    if whichMethod === :ForwardEquations && numberOfprocesses == 1

        _chunkSize = isnothing(chunkSize) ? ForwardDiff.Chunk(θ_dynamic) : ForwardDiff.Chunk(chunkSize)
        if sensealg === :ForwardDiff && splitOverConditions == false
            _solveODEAllExperimentalConditions! = (odeSolutionValues, θ) -> begin
                solveODEAllExperimentalConditions!(odeSolutionValues, θ, petabODESolverCache,
                    simulationInfo.odeSolutionsDerivatives, odeProblem, petabModel, simulationInfo, odeSolverOptions,
                    ssSolverOptions, θ_indices, petabODECache, onlySaveAtObservedTimes=true, expIDSolve=[:all],
                    computeForwardSensitivitesAD=true)
            end
            cfg = ForwardDiff.JacobianConfig(_solveODEAllExperimentalConditions!, petabODECache.odeSolutionValues,
                petabODECache.θ_dynamic, _chunkSize)
        end

        if sensealg === :ForwardDiff && splitOverConditions == true

            _solveODEAllExperimentalConditions! = (odeSolutionValues, θ, _expIdSolve) -> begin
                solveODEAllExperimentalConditions!(odeSolutionValues, θ, petabODESolverCache,
                    simulationInfo.odeSolutionsDerivatives, odeProblem, petabModel, simulationInfo, odeSolverOptions,
                    ssSolverOptions, θ_indices, petabODECache, onlySaveAtObservedTimes=true, expIDSolve=_expIdSolve,
                    computeForwardSensitivitesAD=true)
            end
            cfg = ForwardDiff.JacobianConfig(_solveODEAllExperimentalConditions!, petabODECache.odeSolutionValues,
                petabODECache.θ_dynamic, _chunkSize)
        end

        if sensealg != :ForwardDiff
            _solveODEAllExperimentalConditions! = (odeSolutions, odeProblem, θ_dynamic, _expIDSolve) -> begin
                solveODEAllExperimentalConditions!(odeSolutions, odeProblem, petabModel, θ_dynamic, petabODESolverCache,
                    simulationInfo, θ_indices, odeSolverOptions, ssSolverOptions, onlySaveAtObservedTimes=true,
                    expIDSolve=_expIDSolve, computeForwardSensitivites=true)
                end
            cfg = nothing
        end

        iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = getIndicesParametersNotInODESystem(θ_indices)
        computeCostNotODESystemθ = (x) -> computeCostNotSolveODE(x[iθ_sd], x[iθ_observable], x[iθ_nonDynamic],
            petabModel, simulationInfo, θ_indices, measurementInfo, parameterInfo, petabODECache, expIDSolve=[:all],
            computeGradientNotSolveForward=true)

        _computeGradient! = (gradient, θ_est; isRemade=false) -> computeGradientForwardEquations!(gradient,
                                                                                                 θ_est,
                                                                                                 computeCostNotODESystemθ,
                                                                                                 petabModel,
                                                                                                 odeProblem,
                                                                                                 sensealg,
                                                                                                 simulationInfo,
                                                                                                 θ_indices,
                                                                                                 measurementInfo,
                                                                                                 parameterInfo,
                                                                                                 _solveODEAllExperimentalConditions!,
                                                                                                 priorInfo,
                                                                                                 cfg,
                                                                                                 petabODECache,
                                                                                                 expIDSolve=[:all],
                                                                                                 splitOverConditions=splitOverConditions,
                                                                                                 isRemade=isRemade)
    end

    if whichMethod === :Zygote

        changeExperimentalCondition = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
        _changeODEProblemParameters = (pODEProblem, θ_est) -> changeODEProblemParameters(pODEProblem, θ_est, θ_indices, petabModel)
        solveODEExperimentalCondition = (odeProblem, conditionId, θ_dynamic, tMax) -> solveOdeModelAtExperimentalCondZygote(odeProblem, conditionId, θ_dynamic, tMax, changeExperimentalCondition, measurementInfo, simulationInfo, odeSolverOptions.solver, odeSolverOptions.abstol, odeSolverOptions.reltol, ssSolverOptions.abstol, ssSolverOptions.reltol, sensealg, petabModel.computeTStops)
        _computeGradient! = (gradient, θ_est) -> computeGradientZygote(gradient,
                                                                      θ_est,
                                                                      odeProblem,
                                                                      petabModel,
                                                                      simulationInfo,
                                                                      θ_indices,
                                                                      measurementInfo,
                                                                      parameterInfo,
                                                                      _changeODEProblemParameters,
                                                                      solveODEExperimentalCondition,
                                                                      priorInfo,
                                                                      petabODECache)
    end

    if false

        _computeGradient! = (gradient, θ_est) -> begin
                                                    gradient .= 0.0
                                                    @inbounds for i in numberOfprocesses:-1:1
                                                        @async put!(jobs[i], tuple(θ_est, whichMethod))
                                                    end
                                                    @inbounds for i in numberOfprocesses:-1:1
                                                        status::Symbol, gradientPart::Vector{Float64} = take!(results[i])
                                                        if status != :Done
                                                            println("Error : Could not compute gradient for ", procs()[i])
                                                        end
                                                        gradient .+= gradientPart
                                                    end
                                                end


    end

    return _computeGradient!
end


function setUpHessian(whichMethod::Symbol,
                      odeProblem::ODEProblem,
                      odeSolverOptions::ODESolverOptions,
                      ssSolverOptions::SteadyStateSolverOptions,
                      petabODECache::PEtabODEProblemCache,
                      petabODESolverCache::PEtabODESolverCache,
                      petabModel::PEtabModel,
                      simulationInfo::SimulationInfo,
                      θ_indices::ParameterIndices,
                      measurementInfo::MeasurementsInfo,
                      parameterInfo::ParametersInfo,
                      priorInfo::PriorInfo,
                      chunkSize::Union{Nothing, Int64};
                      reuseS::Bool=false,
                      numberOfprocesses::Int64=1,
                      jobs=nothing,
                      results=nothing,
                      splitOverConditions::Bool=false,
                      returnJacobian::Bool=false)

    θ_dynamic = petabODECache.θ_dynamic
    θ_sd = petabODECache.θ_sd
    θ_observable = petabODECache.θ_observable
    θ_nonDynamic = petabODECache.θ_nonDynamic

    if whichMethod === :ForwardDiff

        if splitOverConditions == false

            _evalHessian = (θ_est) -> computeCost(θ_est, odeProblem, odeSolverOptions, ssSolverOptions, petabModel,
                simulationInfo, θ_indices, measurementInfo, parameterInfo, priorInfo, petabODECache,
                petabODESolverCache, [:all], false, true, false)

            _chunkSize = isnothing(chunkSize) ? ForwardDiff.Chunk(zeros(length(θ_indices.θ_estNames))) : ForwardDiff.Chunk(chunkSize)
            cfg = ForwardDiff.HessianConfig(_evalHessian, zeros(length(θ_indices.θ_estNames)), _chunkSize)

            _computeHessian = (hessian, θ_est) -> computeHessian!(hessian,
                                                                  θ_est,
                                                                  _evalHessian,
                                                                  cfg,
                                                                  simulationInfo,
                                                                  θ_indices,
                                                                  priorInfo)
        end

        if splitOverConditions == true
            _evalHessian = (θ_est) -> computeCost(θ_est, odeProblem, odeSolverOptions, ssSolverOptions, petabModel,
                                                  simulationInfo, θ_indices, measurementInfo, parameterInfo, priorInfo,
                                                  petabODECache, petabODESolverCache, [:all], false, true, false)
            _computeHessian = (hessian, θ_est) -> computeHessianSplitOverConditions!(hessian,
                                                                                     θ_est,
                                                                                     _evalHessian,
                                                                                     simulationInfo,
                                                                                     θ_indices,
                                                                                     priorInfo)
        end

    end

    # Functions needed for mapping θ_est to the ODE problem, and then for solving said ODE-system
    if whichMethod === :BlockForwardDiff

        iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = getIndicesParametersNotInODESystem(θ_indices)
        computeCostNotODESystemθ = (x) -> computeCostNotSolveODE(x[iθ_sd], x[iθ_observable], x[iθ_nonDynamic],
                                                                 petabModel, simulationInfo, θ_indices, measurementInfo,
                                                                 parameterInfo, petabODECache, expIDSolve=[:all],
                                                                 computeGradientNotSolveAutoDiff=true)

        if splitOverConditions == false

            computeCostDynamicθ = (x) -> computeCostSolveODE(x, θ_sd, θ_observable, θ_nonDynamic, odeProblem,
                odeSolverOptions, ssSolverOptions, petabModel, simulationInfo, θ_indices, measurementInfo,
                parameterInfo, petabODECache, petabODESolverCache, computeGradientDynamicθ=true, expIDSolve=[:all])

            _chunkSize = isnothing(chunkSize) ? ForwardDiff.Chunk(θ_dynamic) : ForwardDiff.Chunk(chunkSize)
            cfg = ForwardDiff.HessianConfig(computeCostDynamicθ, θ_dynamic, ForwardDiff.Chunk(chunkSize))

            _computeHessian = (hessian, θ_est) -> computeHessianBlockApproximation!(hessian,
                                                                                        θ_est,
                                                                                        computeCostNotODESystemθ,
                                                                                        computeCostDynamicθ,
                                                                                        petabODECache,
                                                                                        cfg,
                                                                                        simulationInfo,
                                                                                        θ_indices,
                                                                                        priorInfo,
                                                                                        expIDSolve=[:all])
        end

        if splitOverConditions == true

            computeCostDynamicθ = (x, _expIDSolve) -> computeCostSolveODE(x, θ_sd, θ_observable, θ_nonDynamic, odeProblem, odeSolverOptions,
                ssSolverOptions, petabModel, simulationInfo, θ_indices, measurementInfo, parameterInfo, petabODECache,
                petabODESolverCache, computeGradientDynamicθ=true, expIDSolve=_expIDSolve)

            _computeHessian = (hessian, θ_est) -> computeHessianBlockApproximationSplitOverConditions!(hessian,
                                                                                                       θ_est,
                                                                                                       computeCostNotODESystemθ,
                                                                                                       computeCostDynamicθ,
                                                                                                       petabODECache,
                                                                                                       simulationInfo,
                                                                                                       θ_indices,
                                                                                                       priorInfo,
                                                                                                       expIDSolve=[:all])
        end
    end

    if whichMethod == :GaussNewton && numberOfprocesses == 1

        changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
        changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)

        if splitOverConditions == false
            _solveODEAllExperimentalConditions! = (odeSolutionValues, θ) -> solveODEAllExperimentalConditions!(odeSolutionValues, θ, petabODESolverCache, simulationInfo.odeSolutionsDerivatives, odeProblem, petabModel, simulationInfo, odeSolverOptions, ssSolverOptions, θ_indices, petabODECache, onlySaveAtObservedTimes=true, expIDSolve=[:all], computeForwardSensitivitesAD=true)
        else
            _solveODEAllExperimentalConditions! = (odeSolutionValues, θ, _expIdSolve) -> solveODEAllExperimentalConditions!(odeSolutionValues, θ, petabODESolverCache, simulationInfo.odeSolutionsDerivatives, odeProblem, petabModel, simulationInfo, odeSolverOptions, ssSolverOptions, θ_indices, petabODECache, onlySaveAtObservedTimes=true, expIDSolve=_expIdSolve, computeForwardSensitivitesAD=true)
        end

        _chunkSize = isnothing(chunkSize) ? ForwardDiff.Chunk(petabODECache.θ_dynamic) : ForwardDiff.Chunk(chunkSize)
        cfg = cfg = ForwardDiff.JacobianConfig(_solveODEAllExperimentalConditions!, petabODECache.odeSolutionValues, petabODECache.θ_dynamic, _chunkSize)

        iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = getIndicesParametersNotInODESystem(θ_indices)
        _computeResidualsNotSolveODE! = (residuals, θ_notOdeSystem) ->  begin
            θ_sd = @view θ_notOdeSystem[iθ_sd]
            θ_observable = @view θ_notOdeSystem[iθ_observable]
            θ_nonDynamic = @view θ_notOdeSystem[iθ_nonDynamic]
            computeResidualsNotSolveODE!(residuals, θ_sd, θ_observable, θ_nonDynamic, petabModel, simulationInfo,
                                         θ_indices, measurementInfo, parameterInfo, petabODECache;
                                         expIDSolve=[:all])
                                                                        end

        _θ_notOdeSystem = zeros(eltype(petabODECache.θ_dynamic), length(iθ_notOdeSystem))
        cfgNotSolveODE = ForwardDiff.JacobianConfig(_computeResidualsNotSolveODE!, petabODECache.residualsGN, _θ_notOdeSystem, ForwardDiff.Chunk(_θ_notOdeSystem))

        _computeHessian = (hessian, θ_est; isRemade=false) -> computeGaussNewtonHessianApproximation!(hessian,
                                                                                                      θ_est,
                                                                                                      odeProblem,
                                                                                                      _computeResidualsNotSolveODE!,
                                                                                                      petabModel,
                                                                                                      simulationInfo,
                                                                                                      θ_indices,
                                                                                                      measurementInfo,
                                                                                                      parameterInfo,
                                                                                                      _solveODEAllExperimentalConditions!,
                                                                                                      priorInfo,
                                                                                                      cfg,
                                                                                                      cfgNotSolveODE,
                                                                                                      petabODECache,
                                                                                                      expIDSolve=[:all],
                                                                                                      reuseS=reuseS,
                                                                                                      returnJacobian=returnJacobian,
                                                                                                      splitOverConditions=splitOverConditions,
                                                                                                      isRemade=isRemade)
    end

    if false

        _computeHessian = (hessian, θ_est) ->   begin
                                                    hessian .= 0.0
                                                    @inbounds for i in numberOfprocesses:-1:1
                                                        @async put!(jobs[i], tuple(θ_est, whichMethod))
                                                    end
                                                    @inbounds for i in numberOfprocesses:-1:1
                                                        status::Symbol, hessianPart::Matrix{Float64} = take!(results[i])
                                                        if status != :Done
                                                            println("Error : Could not send ODE problem to process ", procs()[i])
                                                        end
                                                        hessian .+= hessianPart
                                                    end
                                                end

    end

    return _computeHessian
end


function getODEProblemForwardEquations(odeProblem::ODEProblem,
                                       sensealgForwardEquations)::ODEProblem
    return odeProblem
end


function createPEtabODEProblemCache(gradientMethod::Symbol,
                                    hessianMethod::Union{Symbol, Nothing},
                                    petabModel::PEtabModel,
                                    sensealg,
                                    measurementInfo::MeasurementsInfo,
                                    simulationInfo::SimulationInfo,
                                    θ_indices::ParameterIndices,
                                    _chunkSize)::PEtabODEProblemCache

    θ_dynamic = zeros(Float64, length(θ_indices.iθ_dynamic))
    θ_observable = zeros(Float64, length(θ_indices.iθ_observable))
    θ_sd = zeros(Float64, length(θ_indices.iθ_sd))
    θ_nonDynamic = zeros(Float64, length(θ_indices.iθ_nonDynamic))

    levelCache = 0
    if hessianMethod ∈ [:ForwardDiff, :BlockForwardDiff, :GaussNewton]
        levelCache = 2
    elseif gradientMethod ∈ [:ForwardDiff, :ForwardEquations]
        levelCache = 1
    else
        levelCache = 0
    end

    if isnothing(_chunkSize)
        chunkSize = length(θ_indices.iθ_dynamic) > 10 ? 10 : length(θ_indices.iθ_dynamic)
    else
        chunkSize = _chunkSize
    end

    _θ_dynamicT = zeros(Float64, length(θ_indices.iθ_dynamic))
    _θ_observableT = zeros(Float64, length(θ_indices.iθ_observable))
    _θ_sdT = zeros(Float64, length(θ_indices.iθ_sd))
    _θ_nonDynamicT = zeros(Float64, length(θ_indices.iθ_nonDynamic))
    θ_dynamicT = DiffCache(_θ_dynamicT, chunkSize, levels=levelCache)
    θ_observableT = DiffCache(_θ_observableT, chunkSize, levels=levelCache)
    θ_sdT = DiffCache(_θ_sdT, chunkSize, levels=levelCache)
    θ_nonDynamicT = DiffCache(_θ_nonDynamicT, chunkSize, levels=levelCache)

    gradientDyanmicθ = zeros(Float64, length(θ_dynamic))
    gradientNotODESystemθ = zeros(Float64, length(θ_indices.iθ_notOdeSystem))

    # For forward sensitivity equations and adjoint sensitivity analysis we need to
    # compute partial derivatives symbolically. Here the helping vectors are pre-allocated
    if gradientMethod ∈ [:Adjoint, :ForwardEquations] || hessianMethod ∈ [:GaussNewton]
        nModelStates = length(states(petabModel.odeSystem))
        nModelParameters = length(parameters(petabModel.odeSystem))
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
    if (gradientMethod === :ForwardEquations && sensealg === :ForwardDiff) || hessianMethod === :GaussNewton
        nModelStates = length(states(petabModel.odeSystem))
        nTimePointsSaveAt = sum(length(simulationInfo.timeObserved[experimentalConditionId]) for experimentalConditionId in simulationInfo.experimentalConditionId)
        S = zeros(Float64, (nTimePointsSaveAt*nModelStates, length(θ_indices.θ_dynamicNames)))
        odeSolutionValues = zeros(Float64, nModelStates, nTimePointsSaveAt)
    else
        S = zeros(Float64, (0, 0))
        odeSolutionValues = zeros(Float64, (0, 0))
    end

    if hessianMethod === :GaussNewton
        jacobianGN = zeros(Float64, length(θ_indices.θ_estNames), length(measurementInfo.time))
        residualsGN = zeros(Float64, length(measurementInfo.time))
    else
        jacobianGN = zeros(Float64, (0, 0))
        residualsGN = zeros(Float64, 0)
    end

    if gradientMethod === :ForwardEquations || hessianMethod === :GaussNewton
        _gradient = zeros(Float64, length(θ_indices.iθ_dynamic))
    else
        _gradient = zeros(Float64, 0)
    end

    if gradientMethod === :Adjoint
        nModelStates = length(states(petabModel.odeSystem))
        nModelParameters = length(parameters(petabModel.odeSystem))
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


function createPEtabODESolverCache(gradientMethod::Symbol,
                                   hessianMethod::Union{Symbol, Nothing},
                                   petabModel::PEtabModel,
                                   simulationInfo::SimulationInfo,
                                   θ_indices::ParameterIndices,
                                   _chunkSize)::PEtabODESolverCache

    nModelStates = length(states(petabModel.odeSystem))
    nModelParameters = length(parameters(petabModel.odeSystem))

    levelCache = 0
    if hessianMethod ∈ [:ForwardDiff, :BlockForwardDiff, :GaussNewton]
        levelCache = 2
    elseif gradientMethod ∈ [:ForwardDiff, :ForwardEquations]
        levelCache = 1
    else
        levelCache = 0
    end

    if isnothing(_chunkSize)
        chunkSize = length(θ_indices.iθ_dynamic) > 10 ? 10 : length(θ_indices.iθ_dynamic)
    else
        chunkSize = _chunkSize
    end

    if simulationInfo.haspreEquilibrationConditionId == true
        conditionsToSimulateOver = unique(vcat(simulationInfo.preEquilibrationConditionId, simulationInfo.experimentalConditionId))
    else
        conditionsToSimulateOver = unique(simulationInfo.experimentalConditionId)
    end

    _pODEProblemCache = Tuple(DiffCache(zeros(Float64, nModelParameters), chunkSize, levels=levelCache) for i in eachindex(conditionsToSimulateOver))
    _u0Cache = Tuple(DiffCache(zeros(Float64, nModelStates), chunkSize, levels=levelCache) for i in eachindex(conditionsToSimulateOver))
    pODEProblemCache::Dict = Dict([(conditionsToSimulateOver[i], _pODEProblemCache[i]) for i in eachindex(_pODEProblemCache)])
    u0Cache::Dict = Dict([(conditionsToSimulateOver[i], _u0Cache[i]) for i in eachindex(_u0Cache)])

    return PEtabODESolverCache(pODEProblemCache, u0Cache)
end
