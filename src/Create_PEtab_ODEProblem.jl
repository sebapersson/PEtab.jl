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
                               sensealg::Union{Nothing, Symbol, SciMLBase.AbstractSensitivityAlgorithm}=nothing,
                               sensealgSS::Union{Nothing, SciMLSensitivity.AbstractAdjointSensitivityAlgorithm}=nothing,
                               chunkSize::Union{Nothing, Int64}=nothing,
                               splitOverConditions::Bool=false,
                               numberOfprocesses::Signed=1,
                               reuseS::Bool=false, 
                               verbose::Bool=true)::PEtabODEProblem

    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && @printf(" Building PEtabODEProblem for %s\n", petabModel.modelName) 

    if isnothing(odeSolverGradientOptions)
        odeSolverGradientOptions = deepcopy(odeSolverOptions)
    end

    # Make sure proper gradient and hessian methods are used 
    allowedCostMethods = [:Standard, :Zygote]
    allowedGradientMethods = [nothing, :ForwardDiff, :ForwardEquations, :Adjoint, :Zygote]
    allowedHessianMethods = [nothing, :ForwardDiff, :BlockForwardDiff, :GaussNewton]
    @assert costMethod ∈ allowedCostMethods "Allowed cost methods are " * string(allowedCostMethods) * " not " * string(costMethod)
    @assert gradientMethod ∈ allowedGradientMethods "Allowed gradient methods are " * string(allowedGradientMethods) * " not " * string(gradientMethod)
    @assert hessianMethod ∈ allowedHessianMethods "Allowed hessian methods are " * string(allowedHessianMethods) * " not " * string(hessianMethod)

    experimentalConditions, measurementsData, parametersData, observablesData = readPEtabFiles(petabModel)
    parameterInfo = processParameters(parametersData)
    measurementInfo = processMeasurements(measurementsData, observablesData)
    simulationInfo = processSimulationInfo(petabModel, measurementInfo, sensealg=sensealg)
    θ_indices = computeIndicesθ(parameterInfo, measurementInfo, petabModel)

    # Set up potential prior for the parameters to estimate
    priorInfo = processPriors(θ_indices, parametersData)

    # Set model parameter values to those in the PeTab parameter to ensure correct value for constant parameters
    setParamToFileValues!(petabModel.parameterMap, petabModel.stateMap, parameterInfo)

    # Sanity check user input 
    if isnothing(gradientMethod)
        @assert isnothing(sensealg) "In case gradient method is not set sensealg cannot be set either"
    elseif gradientMethod === :ForwardEquations
        @assert sensealg == :ForwardDiff || any(typeof(sensealg) .<: [ForwardSensitivity, ForwardDiffSensitivity]) "For gradient method :ForwardEquations allowed sensealg args are :ForwardDiff, ForwardSensitivity(), ForwardDiffSensitivity() not $sensealg"
    elseif gradientMethod === :Adjoint
        @assert any(typeof(sensealg) .<: [InterpolatingAdjoint, QuadratureAdjoint]) "For gradient method :Adjoint allowed sensealg args are InterpolatingAdjoint, QuadratureAdjoint not $sensealg"
    elseif gradientMethod === :Zygote
        @assert (typeof(sensealg) <: SciMLSensitivity.AbstractSensitivityAlgorithm) "For Zygote an abstract sensitivity algorithm from SciMLSensitivity must be used"
    end
    
    # In case not specified by the user set options for ODE solver, gradient, 
    nODEs = length(states(petabModel.odeSystem)) 
    isSmallModel = nODEs ≤ 15 && length(θ_indices.θ_dynamicNames) ≤ 20
    isMediumModel = !isSmallModel && nODEs ≤ 50 && length(θ_indices.θ_dynamicNames) ≤ 69
    isLargeModel = !isSmallModel && !isMediumModel
    # ODE-solver options
    if isnothing(odeSolverOptions)
        if isSmallModel
            odeSolverOptions = ODESolverOptions(Rodas5P())
        elseif isMediumModel
            odeSolverOptions = ODESolverOptions(QNDF())
        elseif isLargeModel
            @warn "For large models we strongly recomend to compare different ODE-solvers instead of using default options"
            if gradientMethod === :Adjoint || isnothing(gradientMethod)
                odeSolverOptions = ODESolverOptions(CVODE_BDF())
            else
                odeSolverOptions = ODESolverOptions(KenCarp4())
            end
        end
    end
    odeSolverGradientOptions = isnothing(odeSolverGradientOptions) ? deepcopy(odeSolverOptions) : odeSolverGradientOptions
    # Steady state solver options 
    if isnothing(ssSolverOptions)
        ssSolverOptions = SteadyStateSolverOptions(:Simulate, 
                                                    abstol=odeSolverOptions.abstol / 100, 
                                                    reltol=odeSolverOptions.reltol / 100)
    end
    if isnothing(ssSolverGradientOptions)
        ssSolverGradientOptions = ssSolverOptions
    end
    # Gradient and Hessian options 
    if isnothing(gradientMethod)
        if isSmallModel
            gradientMethod = :ForwardDiff
        elseif isMediumModel
            if reuseS == false
                gradientMethod = :ForwardDiff
            else
                gradientMethod = :ForwardEquations
                sensealg = :ForwardDiff
            end
        elseif isLargeModel
            gradientMethod = :Adjoint
        end 
    end
    if isnothing(hessianMethod)
        if isSmallModel
            hessianMethod = :ForwardDiff
        elseif isMediumModel
            hessianMethod = :GaussNewton
        elseif isLargeModel
            hessianMethod = nothing
        end 
    end
    # Sparsity of ODE system 
    if isnothing(sparseJacobian)
        sparseJacobian = isLargeModel ? true : false
    end
    # Sensealg
    if isnothing(sensealg)
        if gradientMethod === :Adjoint
            sensealg = InterpolatingAdjoint()
        elseif gradientMethod === :ForwardDiff || gradientMethod === :ForwardEquations
            sensealg = :ForwardDiff
        end
    end
    sensealgSS = isnothing(sensealgSS) ? InterpolatingAdjoint(autojacvec=ReverseDiffVJP()) : sensealgSS

    # Fast but numerically unstable method - warn the user 
    if simulationInfo.haspreEquilibrationConditionId == true && typeof(sensealgSS) <: SteadyStateAdjoint
        @warn "If you are using adjoint sensitivity analysis for a model with PreEq-criteria the most the most efficient sensealgSS is as provided SteadyStateAdjoint. However, SteadyStateAdjoint fails if the Jacobian is singular hence we recomend you check that the Jacobian is non-singular."
    end

    # The time-span 5e3 is overwritten when performing forward simulations. As we solve an expanded system with the forward
    # equations, we need a seperate problem for it 
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && @printf(" Building ODEProblem from ODESystem ...")
    bBuild = @elapsed begin
    _odeProblem = ODEProblem{true, specializeLevel}(petabModel.odeSystem, petabModel.stateMap, [0.0, 5e3], petabModel.parameterMap, jac=true, sparse=sparseJacobian)
    odeProblem = remake(_odeProblem, p = convert.(Float64, _odeProblem.p), u0 = convert.(Float64, _odeProblem.u0))
    end
    verbose == true && @printf(" done. Time = %.1e\n", bBuild)

    _ssSolverOptions = _getSteadyStateSolverOptions(ssSolverOptions, odeProblem, ssSolverOptions.abstol, ssSolverOptions.reltol, ssSolverOptions.maxiters)
    _ssSolverGradientOptions = _getSteadyStateSolverOptions(ssSolverGradientOptions, odeProblem, ssSolverGradientOptions.abstol, ssSolverGradientOptions.reltol, ssSolverGradientOptions.maxiters)

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

    petabODECache = createPEtabODEProblemCache(gradientMethod, hessianMethod, petabModel, sensealg, measurementInfo, simulationInfo, θ_indices, chunkSize)
    petabODESolverCache = createPEtabODESolverCache(gradientMethod, hessianMethod, petabModel, simulationInfo, θ_indices, chunkSize)
    
    # The cost (likelihood) can either be computed in the standard way or the Zygote way. The second consumes more
    # memory as in-place mutations are not compatible with Zygote
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building cost function for method ", string(costMethod), " ...")
    bBuild = @elapsed computeCost = setUpCost(costMethod, odeProblem, odeSolverOptions, _ssSolverOptions, petabODECache, petabODESolverCache, 
                            petabModel, simulationInfo, θ_indices, measurementInfo, parameterInfo, priorInfo,
                            numberOfprocesses=numberOfprocesses, jobs=jobs, results=results)
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
    computeSimulatedValues = (θ; asArray=false) -> begin
        _ = computeCost(θ)
        return measurementInfo.simulatedValues
    end

    verbose == true && @printf(" done. Time = %.1e\n", bBuild)

    # The gradient can either be computed via autodiff, forward sensitivity equations, adjoint sensitivity equations
    # and Zygote
    if gradientMethod === :ForwardEquations
        @assert (typeof(sensealg) <: SciMLSensitivity.AbstractForwardSensitivityAlgorithm || typeof(sensealg) <: Symbol) "For forward equations allowed sensealg are ForwardDiffSensitivity(), ForwardSensitivity(), or :ForwardDiff"
    elseif gradientMethod === :Adjoint
        @assert (typeof(sensealg) <: SciMLSensitivity.AbstractAdjointSensitivityAlgorithm) "For adjoint sensitivity analysis allowed sensealg are InterpolatingAdjoint() or QuadratureAdjoint()"
    elseif gradientMethod === :Zygote
        @assert (typeof(sensealg) <: SciMLBase.AbstractSensitivityAlgorithm) "For Zygote an abstract sensitivity algorithm from SciMLSensitivity must be used"
    end
    odeProblemGradient = gradientMethod === :ForwardEquations ? getODEProblemForwardEquations(odeProblem, sensealg) : getODEProblemForwardEquations(odeProblem, :NoSpecialProblem)
    
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building gradient function for method ", string(gradientMethod), " ...")
    bBuild = @elapsed computeGradient! = setUpGradient(gradientMethod, odeProblemGradient, odeSolverGradientOptions, _ssSolverGradientOptions, petabODECache, 
                                     petabODESolverCache, petabModel, simulationInfo, θ_indices, measurementInfo, parameterInfo, priorInfo,
                                     chunkSize=chunkSize, numberOfprocesses=numberOfprocesses, jobs=jobs, results=results,
                                     splitOverConditions=splitOverConditions, sensealg=sensealg, sensealgSS=sensealgSS)
    computeGradient = (θ) -> begin
        gradient = zeros(Float64, length(θ))
        computeGradient!(gradient, θ)
        return gradient
    end                                            
    verbose == true && @printf(" done. Time = %.1e\n", bBuild)

    # The Hessian can either be computed via automatic differentation, or approximated via a block approximation or the
    # Gauss Newton method
    verbose == true && printstyled("[ Info:", color=123, bold=true)
    verbose == true && print(" Building hessian function for method ", string(hessianMethod), " ...")
    if !isnothing(hessianMethod)
        bBuild = @elapsed computeHessian! = setUpHessian(hessianMethod, odeProblem, odeSolverOptions, _ssSolverOptions, petabODECache, petabODESolverCache,
                                    petabModel, simulationInfo, θ_indices, measurementInfo, parameterInfo, priorInfo, chunkSize,
                                    numberOfprocesses=numberOfprocesses, jobs=jobs, results=results, splitOverConditions=splitOverConditions, 
                                    reuseS=reuseS)
        computeHessian = (θ) -> begin
            hessian = zeros(Float64, length(θ), length(θ))
            computeHessian!(hessian, θ)
            return hessian
        end                                   
    else
        computeHessian! = nothing
        computeHessian = nothing
    end
    verbose == true && @printf(" done. Time = %.1e\n", bBuild)                                   
    
    # Extract nominal parameter vector and parameter bounds. If needed transform parameters
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
                                   gradientMethod, 
                                   Symbol(hessianMethod), 
                                   Int64(length(θ_estNames)),
                                   θ_estNames,
                                   θ_nominal,
                                   θ_nominalT,
                                   lowerBounds,
                                   upperBounds,
                                   joinpath(petabModel.dirJulia, "Cube" * petabModel.modelName * ".csv"),
                                   petabModel, 
                                   odeSolverOptions, 
                                   odeSolverGradientOptions, 
                                   ssSolverOptions, 
                                   ssSolverGradientOptions)
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
                   priorInfo::PriorInfo;
                   sensealg=ForwardDiffSensitivity(),
                   numberOfprocesses::Int64=1,
                   jobs=nothing,
                   results=nothing, 
                   computeResiduals::Bool=false,)

    # Functions needed for mapping θ_est to the ODE problem, and then for solving said ODE-system
    if whichMethod == :Standard && numberOfprocesses == 1

        __computeCost = (θ_est) -> computeCost(θ_est,
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
                                               expIDSolve=[:all], 
                                               computeCost=true,
                                               computeResiduals=computeResiduals)

    elseif whichMethod == :Zygote
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

    else
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
                       priorInfo::PriorInfo;
                       chunkSize::Union{Nothing, Int64}=nothing,
                       sensealg=nothing,
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
            computeCostDynamicθ = (x) -> computeCostSolveODE(x, θ_sd, θ_observable, θ_nonDynamic, odeProblem, odeSolverOptions,
                                                             ssSolverOptions, petabModel, simulationInfo, θ_indices, measurementInfo, 
                                                             parameterInfo, petabODECache, petabODESolverCache,
                                                             computeGradientDynamicθ=true, expIDSolve=[:all])
            if !isnothing(chunkSize)
                cfg = ForwardDiff.GradientConfig(computeCostDynamicθ, θ_dynamic, ForwardDiff.Chunk(chunkSize))
            else
                cfg = ForwardDiff.GradientConfig(computeCostDynamicθ, θ_dynamic, ForwardDiff.Chunk(θ_dynamic))
            end                                                         

            _computeGradient! = (gradient, θ_est) -> computeGradientAutoDiff!(gradient,
                                                                              θ_est,
                                                                              computeCostNotODESystemθ,
                                                                              computeCostDynamicθ,
                                                                              petabODECache,
                                                                              cfg,
                                                                              simulationInfo,
                                                                              θ_indices,
                                                                              priorInfo)
        else

            computeCostDynamicθ = (x, _expIdSolve) -> computeCostSolveODE(x, θ_sd, θ_observable, θ_nonDynamic, odeProblem, odeSolverOptions, 
                                                                          ssSolverOptions, petabModel, simulationInfo, θ_indices, measurementInfo, parameterInfo,
                                                                          petabODECache, petabODESolverCache,
                                                                          computeGradientDynamicθ=true, expIDSolve=_expIdSolve)

            _computeGradient! = (gradient, θ_est) -> computeGradientAutoDiffSplitOverConditions!(gradient,
                                                                                                 θ_est,
                                                                                                 computeCostNotODESystemθ,
                                                                                                 computeCostDynamicθ,
                                                                                                 petabODECache,
                                                                                                 simulationInfo,
                                                                                                 θ_indices,
                                                                                                 priorInfo)   
        end

    elseif whichMethod === :ForwardEquations && numberOfprocesses == 1
        if sensealg === :ForwardDiff

            if splitOverConditions == false            
                _solveODEAllExperimentalConditions! = (odeSolutionValues, θ) -> solveODEAllExperimentalConditions!(odeSolutionValues, θ, petabODESolverCache, simulationInfo.odeSolutionsDerivatives, odeProblem, petabModel, simulationInfo, odeSolverOptions, ssSolverOptions, θ_indices, onlySaveAtObservedTimes=true, expIDSolve=[:all])
            else
                _solveODEAllExperimentalConditions! = (odeSolutionValues, θ, _expIdSolve) -> solveODEAllExperimentalConditions!(odeSolutionValues, θ, petabODESolverCache, simulationInfo.odeSolutionsDerivatives, odeProblem, petabModel, simulationInfo, odeSolverOptions, ssSolverOptions, θ_indices, onlySaveAtObservedTimes=true, expIDSolve=_expIdSolve)
            end
            
            if !isnothing(chunkSize)
                cfg = ForwardDiff.JacobianConfig(_solveODEAllExperimentalConditions!, petabODECache.odeSolutionValues, petabODECache.θ_dynamic, ForwardDiff.Chunk(chunkSize))
            else
                cfg = ForwardDiff.JacobianConfig(_solveODEAllExperimentalConditions!, petabODECache.odeSolutionValues, petabODECache.θ_dynamic, ForwardDiff.Chunk(θ_dynamic))
            end

        else
            _solveODEAllExperimentalConditions! = (odeSolutions, odeProblem, θ_dynamic, _expIDSolve) -> solveODEAllExperimentalConditions!(odeSolutions, odeProblem, petabModel, θ_dynamic, petabODESolverCache, simulationInfo, θ_indices, odeSolverOptions, ssSolverOptions, onlySaveAtObservedTimes=true, expIDSolve=_expIDSolve, computeForwardSensitivites=true)
            cfg = nothing
        end

        iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = getIndicesParametersNotInODESystem(θ_indices)
        computeCostNotODESystemθ = (x) -> computeCostNotSolveODE(x[iθ_sd], x[iθ_observable], x[iθ_nonDynamic],
                                                                 petabModel, simulationInfo, θ_indices, measurementInfo,
                                                                 parameterInfo, petabODECache, expIDSolve=[:all],
                                                                 computeGradientNotSolveForward=true)

        _computeGradient! = (gradient, θ_est) -> computeGradientForwardEquations!(gradient,
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
                                                                                  splitOverConditions=splitOverConditions)

    elseif whichMethod === :Adjoint && numberOfprocesses == 1

        iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = getIndicesParametersNotInODESystem(θ_indices)
        computeCostNotODESystemθ = (x) -> computeCostNotSolveODE(x[iθ_sd], x[iθ_observable], x[iθ_nonDynamic],
                                                                 petabModel, simulationInfo, θ_indices, measurementInfo,
                                                                 parameterInfo, petabODECache, expIDSolve=[:all],
                                                                 computeGradientNotSolveAdjoint=true)

        _computeGradient! = (gradient, θ_est) -> computeGradientAdjointEquations!(gradient,
                                                                                 θ_est,
                                                                                 odeSolverOptions,
                                                                                 ssSolverOptions,
                                                                                 computeCostNotODESystemθ,
                                                                                 sensealg,
                                                                                 sensealgSS,
                                                                                 odeProblem,
                                                                                 petabModel,
                                                                                 simulationInfo,
                                                                                 θ_indices,
                                                                                 measurementInfo,
                                                                                 parameterInfo,
                                                                                 priorInfo,
                                                                                 petabODECache,
                                                                                 petabODESolverCache,
                                                                                 expIDSolve=[:all])

    elseif whichMethod === :Zygote

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

    else

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

    # Functions needed for mapping θ_est to the ODE problem, and then for solving said ODE-system
    if (whichMethod === :ForwardDiff || whichMethod === :BlockForwardDiff) && numberOfprocesses == 1

        if whichMethod === :ForwardDiff

            if splitOverConditions == false
                _evalHessian = (θ_est) -> computeCost(θ_est, odeProblem, odeSolverOptions, ssSolverOptions, petabModel, simulationInfo, θ_indices,
                                                      measurementInfo, parameterInfo, priorInfo, petabODECache, petabODESolverCache,
                                                      computeHessian=true, expIDSolve=[:all])

                if !isnothing(chunkSize)
                    cfg = ForwardDiff.HessianConfig(_evalHessian, θ_est, ForwardDiff.Chunk(chunkSize))
                else
                    _θ_est = zeros(Float64, length(θ_indices.θ_estNames))
                    cfg = ForwardDiff.HessianConfig(_evalHessian, _θ_est, ForwardDiff.Chunk(_θ_est))
                end
                _computeHessian = (hessian, θ_est) -> computeHessian!(hessian,
                                                                      θ_est,
                                                                      _evalHessian,
                                                                      cfg,
                                                                      simulationInfo,
                                                                      θ_indices, 
                                                                      priorInfo)

            else
                _evalHessian = (θ_est, _expIDSolve) -> computeCost(θ_est, odeProblem, odeSolverOptions, ssSolverOptions, petabModel, simulationInfo, θ_indices,
                                                                   measurementInfo, parameterInfo, priorInfo, petabODECache, petabODESolverCache,
                                                                   computeHessian=true, expIDSolve=_expIDSolve)
                _computeHessian = (hessian, θ_est) -> computeHessianSplitOverConditions!(hessian,
                                                                                         θ_est,
                                                                                         _evalHessian,  
                                                                                         simulationInfo,
                                                                                         θ_indices, 
                                                                                         priorInfo)
            end

        elseif whichMethod === :BlockForwardDiff
                      
            iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = getIndicesParametersNotInODESystem(θ_indices)
            computeCostNotODESystemθ = (x) -> computeCostNotSolveODE(x[iθ_sd], x[iθ_observable], x[iθ_nonDynamic],
                                                                    petabModel, simulationInfo, θ_indices, measurementInfo,
                                                                    parameterInfo, petabODECache, expIDSolve=[:all],
                                                                    computeGradientNotSolveAutoDiff=true)

            if splitOverConditions == false                                                                    
                # Compute gradient for parameters which are a part of the ODE-system (dynamic parameters)
                computeCostDynamicθ = (x) -> computeCostSolveODE(x, θ_sd, θ_observable, θ_nonDynamic, odeProblem, odeSolverOptions, 
                                                                 ssSolverOptions, petabModel, simulationInfo, θ_indices, measurementInfo, parameterInfo,
                                                                 petabODECache, petabODESolverCache,
                                                                 computeGradientDynamicθ=true, expIDSolve=[:all])
                if !isnothing(chunkSize)
                    cfg = ForwardDiff.HessianConfig(computeCostDynamicθ, θ_dynamic, ForwardDiff.Chunk(chunkSize))
                else
                    cfg = ForwardDiff.HessianConfig(computeCostDynamicθ, θ_dynamic, ForwardDiff.Chunk(θ_dynamic))
                end             

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
            else
                computeCostDynamicθ = (x, _expIDSolve) -> computeCostSolveODE(x, θ_sd, θ_observable, θ_nonDynamic, odeProblem, odeSolverOptions,
                                                                              ssSolverOptions, petabModel, simulationInfo, θ_indices, measurementInfo, 
                                                                              parameterInfo, petabODECache, petabODESolverCache,
                                                                              computeGradientDynamicθ=true, expIDSolve=_expIDSolve)

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

    elseif whichMethod == :GaussNewton && numberOfprocesses == 1
        
        changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
        changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)

        if splitOverConditions == false            
            _solveODEAllExperimentalConditions! = (odeSolutionValues, θ) -> solveODEAllExperimentalConditions!(odeSolutionValues, θ, petabODESolverCache, simulationInfo.odeSolutionsDerivatives, odeProblem, petabModel, simulationInfo, odeSolverOptions, ssSolverOptions, θ_indices, onlySaveAtObservedTimes=true, expIDSolve=[:all])
        else
            _solveODEAllExperimentalConditions! = (odeSolutionValues, θ, _expIdSolve) -> solveODEAllExperimentalConditions!(odeSolutionValues, θ, petabODESolverCache, simulationInfo.odeSolutionsDerivatives, odeProblem, petabModel, simulationInfo, odeSolverOptions, ssSolverOptions, θ_indices, onlySaveAtObservedTimes=true, expIDSolve=_expIdSolve)
        end
        
        if !isnothing(chunkSize)
            cfg = ForwardDiff.JacobianConfig(_solveODEAllExperimentalConditions!, petabODECache.odeSolutionValues, petabODECache.θ_dynamic, ForwardDiff.Chunk(chunkSize))
        else
            cfg = ForwardDiff.JacobianConfig(_solveODEAllExperimentalConditions!, petabODECache.odeSolutionValues, petabODECache.θ_dynamic, ForwardDiff.Chunk(petabODECache.θ_dynamic))
        end

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

        _computeHessian = (hessian, θ_est) -> computeGaussNewtonHessianApproximation!(hessian,
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
                                                                                      splitOverConditions=splitOverConditions)

    else

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
                                       sensealgForwardEquations::SciMLSensitivity.AbstractForwardSensitivityAlgorithm)::ODEProblem
    return ODEForwardSensitivityProblem(odeProblem.f, odeProblem.u0, odeProblem.tspan, odeProblem.p, sensealg=sensealgForwardEquations)
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
                                         odeSolutionValues)

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

    pODEProblemCache = NamedTuple{Tuple(name for name in conditionsToSimulateOver)}(Tuple(DiffCache(zeros(Float64, nModelParameters), chunkSize, levels=levelCache) for i in eachindex(conditionsToSimulateOver)))
    u0Cache = NamedTuple{Tuple(name for name in conditionsToSimulateOver)}(Tuple(DiffCache(zeros(Float64, nModelStates), chunkSize, levels=levelCache) for i in eachindex(conditionsToSimulateOver)))

    return PEtabODESolverCache(pODEProblemCache, u0Cache)

end


"""
    ODESolverOptions(solver, <keyword arguments>)

ODE-solver options (solver, tolerances, etc...) to use when computing gradient/cost for a PEtabODEProblem.

More information about the available options and solvers can be found in the documentation for DifferentialEquations.jl (https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/). Recommended settings for which solver and options to use for different problems can be found below and in the documentation.

# Arguments
- `solver`: Any of the ODE solvers in DifferentialEquations.jl. For small (≤20 states) mildly stiff models, composite solvers such as `AutoVern7(Rodas5P())` perform well. For stiff small models, `Rodas5P()` performs well. For medium-sized models (≤75 states), `QNDF()`, `FBDF()`, and `CVODE_BDF()` perform well. `CVODE_BDF()` is not compatible with automatic differentiation and thus cannot be used if the gradient is computed via automatic differentiation or if the Gauss-Newton Hessian approximation is used. If the gradient is computed via adjoint sensitivity analysis, `CVODE_BDF()` is often the best choice as it is typically more reliable than `QNDF()` and `FBDF()` (fails less often).
- `abstol=1e-8`: Absolute tolerance when solving the ODE system. Not recommended to increase above 1e-6 for gradients. 
- `reltol=1e-8`: Relative tolerance when solving the ODE system. Not recommended to increase above 1e-6 for gradients. 
- `force_dtmin=false`: Whether or not to force `dtmin` when solving the ODE system.
- `dtmin=nothing`: Minimal acceptable step-size when solving the ODE system.
- `maxiters=10000`: Maximum number of iterations when solving the ODE system. Increasing above the default value can cause the optimization to take substantial time.
"""
function ODESolverOptions(solver::T1; 
                          abstol::Float64=1e-8, 
                          reltol::Float64=1e-8, 
                          force_dtmin::Bool=false, 
                          dtmin::Union{Float64, Nothing}=nothing, 
                          maxiters::Int64=Int64(1e4)) where T1 <: SciMLAlgorithm 

    return ODESolverOptions(solver, abstol, reltol, force_dtmin, dtmin, maxiters)
end


# For better printing of the PEtab ODEProblem 
import Base.show
function show(io::IO, a::ODESolverOptions)
    # Extract ODE solver as a readable string (without everything between)
    solverStrWrite, optionsStr = getStringSolverOptions(a)
    printstyled("ODESolverOptions", color=116)
    print(" with ODE solver ")
    printstyled(solverStrWrite, color=116)
    @printf(". Options %s", optionsStr)
end
function show(io::IO, a::PEtabODEProblem)

    modelName = a.petabModel.modelName
    numberOfODEStates = length(a.petabModel.stateNames)
    numberOfParametersToEstimate = length(a.θ_estNames)
    θ_indices = a.computeCost.θ_indices
    numberOfDynamicParameters = length(θ_indices.iθ_dynamic)

    solverStrWrite, optionsStr = getStringSolverOptions(a.odeSolverOptions)
    solverGradStrWrite, optionsGradStr = getStringSolverOptions(a.odeSolverGradientOptions)

    gradientMethod = string(a.gradientMethod)
    hessianMethod = string(a.hessianMethod)
    
    printstyled("PEtabODEProblem", color=116)
    print(" for ")
    printstyled(modelName, color=116)
    @printf(". ODE-states: %d. Parameters to estimate: %d where %d are dynamic.\n---------- Problem settings ----------\nGradient method : ",
            numberOfODEStates, numberOfParametersToEstimate, numberOfDynamicParameters)
    printstyled(gradientMethod, color=116)
    if !isnothing(hessianMethod)
        print("\nHessian method : ")
        printstyled(hessianMethod, color=116)
    end
    print("\n--------- ODE-solver settings --------")
    printstyled("\nCost ")
    printstyled(solverStrWrite, color=116)
    @printf(". Options %s", optionsStr)
    printstyled("\nGradient ")
    printstyled(solverGradStrWrite, color=116)
    @printf(". Options %s", optionsGradStr)

    if a.computeCost.simulationInfo.haspreEquilibrationConditionId == true
        print("\n--------- SS solver settings ---------")
        # Print cost steady state solver
        print("\nCost ")
        printstyled(string(a.ssSolverOptions.method), color=116)
        if a.ssSolverOptions.method === :Simulate && a.ssSolverOptions.howCheckSimulationReachedSteadyState === :wrms
            @printf(". Option wrms with (abstol, reltol) = (%.1e, %.1e)", a.ssSolverOptions.abstol, a.ssSolverOptions.reltol)
        elseif a.ssSolverOptions.method === :Simulate && a.ssSolverOptions.howCheckSimulationReachedSteadyState === :Newton
            @printf(". Option small Newton-step with (abstol, reltol) = (%.1e, %.1e)", a.ssSolverOptions.abstol, a.ssSolverOptions.reltol)
        elseif a.ssSolverOptions.method === :Rootfinding
            algStr = string(a.ssSolverOptions.rootfindingAlgorithm)
            iEnd = findfirst(x -> x == '{', algStr)
            algStr = algStr[1:iEnd-1] * "()"
            @printf(". Algorithm %s with (abstol, reltol, maxiters) = (%.1e, %.1e, %.1e)", algStr, a.ssSolverOptions.abstol, a.ssSolverOptions.reltol, a.ssSolverOptions.maxiters)
        end

        # Print gradient steady state solver
        print("\nGradient ")
        printstyled(string(a.ssSolverGradientOptions.method), color=116)
        if a.ssSolverGradientOptions.method === :Simulate && a.ssSolverGradientOptions.howCheckSimulationReachedSteadyState === :wrms
            @printf(". Options wrms with (abstol, reltol) = (%.1e, %.1e)", a.ssSolverGradientOptions.abstol, a.ssSolverGradientOptions.reltol)
        elseif a.ssSolverGradientOptions.method === :Simulate && a.ssSolverGradientOptions.howCheckSimulationReachedSteadyState === :Newton
            @printf(". Option small Newton-step with (abstol, reltol) = (%.1e, %.1e)", a.ssSolverGradientOptions.abstol, a.ssSolverGradientOptions.reltol)
        elseif a.ssSolverGradientOptions.method === :Rootfinding
            algStr = string(a.ssSolverGradientOptions.rootfindingAlgorithm)
            iEnd = findfirst(x -> x == '{', algStr)
            algStr = algStr[1:iEnd-1] * "()"
            @printf(". Algorithm %s with (abstol, reltol, maxiters) = (%.1e, %.1e, %.1e)", algStr, a.ssSolverGradientOptions.abstol, a.ssSolverGradientOptions.reltol, a.ssSolverGradientOptions.maxiters)
        end
    end
end


function getStringSolverOptions(a::ODESolverOptions)
    solverStr = string(a.solver)
    iEnd = findfirst(x -> x == '{', solverStr)
    solverStrWrite = solverStr[1:iEnd-1] * "()"
    optionsStr = @sprintf("(abstol, reltol, maxiters) = (%.1e, %.1e, %.1e)", a.abstol, a.reltol, a.maxiters)
    return solverStrWrite, optionsStr
end