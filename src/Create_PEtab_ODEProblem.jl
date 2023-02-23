function setUpPEtabODEProblem(petabModel::PEtabModel,
                              odeSolver::SciMLAlgorithm;
                              costMethod::Symbol=:Standard,
                              gradientMethod::Symbol=:ForwardDiff,
                              hessianMethod::Symbol=:ForwardDiff,
                              solverAbsTol::Float64=1e-8,
                              solverRelTol::Float64=1e-8,
                              solverSSRelTol::Float64=1e-6,
                              solverSSAbsTol::Float64=1e-8,
                              sparseJacobian::Bool=false,
                              specializeLevel=SciMLBase.FullSpecialize,
                              sensealgZygote=ForwardDiffSensitivity(),
                              odeSolverForwardEquations::SciMLAlgorithm=Rodas5(autodiff=false),
                              sensealgForwardEquations::Union{Symbol, SciMLSensitivity.AbstractForwardSensitivityAlgorithm}=ForwardSensitivity(),
                              odeSolverAdjoint::SciMLAlgorithm=KenCarp4(),
                              solverAdjointAbsTol::Float64=1e-8,
                              solverAdjointRelTol::Float64=1e-8,
                              sensealgAdjoint::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false)),
                              sensealgAdjointSS::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm=SteadyStateAdjoint(),
                              chunkSize::Union{Nothing, Int64}=nothing,
                              terminateSSMethod::Symbol=:Norm,
                              splitOverConditions::Bool=false,
                              numberOfprocesses::Signed=1,
                              reuseS::Bool=false)::PEtabODEProblem

    if !(typeof(sensealgAdjointSS) <: SteadyStateAdjoint)
        println("If you are using adjoint sensitivity analysis for a model with PreEq-criteria the most the most efficient adjSensealgSS is usually SteadyStateAdjoint. The algorithm you have provided, ", sensealgAdjointSS, "might not work (as there are some bugs here). In case it does not work, and SteadyStateAdjoint fails (because a dependancy on time or a singular Jacobian) a good choice might be QuadratureAdjoint(autodiff=false, autojacvec=false)")
    end

    # Make sure proper gradient and hessian methods are used 
    allowedCostMethods = [:Standard, :Zygote]
    allowedGradientMethods = [:ForwardDiff, :ForwardEquations, :Adjoint, :Zygote]
    allowedHessianMethods = [:ForwardDiff, :BlockForwardDiff, :GaussNewton]
    @assert costMethod ∈ allowedCostMethods "Allowed cost methods are " * string(allowedCostMethods) * " not " * string(costMethod)
    @assert gradientMethod ∈ allowedGradientMethods "Allowed gradient methods are " * string(allowedGradientMethods) * " not " * string(gradientMethod)
    @assert hessianMethod ∈ allowedHessianMethods "Allowed hessian methods are " * string(allowedHessianMethods) * " not " * string(hessianMethod)

    experimentalConditions, measurementsData, parametersData, observablesData = readPEtabFiles(petabModel)
    parameterInfo = processParameters(parametersData)
    measurementInfo = processMeasurements(measurementsData, observablesData)
    simulationInfo = processSimulationInfo(petabModel, measurementInfo, parameterInfo, sensealg=sensealgAdjoint, absTolSS=solverSSAbsTol, relTolSS=solverSSRelTol, terminateSSMethod=terminateSSMethod, sensealgForwardEquations=sensealgForwardEquations)
    θ_indices = computeIndicesθ(parameterInfo, measurementInfo, petabModel.odeSystem, experimentalConditions)

    # Set up potential prior for the parameters to estimate
    priorInfo = processPriors(θ_indices, parametersData)

    # Set model parameter values to those in the PeTab parameter to ensure correct value for constant parameters
    setParamToFileValues!(petabModel.parameterMap, petabModel.stateMap, parameterInfo)

    # The time-span 5e3 is overwritten when performing forward simulations. As we solve an expanded system with the forward
    # equations, we need a seperate problem for it 
    _odeProblem = ODEProblem{true, specializeLevel}(petabModel.odeSystem, petabModel.stateMap, [0.0, 5e3], petabModel.parameterMap, jac=true, sparse=sparseJacobian)
    odeProblem = remake(_odeProblem, p = convert.(Float64, _odeProblem.p), u0 = convert.(Float64, _odeProblem.u0))

    # If we are computing the cost, gradient and hessians accross several processes we need to send ODEProblem, and
    # PEtab structs to each process
    if numberOfprocesses > 1
        jobs, results = setUpProcesses(petabModel, odeSolver, solverAbsTol, solverRelTol, odeSolverAdjoint, sensealgAdjoint,
                                       sensealgAdjointSS, solverAdjointAbsTol, solverAdjointRelTol, odeSolverForwardEquations,
                                       sensealgForwardEquations, parameterInfo, measurementInfo, simulationInfo, θ_indices,
                                       priorInfo, odeProblem, chunkSize)
    else
        jobs, results = nothing, nothing
    end

    petabODECache = createPEtabODEProblemCache(gradientMethod, hessianMethod, petabModel, sensealgForwardEquations, measurementInfo, simulationInfo, θ_indices, chunkSize)
    petabODESolverCache = createPEtabODESolverCache(gradientMethod, hessianMethod, petabModel, simulationInfo, θ_indices, chunkSize)

    # The cost (likelihood) can either be computed in the standard way or the Zygote way. The second consumes more
    # memory as in-place mutations are not compatible with Zygote
    computeCost = setUpCost(costMethod, odeProblem, odeSolver, petabODECache, petabODESolverCache, solverAbsTol, solverRelTol, petabModel,
                            simulationInfo, θ_indices, measurementInfo, parameterInfo, priorInfo,
                            numberOfprocesses=numberOfprocesses, jobs=jobs, results=results)

    # The gradient can either be computed via autodiff, forward sensitivity equations, adjoint sensitivity equations
    # and Zygote
    if gradientMethod === :ForwardEquations
        sensealgArg = sensealgForwardEquations
        odeSolverGradient = odeSolverForwardEquations
    elseif gradientMethod === :ForwardDiff
        sensealgArg = nothing
        odeSolverGradient = odeSolver
    elseif gradientMethod === :Adjoint
        sensealgArg = sensealgAdjoint
        odeSolverGradient = odeSolverAdjoint
    elseif gradientMethod === :Zygote
        sensealgArg = sensealgZygote
        odeSolverGradient = odeSolver
    end
    odeProblemGradient = gradientMethod === :ForwardEquations ? getODEProblemForwardEquations(odeProblem, sensealgArg) : getODEProblemForwardEquations(odeProblem, :NoSpecialProblem)
    solverAbsGradientTol, solverRelGradientTol = gradientMethod === :Adjoint ? (solverAdjointAbsTol, solverAdjointRelTol) : (solverAbsTol, solverRelTol)

    computeGradient! = setUpGradient(gradientMethod, odeProblemGradient, odeSolverGradient, petabODECache, petabODESolverCache, solverAbsGradientTol, solverRelGradientTol, petabModel,
                                     simulationInfo, θ_indices, measurementInfo, parameterInfo, priorInfo,
                                     chunkSize=chunkSize, numberOfprocesses=numberOfprocesses, jobs=jobs, results=results,
                                     splitOverConditions=splitOverConditions, sensealg=sensealgArg, 
                                     sensealgSS=sensealgAdjointSS)


    # The Hessian can either be computed via automatic differentation, or approximated via a block approximation or the
    # Gauss Newton method
    if hessianMethod === :GaussNewton
        odeSolverHessian = odeSolverForwardEquations
    else
        odeSolverHessian = odeSolver
    end
    computeHessian! = setUpHessian(hessianMethod, odeProblem, odeSolverHessian, petabODECache, petabODESolverCache, solverAbsTol, solverRelTol, 
                                   petabModel, simulationInfo, θ_indices, measurementInfo, parameterInfo, priorInfo, chunkSize,
                                   numberOfprocesses=numberOfprocesses, jobs=jobs, results=results, splitOverConditions=splitOverConditions, 
                                   reuseS=reuseS)
    
    # Extract nominal parameter vector and parameter bounds. If needed transform parameters
    θ_estNames = θ_indices.θ_estNames
    lowerBounds = [parameterInfo.lowerBound[findfirst(x -> x == θ_estNames[i], parameterInfo.parameterId)] for i in eachindex(θ_estNames)]
    upperBounds = [parameterInfo.upperBound[findfirst(x -> x == θ_estNames[i], parameterInfo.parameterId)] for i in eachindex(θ_estNames)]
    θ_nominal = [parameterInfo.nominalValue[findfirst(x -> x == θ_estNames[i], parameterInfo.parameterId)] for i in eachindex(θ_estNames)]
    transformθ!(lowerBounds, θ_estNames, θ_indices, reverseTransform=true)
    transformθ!(upperBounds, θ_estNames, θ_indices, reverseTransform=true)
    θ_nominalT = transformθ(θ_nominal, θ_estNames, θ_indices, reverseTransform=true)

    petabProblem = PEtabODEProblem(computeCost,
                                   computeGradient!,
                                   computeHessian!,
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
                                   petabModel)
    return petabProblem
end


function setUpCost(whichMethod::Symbol,
                   odeProblem::ODEProblem,
                   odeSolver::SciMLAlgorithm,
                   petabODECache::PEtabODEProblemCache,
                   petabODESolverCache::PEtabODESolverCache,
                   solverAbsTol::Float64,
                   solverRelTol::Float64,
                   petabModel::PEtabModel,
                   simulationInfo::SimulationInfo,
                   θ_indices::ParameterIndices,
                   measurementInfo::MeasurementsInfo,
                   parameterInfo::ParametersInfo,
                   priorInfo::PriorInfo;
                   sensealgZygote=ForwardDiffSensitivity(),
                   numberOfprocesses::Int64=1,
                   jobs=nothing,
                   results=nothing, 
                   computeResiduals::Bool=false)

    # Functions needed for mapping θ_est to the ODE problem, and then for solving said ODE-system
    if whichMethod == :Standard && numberOfprocesses == 1

        _changeODEProblemParameters! = (pODEProblem, u0, θ_est) -> changeODEProblemParameters!(pODEProblem, u0, θ_est, θ_indices, petabModel)
        changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
        _solveODEAllExperimentalConditions! = (odeSolutions, odeProblem, θ_dynamic, _expIDSolve) -> solveODEAllExperimentalConditions!(odeSolutions, odeProblem, θ_dynamic, petabODESolverCache, changeExperimentalCondition!, simulationInfo, odeSolver, solverAbsTol, solverRelTol, petabModel.computeTStops, onlySaveAtObservedTimes=true, expIDSolve=_expIDSolve, convertTspan=petabModel.convertTspan)
        __computeCost = (θ_est) -> computeCost(θ_est,
                                                odeProblem,
                                                petabModel,
                                                simulationInfo,
                                                θ_indices,
                                                measurementInfo,
                                                parameterInfo,
                                                _changeODEProblemParameters!,
                                                _solveODEAllExperimentalConditions!,
                                                priorInfo,
                                                petabODECache,
                                                expIDSolve=[:all], 
                                                computeResiduals=computeResiduals)

    elseif whichMethod == :Zygote
        changeExperimentalCondition = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
        _changeODEProblemParameters = (pODEProblem, θ_est) -> changeODEProblemParameters(pODEProblem, θ_est, θ_indices, petabModel)
        solveODEExperimentalCondition = (odeProblem, conditionId, θ_dynamic, tMax) -> solveOdeModelAtExperimentalCondZygote(odeProblem, conditionId, θ_dynamic, tMax, changeExperimentalCondition, measurementInfo, simulationInfo, odeSolver, solverAbsTol, solverRelTol, sensealgZygote, petabModel.computeTStops)
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
                       odeSolver::SciMLAlgorithm,
                       petabODECache::PEtabODEProblemCache,
                       petabODESolverCache::PEtabODESolverCache,
                       solverAbsTol::Float64,
                       solverRelTol::Float64,
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
    θ_nonDynamic = petabODECache.θ_observable

    if whichMethod == :ForwardDiff && numberOfprocesses == 1

        _changeODEProblemParameters! = (pODEProblem, u0, θ_est) -> changeODEProblemParameters!(pODEProblem, u0, θ_est, θ_indices, petabModel)
        changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
        _solveODEAllExperimentalConditions! = (odeSolutions, odeProblem, θ_dynamic, _expIDSolve) -> solveODEAllExperimentalConditions!(odeSolutions, odeProblem, θ_dynamic, petabODESolverCache, changeExperimentalCondition!, simulationInfo, odeSolver, solverAbsTol, solverRelTol, petabModel.computeTStops, onlySaveAtObservedTimes=true, expIDSolve=_expIDSolve, convertTspan=petabModel.convertTspan)

        # Compute gradient for parameters which are a part of the ODE-system (dynamic parameters)
        computeCostDynamicθ = (x) -> computeCostSolveODE(x, θ_sd, θ_observable, θ_nonDynamic, odeProblem, petabModel,
                                                         simulationInfo, θ_indices, measurementInfo, parameterInfo,
                                                         _changeODEProblemParameters!, _solveODEAllExperimentalConditions!,
                                                         petabODECache,
                                                         computeGradientDynamicθ=true, expIDSolve=[:all])
        if !isnothing(chunkSize)
            cfg = ForwardDiff.GradientConfig(computeCostDynamicθ, θ_dynamic, ForwardDiff.Chunk(chunkSize))
        else
            cfg = ForwardDiff.GradientConfig(computeCostDynamicθ, θ_dynamic, ForwardDiff.Chunk(θ_dynamic))
        end                                                         

        iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = getIndicesParametersNotInODESystem(θ_indices)
        computeCostNotODESystemθ = (x) -> computeCostNotSolveODE(x[iθ_sd], x[iθ_observable], x[iθ_nonDynamic],
                                                                 petabModel, simulationInfo, θ_indices, measurementInfo,
                                                                 parameterInfo, petabODECache, expIDSolve=[:all],
                                                                 computeGradientNotSolveAutoDiff=true)

        _computeGradient! = (gradient, θ_est) -> computeGradientAutoDiff!(gradient,
                                                                          θ_est,
                                                                          computeCostNotODESystemθ,
                                                                          computeCostDynamicθ,
                                                                          petabODECache,
                                                                          cfg,
                                                                          simulationInfo,
                                                                          θ_indices,
                                                                          priorInfo)

    elseif whichMethod === :ForwardEquations && numberOfprocesses == 1
        _changeODEProblemParameters! = (pODEProblem, u0, θ_est) -> changeODEProblemParameters!(pODEProblem, u0, θ_est, θ_indices, petabModel)
        if sensealg === :ForwardDiff
            changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
            _solveODEAllExperimentalConditions! = (odeSolutionValues, θ) -> solveODEAllExperimentalConditions!(odeSolutionValues, θ, petabODESolverCache, simulationInfo.odeSolutionsDerivatives, odeProblem, changeExperimentalCondition!, _changeODEProblemParameters!, simulationInfo, odeSolver, solverAbsTol, solverRelTol, petabModel.computeTStops, onlySaveAtObservedTimes=true, expIDSolve=[:all], convertTspan=petabModel.convertTspan)
            
            if !isnothing(chunkSize)
                cfg = ForwardDiff.JacobianConfig(_solveODEAllExperimentalConditions!, petabODECache.odeSolutionValues, petabODECache.θ_dynamic, ForwardDiff.Chunk(chunkSize))
            else
                cfg = ForwardDiff.JacobianConfig(_solveODEAllExperimentalConditions!, petabODECache.odeSolutionValues, petabODECache.θ_dynamic, ForwardDiff.Chunk(θ_dynamic))
            end

        else
            changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices, computeForwardSensitivites=true)
            _solveODEAllExperimentalConditions! = (odeSolutions, odeProblem, θ_dynamic, _expIDSolve) -> solveODEAllExperimentalConditions!(odeSolutions, odeProblem, θ_dynamic, petabODESolverCache, changeExperimentalCondition!, simulationInfo, odeSolver, solverAbsTol, solverRelTol, petabModel.computeTStops, onlySaveAtObservedTimes=true, expIDSolve=_expIDSolve, convertTspan=petabModel.convertTspan)
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
                                                                                  _changeODEProblemParameters!,
                                                                                  _solveODEAllExperimentalConditions!,
                                                                                  priorInfo,
                                                                                  cfg,
                                                                                  petabODECache,
                                                                                  expIDSolve=[:all])

    elseif whichMethod === :Adjoint && numberOfprocesses == 1

        iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = getIndicesParametersNotInODESystem(θ_indices)
        computeCostNotODESystemθ = (x) -> computeCostNotSolveODE(x[iθ_sd], x[iθ_observable], x[iθ_nonDynamic],
                                                                 petabModel, simulationInfo, θ_indices, measurementInfo,
                                                                 parameterInfo, petabODECache, expIDSolve=[:all],
                                                                 computeGradientNotSolveAdjoint=true)

        _changeODEProblemParameters! = (pODEProblem, u0, θ_est) -> changeODEProblemParameters!(pODEProblem, u0, θ_est, θ_indices, petabModel)
        changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
        _solveODEAllExperimentalConditions! = (odeSolutions, odeProblem, θ_dynamic, _expIDSolve) -> solveODEAllExperimentalConditions!(odeSolutions, odeProblem, θ_dynamic, petabODESolverCache, changeExperimentalCondition!, simulationInfo, odeSolver, solverAbsTol, solverRelTol, petabModel.computeTStops, denseSolution=true, expIDSolve=_expIDSolve, trackCallback=true)
        _computeGradient! = (gradient, θ_est) -> computeGradientAdjointEquations!(gradient,
                                                                                 θ_est,
                                                                                 odeSolver,
                                                                                 computeCostNotODESystemθ,
                                                                                 sensealg,
                                                                                 sensealgSS,
                                                                                 solverAbsTol,
                                                                                 solverRelTol,
                                                                                 odeProblem,
                                                                                 petabModel,
                                                                                 simulationInfo,
                                                                                 θ_indices,
                                                                                 measurementInfo,
                                                                                 parameterInfo,
                                                                                 _changeODEProblemParameters!,
                                                                                 _solveODEAllExperimentalConditions!,
                                                                                 priorInfo,
                                                                                 petabODECache,
                                                                                 expIDSolve=[:all])

    elseif whichMethod === :Zygote

        changeExperimentalCondition = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
        _changeODEProblemParameters = (pODEProblem, θ_est) -> changeODEProblemParameters(pODEProblem, θ_est, θ_indices, petabModel)
        solveODEExperimentalCondition = (odeProblem, conditionId, θ_dynamic, tMax) -> solveOdeModelAtExperimentalCondZygote(odeProblem, conditionId, θ_dynamic, tMax, changeExperimentalCondition, measurementInfo, simulationInfo, odeSolver, solverAbsTol, solverRelTol, sensealg, petabModel.computeTStops)
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
                      odeSolver::SciMLAlgorithm,
                      petabODECache::PEtabODEProblemCache,
                      petabODESolverCache::PEtabODESolverCache,
                      solverAbsTol::Float64,
                      solverRelTol::Float64,
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
    θ_nonDynamic = petabODECache.θ_observable                      

    # Functions needed for mapping θ_est to the ODE problem, and then for solving said ODE-system
    if (whichMethod === :ForwardDiff || whichMethod === :BlockForwardDiff) && numberOfprocesses == 1
        _changeODEProblemParameters! = (pODEProblem, u0, θ_est) -> changeODEProblemParameters!(pODEProblem, u0, θ_est, θ_indices, petabModel)
        changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
        _solveODEAllExperimentalConditions! = (odeSolutions, odeProblem, θ_dynamic, _expIDSolve) -> solveODEAllExperimentalConditions!(odeSolutions, odeProblem, θ_dynamic, petabODESolverCache, changeExperimentalCondition!, simulationInfo, odeSolver, solverAbsTol, solverRelTol, petabModel.computeTStops, onlySaveAtObservedTimes=true, expIDSolve=_expIDSolve, convertTspan=petabModel.convertTspan)

        if whichMethod === :ForwardDiff

            _evalHessian = (θ_est) -> computeCost(θ_est, odeProblem, petabModel, simulationInfo, θ_indices,
                                                  measurementInfo, parameterInfo, _changeODEProblemParameters!,
                                                  _solveODEAllExperimentalConditions!, priorInfo, petabODECache, 
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

            # Compute gradient for parameters which are a part of the ODE-system (dynamic parameters)
            computeCostDynamicθ = (x) -> computeCostSolveODE(x, θ_sd, θ_observable, θ_nonDynamic, odeProblem, petabModel,
                                                             simulationInfo, θ_indices, measurementInfo, parameterInfo,
                                                             _changeODEProblemParameters!, _solveODEAllExperimentalConditions!,
                                                             petabODECache,
                                                             computeGradientDynamicθ=true, expIDSolve=[:all])
            if !isnothing(chunkSize)
                cfg = ForwardDiff.HessianConfig(computeCostDynamicθ, θ_dynamic, ForwardDiff.Chunk(chunkSize))
            else
                cfg = ForwardDiff.HessianConfig(computeCostDynamicθ, θ_dynamic, ForwardDiff.Chunk(θ_dynamic))
            end                                                         

            iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = getIndicesParametersNotInODESystem(θ_indices)
            computeCostNotODESystemθ = (x) -> computeCostNotSolveODE(x[iθ_sd], x[iθ_observable], x[iθ_nonDynamic],
                                                                    petabModel, simulationInfo, θ_indices, measurementInfo,
                                                                    parameterInfo, petabODECache, expIDSolve=[:all],
                                                                    computeGradientNotSolveAutoDiff=true)

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

    elseif whichMethod == :GaussNewton && numberOfprocesses == 1
        
        _changeODEProblemParameters! = (pODEProblem, u0, θ_est) -> changeODEProblemParameters!(pODEProblem, u0, θ_est, θ_indices, petabModel)
        changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
        changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
        _solveODEAllExperimentalConditions! = (odeSolutionValues, θ) -> solveODEAllExperimentalConditions!(odeSolutionValues, θ, petabODESolverCache, simulationInfo.odeSolutionsDerivatives, odeProblem, changeExperimentalCondition!, _changeODEProblemParameters!, simulationInfo, odeSolver, solverAbsTol, solverRelTol, petabModel.computeTStops, onlySaveAtObservedTimes=true, expIDSolve=[:all])
        
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
                                                                                      _changeODEProblemParameters!,
                                                                                      _solveODEAllExperimentalConditions!,
                                                                                      priorInfo,
                                                                                      cfg, 
                                                                                      cfgNotSolveODE,
                                                                                      petabODECache,
                                                                                      expIDSolve=[:all],
                                                                                      reuseS=reuseS, 
                                                                                      returnJacobian=returnJacobian)

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
                                    hessianMethod::Symbol, 
                                    petabModel::PEtabModel,
                                    sensealgForwardEquations,
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
    if gradientMethod === :ForwardEquations && sensealgForwardEquations === :ForwardDiff
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
                                   hessianMethod::Symbol,
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