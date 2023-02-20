function setUpPEtabODEProblem(petabModel::PEtabModel,
                              odeSolver::SciMLAlgorithm;
                              solverAbsTol::Float64=1e-8,
                              solverRelTol::Float64=1e-8,
                              solverSSRelTol::Float64=1e-6,
                              solverSSAbsTol::Float64=1e-8,
                              sparseJacobian::Bool=false,
                              numberOfprocesses::Signed=1,
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
                              reuseS::Bool=false, 
                              specializeLevel=SciMLBase.FullSpecialize)::PEtabODEProblem

    if !(typeof(sensealgAdjointSS) <: SteadyStateAdjoint)
        println("If you are using adjoint sensitivity analysis for a model with PreEq-criteria the most the most efficient adjSensealgSS is usually SteadyStateAdjoint. The algorithm you have provided, ", sensealgAdjointSS, "might not work (as there are some bugs here). In case it does not work, and SteadyStateAdjoint fails (because a dependancy on time or a singular Jacobian) a good choice might be QuadratureAdjoint(autodiff=false, autojacvec=false)")
    end

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
    odeProblemForwardEquations = getODEProblemForwardEquations(odeProblem, sensealgForwardEquations)

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

    # The cost (likelihood) can either be computed in the standard way or the Zygote way. The second consumes more
    # memory as in-place mutations are not compatible with Zygote
    expIdSolve = [:all]
    computeCost = setUpCost(:Standard, odeProblem, odeSolver, solverAbsTol, solverRelTol, petabModel,
                            simulationInfo, θ_indices, measurementInfo, parameterInfo, priorInfo, expIdSolve,
                            numberOfprocesses=numberOfprocesses, jobs=jobs, results=results)
    computeCostZygote = setUpCost(:Zygote, odeProblem, odeSolver, solverAbsTol, solverRelTol,
                                  petabModel, simulationInfo, θ_indices, measurementInfo, parameterInfo,
                                  priorInfo, expIdSolve, sensealgZygote=sensealgZygote)

    # The gradient can either be computed via autodiff, forward sensitivity equations, adjoint sensitivity equations
    # and Zygote
    computeGradientAutoDiff = setUpGradient(:AutoDiff, odeProblem, odeSolver, solverAbsTol, solverRelTol, petabModel,
                                            simulationInfo, θ_indices, measurementInfo, parameterInfo, priorInfo, expIdSolve,
                                            chunkSize=chunkSize, numberOfprocesses=numberOfprocesses, jobs=jobs, results=results,
                                            splitOverConditions=splitOverConditions)
    computeGradientForwardEquations = setUpGradient(:ForwardEquations, odeProblemForwardEquations, odeSolverForwardEquations, solverAbsTol,
                                                    solverRelTol, petabModel, simulationInfo, θ_indices, measurementInfo,
                                                    parameterInfo, priorInfo, expIdSolve, sensealg=sensealgForwardEquations, chunkSize=chunkSize,
                                                    numberOfprocesses=numberOfprocesses, jobs=jobs, results=results, splitOverConditions=splitOverConditions)
    computeGradientAdjoint = setUpGradient(:Adjoint, odeProblem, odeSolverAdjoint, solverAdjointAbsTol, solverAdjointRelTol,
                                           petabModel, simulationInfo, θ_indices, measurementInfo, parameterInfo, priorInfo,
                                           expIdSolve, sensealg=sensealgAdjoint, sensealgSS=sensealgAdjointSS,
                                           numberOfprocesses=numberOfprocesses, jobs=jobs, results=results)
    computeGradientZygote = setUpGradient(:Zygote, odeProblem, odeSolver, solverAbsTol, solverRelTol, petabModel,
                                          simulationInfo, θ_indices, measurementInfo, parameterInfo, priorInfo,
                                          expIdSolve, sensealg=sensealgZygote)

    # The Hessian can either be computed via automatic differentation, or approximated via a block approximation or the
    # Gauss Newton method
    computeHessian = setUpHessian(:HessianAutoDiff, odeProblem, odeSolver, solverAbsTol, solverRelTol, petabModel, simulationInfo,
                                  θ_indices, measurementInfo, parameterInfo, priorInfo, chunkSize, expIdSolve,
                                  numberOfprocesses=numberOfprocesses, jobs=jobs, results=results, splitOverConditions=splitOverConditions)
    computeHessianBlock = setUpHessian(:BlockAutoDiff, odeProblem, odeSolver, solverAbsTol, solverRelTol, petabModel, simulationInfo,
                                        θ_indices, measurementInfo, parameterInfo, priorInfo, chunkSize, expIdSolve,
                                        numberOfprocesses=numberOfprocesses, jobs=jobs, results=results, splitOverConditions=splitOverConditions)
    computeHessianGN = setUpHessian(:GaussNewton, odeProblem, odeSolver, solverAbsTol, solverRelTol, petabModel, simulationInfo,
                                    θ_indices, measurementInfo, parameterInfo, priorInfo, chunkSize, expIdSolve, reuseS=reuseS,
                                    numberOfprocesses=numberOfprocesses, jobs=jobs, results=results, splitOverConditions=splitOverConditions)

    # Extract nominal parameter vector and parameter bounds. If needed transform parameters
    θ_estNames = θ_indices.θ_estNames
    lowerBounds = [parameterInfo.lowerBound[findfirst(x -> x == θ_estNames[i], parameterInfo.parameterId)] for i in eachindex(θ_estNames)]
    upperBounds = [parameterInfo.upperBound[findfirst(x -> x == θ_estNames[i], parameterInfo.parameterId)] for i in eachindex(θ_estNames)]
    θ_nominal = [parameterInfo.nominalValue[findfirst(x -> x == θ_estNames[i], parameterInfo.parameterId)] for i in eachindex(θ_estNames)]
    transformθ!(lowerBounds, θ_estNames, θ_indices, reverseTransform=true)
    transformθ!(upperBounds, θ_estNames, θ_indices, reverseTransform=true)
    θ_nominalT = transformθ(θ_nominal, θ_estNames, θ_indices, reverseTransform=true)

    petabProblem = PEtabODEProblem(computeCost,
                                   computeCostZygote,
                                   computeGradientAutoDiff,
                                   computeGradientZygote,
                                   computeGradientAdjoint,
                                   computeGradientForwardEquations,
                                   computeHessian,
                                   computeHessianBlock,
                                   computeHessianGN,
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
                   solverAbsTol::Float64,
                   solverRelTol::Float64,
                   petabModel::PEtabModel,
                   simulationInfo::SimulationInfo,
                   θ_indices::ParameterIndices,
                   measurementInfo::MeasurementsInfo,
                   parameterInfo::ParametersInfo,
                   priorInfo::PriorInfo,
                   expIDSolveArg::Vector{Symbol};
                   sensealgZygote=ForwardDiffSensitivity(),
                   numberOfprocesses::Int64=1,
                   jobs=nothing,
                   results=nothing)

    # Functions needed for mapping θ_est to the ODE problem, and then for solving said ODE-system
    if whichMethod == :Standard && numberOfprocesses == 1

        _changeODEProblemParameters! = (pODEProblem, u0, θ_est) -> changeODEProblemParameters!(pODEProblem, u0, θ_est, θ_indices, petabModel)
        changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
        _solveODEAllExperimentalConditions! = (odeSolutions, odeProblem, θ_dynamic, _expIDSolve) -> solveODEAllExperimentalConditions!(odeSolutions, odeProblem, θ_dynamic, changeExperimentalCondition!, simulationInfo, odeSolver, solverAbsTol, solverRelTol, petabModel.computeTStops, onlySaveAtObservedTimes=true, expIDSolve=_expIDSolve, convertTspan=petabModel.convertTspan)
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
                                                expIDSolve=expIDSolveArg)

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
                       solverAbsTol::Float64,
                       solverRelTol::Float64,
                       petabModel::PEtabModel,
                       simulationInfo::SimulationInfo,
                       θ_indices::ParameterIndices,
                       measurementInfo::MeasurementsInfo,
                       parameterInfo::ParametersInfo,
                       priorInfo::PriorInfo,
                       expIDSolve::Vector{Symbol};
                       chunkSize::Union{Nothing, Int64}=nothing,
                       sensealg=nothing,
                       sensealgSS=nothing,
                       numberOfprocesses::Int64=1,
                       jobs=nothing,
                       results=nothing,
                       splitOverConditions::Bool=false)

    # Functions needed for mapping θ_est to the ODE problem, and then for solving said ODE-system
    if whichMethod == :AutoDiff && numberOfprocesses == 1
        _changeODEProblemParameters! = (pODEProblem, u0, θ_est) -> changeODEProblemParameters!(pODEProblem, u0, θ_est, θ_indices, petabModel)
        changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
        _solveODEAllExperimentalConditions! = (odeSolutions, odeProblem, θ_dynamic, _expIDSolve) -> solveODEAllExperimentalConditions!(odeSolutions, odeProblem, θ_dynamic, changeExperimentalCondition!, simulationInfo, odeSolver, solverAbsTol, solverRelTol, petabModel.computeTStops, onlySaveAtObservedTimes=true, expIDSolve=_expIDSolve, convertTspan=petabModel.convertTspan)
        _computeGradient! = (gradient, θ_est) -> computeGradientAutoDiff!(gradient,
                                                                         θ_est,
                                                                         odeProblem,
                                                                         petabModel,
                                                                         simulationInfo,
                                                                         θ_indices,
                                                                         measurementInfo,
                                                                         parameterInfo,
                                                                         _changeODEProblemParameters!,
                                                                         _solveODEAllExperimentalConditions!,
                                                                         priorInfo,
                                                                         chunkSize,
                                                                         expIDSolve=expIDSolve,
                                                                         splitOverConditions=splitOverConditions)

    elseif whichMethod == :ForwardEquations && numberOfprocesses == 1
        _changeODEProblemParameters! = (pODEProblem, u0, θ_est) -> changeODEProblemParameters!(pODEProblem, u0, θ_est, θ_indices, petabModel)
        if sensealg == :AutoDiffForward
            nTimePointsSaveAt = sum(length(simulationInfo.timeObserved[experimentalConditionId]) for experimentalConditionId in simulationInfo.experimentalConditionId)
            nModelStates = length(odeProblem.u0)
            odeSolutionValues = zeros(Float64, nModelStates, nTimePointsSaveAt)
            changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
            _solveODEAllExperimentalConditions! = (odeSolutions, S, odeProblem, θ_dynamic, _expIDSolve) -> solveODEAllExperimentalConditions!(odeSolutions, S, odeProblem, θ_dynamic, changeExperimentalCondition!, _changeODEProblemParameters!, simulationInfo, θ_indices, odeSolver, solverAbsTol, solverRelTol, petabModel.computeTStops, odeSolutionValues, onlySaveAtObservedTimes=true, expIDSolve=_expIDSolve, chunkSize=chunkSize, convertTspan=petabModel.convertTspan, splitOverConditions=splitOverConditions)

        else
            changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices, computeForwardSensitivites=true)
            _solveODEAllExperimentalConditions! = (odeSolutions, odeProblem, θ_dynamic, _expIDSolve) -> solveODEAllExperimentalConditions!(odeSolutions, odeProblem, θ_dynamic, changeExperimentalCondition!, simulationInfo, odeSolver, solverAbsTol, solverRelTol, petabModel.computeTStops, onlySaveAtObservedTimes=true, expIDSolve=_expIDSolve)
        end
        _computeGradient! = (gradient, θ_est) -> computeGradientForwardEquations!(gradient,
                                                                                 θ_est,
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
                                                                                 expIDSolve=expIDSolve)

    elseif whichMethod == :Adjoint && numberOfprocesses == 1
        _changeODEProblemParameters! = (pODEProblem, u0, θ_est) -> changeODEProblemParameters!(pODEProblem, u0, θ_est, θ_indices, petabModel)
        changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
        _solveODEAllExperimentalConditions! = (odeSolutions, odeProblem, θ_dynamic, _expIDSolve) -> solveODEAllExperimentalConditions!(odeSolutions, odeProblem, θ_dynamic, changeExperimentalCondition!, simulationInfo, odeSolver, solverAbsTol, solverRelTol, petabModel.computeTStops, denseSolution=true, expIDSolve=_expIDSolve, trackCallback=true)
        _computeGradient! = (gradient, θ_est) -> computeGradientAdjointEquations!(gradient,
                                                                                 θ_est,
                                                                                 odeSolver,
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
                                                                                 expIDSolve=expIDSolve)

    elseif whichMethod == :Zygote

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
                                                                      priorInfo)

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
                      solverAbsTol::Float64,
                      solverRelTol::Float64,
                      petabModel::PEtabModel,
                      simulationInfo::SimulationInfo,
                      θ_indices::ParameterIndices,
                      measurementInfo::MeasurementsInfo,
                      parameterInfo::ParametersInfo,
                      priorInfo::PriorInfo,
                      chunkSize::Union{Nothing, Int64},
                      expIDSolve::Vector{Symbol};
                      reuseS::Bool=false,
                      numberOfprocesses::Int64=1,
                      jobs=nothing,
                      results=nothing,
                      splitOverConditions::Bool=false)

    # Functions needed for mapping θ_est to the ODE problem, and then for solving said ODE-system
    if (whichMethod == :HessianAutoDiff || whichMethod == :BlockAutoDiff) && numberOfprocesses == 1
        _changeODEProblemParameters! = (pODEProblem, u0, θ_est) -> changeODEProblemParameters!(pODEProblem, u0, θ_est, θ_indices, petabModel)
        changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
        _solveODEAllExperimentalConditions! = (odeSolutions, odeProblem, θ_dynamic, _expIDSolve) -> solveODEAllExperimentalConditions!(odeSolutions, odeProblem, θ_dynamic, changeExperimentalCondition!, simulationInfo, odeSolver, solverAbsTol, solverRelTol, petabModel.computeTStops, onlySaveAtObservedTimes=true, expIDSolve=_expIDSolve, convertTspan=petabModel.convertTspan)

        if whichMethod == :HessianAutoDiff
            _computeHessian = (hessian, θ_est) -> computeHessian!(hessian,
                                                                θ_est,
                                                                odeProblem,
                                                                petabModel,
                                                                simulationInfo,
                                                                θ_indices,
                                                                measurementInfo,
                                                                parameterInfo,
                                                                _changeODEProblemParameters!,
                                                                _solveODEAllExperimentalConditions!,
                                                                priorInfo,
                                                                chunkSize,
                                                                expIDSolve=expIDSolve,
                                                                splitOverConditions=splitOverConditions)
        else
            _computeHessian = (hessian, θ_est) -> computeHessianBlockApproximation!(hessian,
                                                                                  θ_est,
                                                                                  odeProblem,
                                                                                  petabModel,
                                                                                  simulationInfo,
                                                                                  θ_indices,
                                                                                  measurementInfo,
                                                                                  parameterInfo,
                                                                                  _changeODEProblemParameters!,
                                                                                  _solveODEAllExperimentalConditions!,
                                                                                  priorInfo,
                                                                                  chunkSize,
                                                                                  expIDSolve=expIDSolve,
                                                                                  splitOverConditions=splitOverConditions)
        end

    elseif whichMethod == :GaussNewton && numberOfprocesses == 1
        _changeODEProblemParameters! = (pODEProblem, u0, θ_est) -> changeODEProblemParameters!(pODEProblem, u0, θ_est, θ_indices, petabModel)
        changeExperimentalCondition! = (pODEProblem, u0, conditionId, θ_dynamic) -> _changeExperimentalCondition!(pODEProblem, u0, conditionId, θ_dynamic, petabModel, θ_indices)
        nTimePointsSaveAt = sum(length(simulationInfo.timeObserved[experimentalConditionId]) for experimentalConditionId in simulationInfo.experimentalConditionId)
        nModelStates = length(odeProblem.u0)
        odeSolutionValues = zeros(Float64, nModelStates, nTimePointsSaveAt)
        _solveODEAllExperimentalConditions! = (odeSolutions, S, odeProblem, θ_dynamic, _expIDSolve) -> solveODEAllExperimentalConditions!(odeSolutions, S, odeProblem, θ_dynamic, changeExperimentalCondition!, _changeODEProblemParameters!, simulationInfo, θ_indices, odeSolver, solverAbsTol, solverRelTol, petabModel.computeTStops, odeSolutionValues, onlySaveAtObservedTimes=true, expIDSolve=_expIDSolve, chunkSize=chunkSize, convertTspan=petabModel.convertTspan, splitOverConditions=splitOverConditions)
        _computeHessian = (hessian, θ_est) -> computeGaussNewtonHessianApproximation!(hessian,
                                                                                    θ_est,
                                                                                    odeProblem,
                                                                                    petabModel,
                                                                                    simulationInfo,
                                                                                    θ_indices,
                                                                                    measurementInfo,
                                                                                    parameterInfo,
                                                                                    _changeODEProblemParameters!,
                                                                                    _solveODEAllExperimentalConditions!,
                                                                                    priorInfo,
                                                                                    expIDSolve=expIDSolve,
                                                                                    reuseS=reuseS)

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
                                       sensealgForwardEquations::Symbol)::ODEProblem
    return deepcopy(odeProblem)
end
