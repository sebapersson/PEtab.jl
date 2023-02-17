

import PEtab: processPriors, changeODEProblemParameters!, PriorInfo, transformθ, solveODEAllExperimentalConditions!, computeGaussNewtonHessianApproximation!


function _testCostGradientOrHessian(petabProblem::PEtabODEProblem,
                                    p::Vector{Float64};
                                    cost::Bool=false,
                                    costZygote::Bool=false,
                                    gradientAutoDiff::Bool=false,
                                    gradientForwardEquations::Bool=false,
                                    gradientAdjoint::Bool=false,
                                    gradientZygote::Bool=false,
                                    hessian::Bool=false,
                                    hessianGN::Bool=false)

    if cost == true
        return petabProblem.computeCost(p)
    end

    if costZygote == true
        return petabProblem.computeCostZygote(p)
    end

    if gradientAutoDiff == true
        _gradientAutoDiff = zeros(length(p))
        petabProblem.computeGradientAutoDiff(_gradientAutoDiff, p)
        return _gradientAutoDiff
    end

    if gradientForwardEquations == true
        _gradientForwardEquations = zeros(length(p))
        petabProblem.computeGradientForwardEquations(_gradientForwardEquations, p)
        return _gradientForwardEquations
    end

    if gradientAdjoint == true
        _gradientAdjoint = zeros(length(p))
        petabProblem.computeGradientAdjoint(_gradientAdjoint, p)
        return _gradientAdjoint
    end

    if gradientZygote == true
        _gradientZygote = zeros(length(p))
        petabProblem.computeGradientZygote(_gradientZygote, p)
        return _gradientZygote
    end

    if hessian == true
        _hessian = zeros(length(p), length(p))
        petabProblem.computeHessian(_hessian, p)
        return _hessian
    end

    if hessianGN == true
        _hessianGN = zeros(length(p), length(p))
        petabProblem.computeHessianGN(_hessianGN, p)
        return _hessianGN
    end
end


function checkGradientResiduals(petabModel::PEtabModel, solver, tol; verbose::Bool=true)

    # Process PeTab files into type-stable Julia structs
    experimentalConditionsFile, measurementDataFile, parameterDataFile, observablesDataFile = readPEtabFiles(petabModel)
    parameterData = processParameters(parameterDataFile)
    measurementData = processMeasurements(measurementDataFile, observablesDataFile)
    simulationInfo = processSimulationInfo(petabModel, measurementData, parameterData)

    # Indices for mapping parameter-estimation vector to dynamic, observable and sd parameters correctly when calculating cost
    paramEstIndices = computeIndicesθ(parameterData, measurementData, petabModel.odeSystem, experimentalConditionsFile)

    # Set model parameter values to those in the PeTab parameter data ensuring correct value of constant parameters
    setParamToFileValues!(petabModel.parameterMap, petabModel.stateMap, parameterData)
    priorInfo::PriorInfo = processPriors(paramEstIndices, parameterDataFile)

    # The time-span 5e3 is overwritten when performing actual forward simulations
    odeProb = ODEProblem(petabModel.odeSystem, petabModel.stateMap, (0.0, 5e3), petabModel.parameterMap, jac=true, sparse=false)
    odeProb = remake(odeProb, p = convert.(Float64, odeProb.p), u0 = convert.(Float64, odeProb.u0))
    # Functions to map experimental conditions and parameters correctly to the ODE model
    changeToExperimentalCondUse! = (pVec, u0Vec, expID, dynParamEst) -> _changeExperimentalCondition!(pVec, u0Vec, expID, dynParamEst, petabModel, paramEstIndices)
    changeModelParamUse! = (pVec, u0Vec, paramEst) -> changeODEProblemParameters!(pVec, u0Vec, paramEst, paramEstIndices, petabModel)
    solveOdeModelAllCondUse! = (solArrayArg, odeProbArg, dynParamEst, expIDSolveArg) -> solveODEAllExperimentalConditions!(solArrayArg, odeProbArg, dynParamEst, changeToExperimentalCondUse!, simulationInfo, solver, tol, tol, petabModel.computeTStops, onlySaveAtObservedTimes=true, expIDSolve=expIDSolveArg, convertTspan=petabModel.convertTspan)
    nTimePointsSaveAt = sum(length(simulationInfo.timeObserved[experimentalConditionId]) for experimentalConditionId in simulationInfo.experimentalConditionId)
    nModelStates = length(odeProb.u0)
    odeSolutionValues = zeros(Float64, nModelStates, nTimePointsSaveAt)
    solveOdeModelAllCondGuassNewtonForwardEq! = (solArrayArg, SMat, odeProbArg, dynParamEst, expIDSolveArg) -> solveODEAllExperimentalConditions!(solArrayArg, SMat, odeProbArg, dynParamEst, changeToExperimentalCondUse!, changeModelParamUse!, simulationInfo, paramEstIndices, solver, tol, tol, petabModel.computeTStops, odeSolutionValues, onlySaveAtObservedTimes=true, expIDSolve=expIDSolveArg, convertTspan=petabModel.convertTspan, splitOverConditions=false)
    evalResiduals = (paramVecEst) -> PEtab.computeCost(paramVecEst, odeProb, petabModel, simulationInfo, paramEstIndices, measurementData, parameterData, changeModelParamUse!, solveOdeModelAllCondUse!, priorInfo, computeResiduals=true)
    evalJacResiduals = (out, paramVecEst) -> computeGaussNewtonHessianApproximation!(out, paramVecEst, odeProb, petabModel, simulationInfo, paramEstIndices, measurementData, parameterData, changeModelParamUse!, solveOdeModelAllCondGuassNewtonForwardEq!, priorInfo, returnJacobian=true)

    # Extract parameter vector
    namesParamEst = paramEstIndices.θ_estNames
    paramVecNominal = [parameterData.nominalValue[findfirst(x -> x == namesParamEst[i], parameterData.parameterId)] for i in eachindex(namesParamEst)]
    paramVec = transformθ(paramVecNominal, namesParamEst, paramEstIndices, reverseTransform=true)

    jacOut = zeros(length(paramVec), length(measurementData.time))
    residualGrad = ForwardDiff.gradient(evalResiduals, paramVec)
    evalJacResiduals(jacOut, paramVec)
    sqDiffResidual = sum((sum(jacOut, dims=2) - residualGrad).^2)
    @test sqDiffResidual ≤ 1e-5
end


function getFileODEvalues(petabModel::PEtabModel)

    # Change model parameters
    experimentalConditionsFile, measurementDataFile, parameterDataFile, observablesDataFile = readPEtabFiles(petabModel)
    parameterInfo = processParameters(parameterDataFile)
    measurementInfo = processMeasurements(measurementDataFile, observablesDataFile)
    θ_indices = computeIndicesθ(parameterInfo, measurementInfo, petabModel.odeSystem, experimentalConditionsFile)

    θ_estNames = θ_indices.θ_estNames
    θ_est = parameterInfo.nominalValue[findall(x -> x ∈ θ_estNames, parameterInfo.parameterId)]

    return θ_est[θ_indices.iθ_dynamic]
end
