

import PEtab: processPriors, changeODEProblemParameters!, PriorInfo, transformθ, solveODEAllExperimentalConditions!, computeGaussNewtonHessianApproximation!, PEtabODESolverCache, createPEtabODESolverCache, createPEtabODEProblemCache


function _testCostGradientOrHessian(petabModel::PEtabModel,
                                    solver, 
                                    tol::Float64,
                                    p::Vector{Float64};
                                    computeCost::Bool=false,
                                    computeGradient::Bool=false,
                                    computeHessian::Bool=false,
                                    costMethod::Symbol=:Standard,
                                    gradientMethod::Symbol=:ForwardDiff,
                                    hessianMethod::Symbol=:ForwardDiff,
                                    solverSSRelTol::Float64=1e-6,
                                    solverSSAbsTol::Float64=1e-8, 
                                    sensealgForwardEquations::Union{Symbol, SciMLSensitivity.AbstractForwardSensitivityAlgorithm}=ForwardSensitivity(),
                                    sensealgZygote=ForwardDiffSensitivity(),
                                    sensealgAdjoint::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false)),
                                    sensealgAdjointSS::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm=SteadyStateAdjoint(),)

    petabProblem = setUpPEtabODEProblem(petabModel,
                                        solver;
                                        costMethod=costMethod,
                                        gradientMethod=gradientMethod,
                                        hessianMethod=hessianMethod,
                                        solverAbsTol=tol,
                                        solverRelTol=tol,
                                        solverSSRelTol=solverSSRelTol,
                                        solverSSAbsTol=solverSSAbsTol,
                                        odeSolverForwardEquations=solver,
                                        sensealgForwardEquations=sensealgForwardEquations,
                                        sensealgZygote=sensealgZygote,
                                        odeSolverAdjoint=solver,
                                        solverAdjointAbsTol=tol,
                                        solverAdjointRelTol=tol,
                                        sensealgAdjoint=sensealgAdjoint,
                                        sensealgAdjointSS=sensealgAdjointSS, 
                                        specializeLevel=SciMLBase.NoSpecialize)        

    if computeCost == true
        return petabProblem.computeCost(p)
    end

    if computeGradient == true
        _gradient = zeros(length(p))
        petabProblem.computeGradient!(_gradient, p)
        return _gradient
    end

    if computeHessian == true
        _hessian = zeros(length(p), length(p))
        petabProblem.computeCost(p)
        petabProblem.computeHessian!(_hessian, p)
        return _hessian
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

    petabODECache = createPEtabODEProblemCache(:ForwardEquations, :GaussNewton, petabModel, :ForwardDiff, measurementData, simulationInfo, paramEstIndices, nothing)
    petabODESolverCache = createPEtabODESolverCache(:ForwardEquations, :GaussNewton, petabModel, simulationInfo, paramEstIndices, nothing)

    # The time-span 5e3 is overwritten when performing actual forward simulations
    odeProb = ODEProblem(petabModel.odeSystem, petabModel.stateMap, (0.0, 5e3), petabModel.parameterMap, jac=true, sparse=false)
    odeProb = remake(odeProb, p = convert.(Float64, odeProb.p), u0 = convert.(Float64, odeProb.u0))
    # Functions to map experimental conditions and parameters correctly to the ODE model
    
    computeJacobian = PEtab.setUpHessian(:GaussNewton, odeProb, solver, petabODECache, petabODESolverCache,
                                         tol, tol, petabModel, simulationInfo, paramEstIndices, measurementData, 
                                         parameterData, priorInfo, nothing, returnJacobian=true)
    computeSumResiduals = PEtab.setUpCost(:Standard, odeProb, solver, petabODECache, petabODESolverCache,
                                         tol, tol, petabModel, simulationInfo, paramEstIndices, measurementData, 
                                         parameterData, priorInfo, computeResiduals=true)

    # Extract parameter vector
    namesParamEst = paramEstIndices.θ_estNames
    paramVecNominal = [parameterData.nominalValue[findfirst(x -> x == namesParamEst[i], parameterData.parameterId)] for i in eachindex(namesParamEst)]
    paramVec = transformθ(paramVecNominal, namesParamEst, paramEstIndices, reverseTransform=true)

    jacOut = zeros(length(paramVec), length(measurementData.time))
    residualGrad = ForwardDiff.gradient(computeSumResiduals, paramVec)
    computeJacobian(jacOut, paramVec)
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
