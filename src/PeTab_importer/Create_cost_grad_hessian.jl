include(joinpath(pwd(), "src", "PeTab_importer", "Common.jl"))
include(joinpath(pwd(), "src", "PeTab_importer", "Map_parameters.jl"))
include(joinpath(pwd(), "src", "PeTab_importer", "Create_obs_u0_sd_functions.jl"))
include(joinpath(pwd(), "src", "PeTab_importer", "Process_PeTab_files.jl"))
include(joinpath(pwd(), "src", "Common.jl"))


using Zygote


"""
    setUpCostGradHess(peTabModel::PeTabModel, solver, tol::Float64)

    For a PeTab-model set up functions for computing i) the likelihood, ii) likelhood gradient, 
    and iii) likelhood Hessian block approximation. The functions are stored in PeTabOpt-struct 
    that can be used as input to the optimizers. 

    Currently the gradient for dynamic parameters (part of ODE-system) is computed via ForwardDiff, 
    and ReverseDiff is used for observable and sd parameters. The hessian approximation assumes the 
    interaction betweeen dynamic and (observable, sd) parameters is zero.
"""
function setUpCostGradHess(peTabModel::PeTabModel, 
                           solver, 
                           tol::Float64; 
                           sensealg=ForwardDiffSensitivity(),
                           sparseJac::Bool=false, 
                           absTolSS::Float64=1e-8, 
                           relTolSS::Float64=1e-6)::PeTabOpt

    # Process PeTab files into type-stable Julia structs 
    experimentalConditionsFile, measurementDataFile, parameterDataFile, observablesDataFile = readDataFiles(peTabModel.dirModel, readObs=true)
    parameterData = processParameterData(parameterDataFile)
    measurementData = processMeasurementData(measurementDataFile, observablesDataFile) 
    simulationInfo = getSimulationInfo(measurementDataFile, measurementData, absTolSS=absTolSS, relTolSS=relTolSS)

    # Indices for mapping parameter-estimation vector to dynamic, observable and sd parameters correctly when calculating cost
    paramEstIndices = getIndicesParam(parameterData, measurementData, peTabModel.odeSystem, experimentalConditionsFile)
    
    # Set up potential prior for the parameters to estimate 
    priorInfo = getPriorInfo(paramEstIndices::ParameterIndices, parameterDataFile::DataFrame)::PriorInfo

    # Set model parameter values to those in the PeTab parameter data ensuring correct value of constant parameters 
    setParamToFileValues!(peTabModel.paramMap, peTabModel.stateMap, parameterData)

    # The time-span 5e3 is overwritten when performing actual forward simulations 
    odeProb = ODEProblem(peTabModel.odeSystem, peTabModel.stateMap, (0.0, 5e3), peTabModel.paramMap, jac=true, sparse=sparseJac)
    odeProb = remake(odeProb, p = convert.(Float64, odeProb.p), u0 = convert.(Float64, odeProb.u0))

    # Functions to map experimental conditions and parameters correctly to the ODE model 
    changeToExperimentalCondUse! = (pVec, u0Vec, expID, dynParamEst) -> changeExperimentalCondEst!(pVec, u0Vec, expID, dynParamEst, peTabModel, paramEstIndices)
    changeToExperimentalCondUse = (pVec, u0Vec, expID, dynParamEst) -> changeExperimentalCondEst(pVec, u0Vec, expID, dynParamEst, peTabModel, paramEstIndices)
    changeModelParamUse! = (pVec, u0Vec, paramEst) -> changeModelParam!(pVec, u0Vec, paramEst, paramEstIndices, peTabModel)
    changeModelParamUse = (pVec, paramEst) -> changeModelParam(pVec, paramEst, paramEstIndices, peTabModel)

    # Set up function which solves the ODE model for all conditions and stores result 
    solveOdeModelAllCondUse! = (solArrayArg, odeProbArg, dynParamEst) -> solveOdeModelAllExperimentalCond!(solArrayArg, odeProbArg, dynParamEst, changeToExperimentalCondUse!, measurementDataFile, simulationInfo, solver, tol, tol, onlySaveAtTobs=true)
    solveOdeModelAtCondZygoteUse = (odeProbArg, conditionId, dynParamEst, t_max) -> solveOdeModelAtExperimentalCondZygote(odeProbArg, conditionId, dynParamEst, t_max, changeToExperimentalCondUse, measurementData, simulationInfo, solver, tol, tol, sensealg)

    evalF = (paramVecEst) -> calcCost(paramVecEst, odeProb, peTabModel, simulationInfo, paramEstIndices, measurementData, parameterData, changeModelParamUse!, solveOdeModelAllCondUse!, priorInfo)
    evalFZygote = (paramVecEst) -> calcCostZygote(paramVecEst, odeProb, peTabModel, simulationInfo, paramEstIndices, measurementData, parameterData, changeModelParamUse, solveOdeModelAtCondZygoteUse, priorInfo)
    
    evalGradF = (grad, paramVecEst) -> calcGradCost!(grad, paramVecEst, odeProb, peTabModel, simulationInfo, paramEstIndices, measurementData, parameterData, changeModelParamUse!, solveOdeModelAllCondUse!, priorInfo)
    evalGradFZygote = (grad, paramVecEst) -> calcGradZygote!(grad, paramVecEst, odeProb, peTabModel, simulationInfo, paramEstIndices, measurementData, parameterData, changeModelParamUse, solveOdeModelAtCondZygoteUse, priorInfo)
    
    evalHessApprox = (hessianMat, paramVecEst) -> calcHessianApprox!(hessianMat, paramVecEst, odeProb, peTabModel, simulationInfo, paramEstIndices, measurementData, parameterData, changeModelParamUse!, solveOdeModelAllCondUse!, priorInfo)
    
    # This is subtle. When computing the hessian via autodiff it is important that the ODE-solution arrary with dual 
    # numbers is used, else dual numbers will be present when computing the cost which will crash the code when taking 
    # the gradient of non-dynamic parameters in optim. 
    _evalHess = (paramVecEst) -> calcCost(paramVecEst, odeProb, peTabModel, simulationInfo, paramEstIndices, measurementData, parameterData, changeModelParamUse!, solveOdeModelAllCondUse!, priorInfo, calcHessian=true)
    evalHess = (hessianMat, paramVec) -> begin hessianMat .= Symmetric(ForwardDiff.hessian(_evalHess, paramVec)) end
    
    # Lower and upper bounds for parameters to estimate 
    namesParamEst = paramEstIndices.namesParamEst
    lowerBounds = [parameterData.lowerBounds[findfirst(x -> x == namesParamEst[i], parameterData.parameterID)] for i in eachindex(namesParamEst)] 
    upperBounds = [parameterData.upperBounds[findfirst(x -> x == namesParamEst[i], parameterData.parameterID)] for i in eachindex(namesParamEst)] 
    # Parameter with nominal values in PeTab file 
    paramVecNominal = [parameterData.paramVal[findfirst(x -> x == namesParamEst[i], parameterData.parameterID)] for i in eachindex(namesParamEst)]

    # Transform upper and lower bounds if the case 
    transformParamVec!(lowerBounds, namesParamEst, parameterData, revTransform=true)
    transformParamVec!(upperBounds, namesParamEst, parameterData, revTransform=true)
    paramVecNominalTransformed = transformParamVec(paramVecNominal, namesParamEst, parameterData, revTransform=true)

    peTabOpt = PeTabOpt(evalF, 
                        evalFZygote,
                        evalGradF, 
                        evalGradFZygote,
                        evalHess,
                        evalHessApprox, 
                        length(namesParamEst), 
                        namesParamEst, 
                        paramVecNominal, 
                        paramVecNominalTransformed, 
                        lowerBounds, 
                        upperBounds, 
                        peTabModel.dirModel * "Cube" * peTabModel.modelName * ".csv",
                        peTabModel)
    return peTabOpt
end


"""
    calcCost(paramVecEst, 
             odeProb::ODEProblem,  
             peTabModel::PeTabModel,
             simulationInfo::SimulationInfo,
             paramIndices::ParameterIndices,
             measurementData::MeasurementData,
             parameterData::ParamData,
             changeModelParamUse!::Function,
             solveOdeModelAllCondUse!::Function;
             calcHessian::Bool=false)

    For a PeTab model compute the cost (likelhood) for a parameter vector 
    paramVecEst. With respect to paramVecEst (all other inputs fixed) 
    the function is compatible with ForwardDiff. 

    To compute the cost an ODE-problem, peTabModel, ODE simulation info, 
    indices to map parameter from paramVecEst, measurement data, parameter 
    data (e.g constant parameters), function to map parameters correctly to 
    ODE-model, and a function to solve the ODE model are required. These 
    are all set up correctly by the `setUpCostGradHess` function. 

    See also: [`setUpCostGradHess`]
"""
function calcCost(paramVecEst,
                  odeProb::ODEProblem,  
                  peTabModel::PeTabModel,
                  simulationInfo::SimulationInfo,
                  paramIndices::ParameterIndices,
                  measurementData::MeasurementData,
                  parameterData::ParamData,
                  changeModelParamUse!::Function,
                  solveOdeModelAllCondUse!::Function, 
                  priorInfo::PriorInfo;
                  calcHessian::Bool=false)
                                                            

    # Correctly map paramVecEst to dynmaic, observable and sd param. The new vectors 
    # are all distinct copies.
    dynamicParamEst = paramVecEst[paramIndices.iDynParam]
    obsParEst = paramVecEst[paramIndices.iObsParam]
    sdParamEst = paramVecEst[paramIndices.iSdParam]
    nonDynamicParamEst = paramVecEst[paramIndices.iNonDynParam]

    logLik = calcLogLikSolveODE(dynamicParamEst, sdParamEst, obsParEst, nonDynamicParamEst, odeProb, peTabModel, simulationInfo, paramIndices, measurementData, parameterData, changeModelParamUse!, solveOdeModelAllCondUse!, priorInfo, calcHessDynParam=calcHessian)

    if priorInfo.hasPriors == true
        paramVecEstTransformed = transformParamVec(paramVecEst, paramIndices.namesParamEst, parameterData)
        logLik += evalPriors(paramVecEst, paramVecEstTransformed, paramIndices.namesParamEst, paramIndices, priorInfo)
    end

    return logLik
end


"""
    calcGradCost!(grad::T1, 
                  paramVecEst, 
                  odeProb::ODEProblem,  
                  peTabModel::PeTabModel,
                  simulationInfo::SimulationInfo,
                  paramIndices::ParameterIndices,
                  measurementData::MeasurementData,
                  parameterData::ParamData, 
                  changeModelParamUse!::Function, 
                  solveOdeModelAllCondUse!::Function) where T1<:Array{<:AbstractFloat, 1}

    For a PeTab model compute inplace the gradient  of the cost (likelhood) for 
    a parameter vector paramVecEst. 

    Currently the gradient for dynamic parameters (part of ODE-system) is computed via ForwardDiff, 
    and ReverseDiff is used for observable and sd parameters. The input arguements are the same 
    as for `calcCost`, and everything is setup by `setUpCostGradHess` function.

    See also: [`setUpCostGradHess`]
"""
function calcGradCost!(grad::T1, 
                       paramVecEst::T2, 
                       odeProb::ODEProblem,  
                       peTabModel::PeTabModel,
                       simulationInfo::SimulationInfo,
                       paramIndices::ParameterIndices,
                       measurementData::MeasurementData,
                       parameterData::ParamData, 
                       changeModelParamUse!::Function, 
                       solveOdeModelAllCondUse!::Function, 
                       priorInfo::PriorInfo)    where {T1<:Array{<:AbstractFloat, 1}, 
                                                       T2<:Vector{<:Real}}
    
    # Split input into observeble and dynamic parameters 
    dynamicParamEst = paramVecEst[paramIndices.iDynParam]
    obsParEst = paramVecEst[paramIndices.iObsParam]
    sdParamEst = paramVecEst[paramIndices.iSdParam]
    noneDynParamEst = paramVecEst[paramIndices.iNonDynParam]
    namesSdParam = paramIndices.namesSdParam
    namesObsParam = paramIndices.namesObsParam
    namesNonDynParam = paramVecEst[paramIndices.namesNonDynParam]
    namesSdObsNonDynPar = paramIndices.namesSdObsNonDynPar

    # Calculate gradient seperately for dynamic and non dynamic parameter. 

    # I have tried to decrease run time here with chunking without success (deafult value performs best). Might be 
    # worth to look into a parellisation over the chunks (as for larger models each call takes relatively long time). 
    # Also parellisation of the chunks should be faster than paralellisation over experimental condtions.
    calcCostDyn = (x) -> calcLogLikSolveODE(x, sdParamEst, obsParEst, noneDynParamEst, odeProb, peTabModel, simulationInfo, paramIndices, measurementData, parameterData, changeModelParamUse!, solveOdeModelAllCondUse!, priorInfo, calcGradDynParam=true)
    grad[paramIndices.iDynParam] .= ForwardDiff.gradient(calcCostDyn, dynamicParamEst)::Vector{Float64}

    # Here it is crucial to account for that obs- and sd parameter can be overlapping. Thus, a name-map 
    # of both Sd and Obs param is used to account for this. This is not a worry for non-dynamic parameters.
    paramNotOdeSys = paramVecEst[paramIndices.iSdObsNonDynPar]
    iSdUse = [findfirst(x -> x == namesSdParam[i], namesSdObsNonDynPar) for i in eachindex(namesSdParam)]
    iObsUse = [findfirst(x -> x == namesObsParam[i],  namesSdObsNonDynPar) for i in eachindex(namesObsParam)]
    iNonDynUse = [findfirst(x -> x == namesNonDynParam[i],  namesSdObsNonDynPar) for i in eachindex(namesNonDynParam)]

    # TODO : Make choice of gradient availble 
    calcCostNonDyn = (x) -> calcLogLikNotSolveODE(dynamicParamEst, x[iSdUse], x[iObsUse], x[iNonDynUse], peTabModel, simulationInfo, paramIndices, measurementData, parameterData, priorInfo, calcGradObsSdParam=true)
    @views ReverseDiff.gradient!(grad[paramIndices.iSdObsNonDynPar], calcCostNonDyn, paramNotOdeSys)
end


"""
    calcHessianApprox!(hessian::T1, 
                       paramVecEst, 
                       odeProb::ODEProblem,  
                       peTabModel::PeTabModel,
                       simulationInfo::SimulationInfo,
                       paramIndices::ParameterIndices,
                       measurementData::MeasurementData,
                       parameterData::ParamData, 
                       changeModelParamUse!::Function,
                       solveOdeModelAllCondUse!::Function) where T1<:Array{<:AbstractFloat, 2}

    For a PeTab model compute inplace hessian approximation of the cost (likelhood) for 
    a parameter vector paramVecEst. 

    The hessian approximation assumes the interaction betweeen dynamic and (observable, sd) parameters is zero. 
    The input arguements are the same as for `calcCost`, and everything is setup by `setUpCostGradHess` function.

    See also: [`setUpCostGradHess`]
"""
function calcHessianApprox!(hessian::T1, 
                            paramVecEst::T2,
                            odeProb::ODEProblem,  
                            peTabModel::PeTabModel,
                            simulationInfo::SimulationInfo,
                            paramIndices::ParameterIndices,
                            measurementData::MeasurementData,
                            parameterData::ParamData, 
                            changeModelParamUse!::Function,
                            solveOdeModelAllCondUse!::Function, 
                            priorInfo::PriorInfo) where {T1<:Array{<:AbstractFloat, 2}, 
                                                         T2<:Vector{<:Real}}

    # Avoid incorrect non-zero values 
    hessian .= 0.0

    # Split input into observeble and dynamic parameters 
    dynamicParamEst = paramVecEst[paramIndices.iDynParam]
    obsParEst = paramVecEst[paramIndices.iObsParam]
    sdParamEst = paramVecEst[paramIndices.iSdParam]
    noneDynParamEst = paramVecEst[paramIndices.iNonDynParam]
    namesSdParam = paramIndices.namesSdParam
    namesObsParam = paramIndices.namesObsParam
    namesNonDynParam = paramVecEst[paramIndices.namesNonDynParam]
    namesSdObsNonDynPar = paramIndices.namesSdObsNonDynPar

    # Calculate gradient seperately for dynamic and non dynamic parameter. 
    calcCostDyn = (x) -> calcLogLikSolveODE(x, sdParamEst, obsParEst, noneDynParamEst, odeProb, peTabModel, simulationInfo, paramIndices, measurementData, parameterData, changeModelParamUse!, solveOdeModelAllCondUse!, priorInfo, calcHessDynParam=true)
    hessian[paramIndices.iDynParam, paramIndices.iDynParam] .= ForwardDiff.hessian(calcCostDyn, dynamicParamEst)::Matrix{Float64}

    # Here it is crucial to account for that obs- and sd parameter can be overlapping. Thus, a name-map 
    # of both Sd and Obs param is used to account for this 
    paramNotOdeSys = paramVecEst[paramIndices.iSdObsNonDynPar]
    iSdUse = [findfirst(x -> x == namesSdParam[i], namesSdObsNonDynPar) for i in eachindex(namesSdParam)]
    iObsUse = [findfirst(x -> x == namesObsParam[i],  namesSdObsNonDynPar) for i in eachindex(namesObsParam)]
    iNonDynUse = [findfirst(x -> x == namesNonDynParam[i],  namesSdObsNonDynPar) for i in eachindex(namesNonDynParam)]
    
    # Compute hessian for none dynamic parameters 
    calcCostNonDyn = (x) -> calcLogLikNotSolveODE(dynamicParamEst, x[iSdUse], x[iObsUse], x[iNonDynUse], peTabModel, simulationInfo, paramIndices, measurementData, parameterData, priorInfo)
    @views ReverseDiff.hessian!(hessian[paramIndices.iSdObsNonDynPar, paramIndices.iSdObsNonDynPar], calcCostNonDyn, paramNotOdeSys)

end


"""
    calcLogLikSolveODE(dynamicParamEst, 
                       sdParamEst, 
                       obsParEst, 
                       nonDynParamEst,
                       odeProb::ODEProblem,
                       peTabModel::PeTabModel,
                       simulationInfo::SimulationInfo,
                       measurementData ::MeasurementData, 
                       parameterData::ParamData, 
                       changeModelParamUse!::Function,
                       solveOdeModelAllCondUse!::Function, 
                       computeGradOrHess::Bool=false)

    Helper function computing the likelhood by solving the ODE system for all 
    PeTab-specifed experimental conditions using the dynamic-parameters, 
    sd-parameters and observable parameters. 

    When computing the cost and gradient/hessian for dynamic parameters the ODE 
    system must be solved before getting the likelhood. Besides the different 
    parameter vector the input arguements are the same as for `calcCost`, and 
    everything is setup by `setUpCostGradHess` function.

    See also: [`calcCost`, `setUpCostGradHess`]
"""
function calcLogLikSolveODE(dynamicParamEst,
                            sdParamEst,
                            obsParEst,
                            nonDynParamEst,
                            odeProb::ODEProblem,
                            peTabModel::PeTabModel,
                            simulationInfo::SimulationInfo,
                            paramIndices::ParameterIndices,
                            measurementData ::MeasurementData, 
                            parameterData::ParamData, 
                            changeModelParamUse!::Function,
                            solveOdeModelAllCondUse!::Function, 
                            priorInfo::PriorInfo;
                            calcHessDynParam::Bool=false, 
                            calcGradDynParam::Bool=false)::Real

    dynamicParamEstUse = transformParamVec(dynamicParamEst, paramIndices.namesDynParam, parameterData)
    sdParamEstUse = transformParamVec(sdParamEst, paramIndices.namesSdParam, parameterData)
    obsParEstUse = transformParamVec(obsParEst, paramIndices.namesObsParam, parameterData)
    nonDynParamEstUse = transformParamVec(nonDynParamEst, paramIndices.namesNonDynParam, parameterData)

    odeProbUse = remake(odeProb, p = convert.(eltype(dynamicParamEstUse), odeProb.p), u0 = convert.(eltype(dynamicParamEstUse), odeProb.u0))
    changeModelParamUse!(odeProbUse.p, odeProbUse.u0, dynamicParamEstUse)
    
    # If computing hessian or gradient store ODE solution in arrary with dual numbers, else use 
    # solution array with floats
    if calcHessDynParam == true || calcGradDynParam == true
        success = solveOdeModelAllCondUse!(simulationInfo.solArrayGrad, odeProbUse, dynamicParamEstUse)
    else
        success = solveOdeModelAllCondUse!(simulationInfo.solArray, odeProbUse, dynamicParamEstUse)
    end
    if success != true
        println("Failed to solve ODE model")
    end

    logLik = calcLogLik(dynamicParamEstUse, sdParamEstUse, obsParEstUse, nonDynParamEstUse, peTabModel, simulationInfo, paramIndices, measurementData, parameterData, calcHessDynParam=calcHessDynParam, calcGradDynParam=calcGradDynParam)

    if priorInfo.hasPriors == true && (calcHessDynParam || calcGradDynParam)        
        logLik += evalPriors(dynamicParamEstUse, dynamicParamEst, paramIndices.namesDynParam, paramIndices, priorInfo)
    end

    return logLik
end


"""
    calcLogLikNotSolveODE(dynamicParamEst, 
                          sdParamEst,
                          obsParamEst,
                          peTabModel::PeTabModel,
                          simulationInfo::SimulationInfo,
                          paramIndices::ParameterIndices,
                          measurementData::MeasurementData,
                          parameterData::ParamData)    

    Helper function computing the likelhood by given  an already existing ODE-solution stored 
    in simulationInfo using the dynamic-parameters, sd-parameters and observable parameters. 

    When computing the cost and gradient/hessian for the sd- and observable-parameters 
    only a solved ODE-system is needed (no need to resolve). This greatly reduces run-time 
    since a lot of dual numbers do not have to be propegated through the ODE solver.
    Besides the different parameter vector the input arguements are the same as for `calcCost`, 
    and everything is setup by `setUpCostGradHess` function.

    See also: [`calcCost`, `setUpCostGradHess`]
"""
function calcLogLikNotSolveODE(dynamicParamEst::T1, 
                               sdParamEst,
                               obsParamEst,
                               nonDynParamEst,
                               peTabModel::PeTabModel,
                               simulationInfo::SimulationInfo,
                               paramIndices::ParameterIndices,
                               measurementData::MeasurementData,
                               parameterData::ParamData, 
                               priorInfo::PriorInfo;
                               calcGradObsSdParam::Bool=false)::Real where T1<:Vector{<:Real}

    # To be able to use ReverseDiff sdParamEstUse and obsParamEstUse cannot be overwritten. 
    # Hence new vectors have to be created. Minimal overhead.
    dynamicParamEstUse = transformParamVec(dynamicParamEst, paramIndices.namesDynParam, parameterData)
    sdParamEstUse = transformParamVec(sdParamEst, paramIndices.namesSdParam, parameterData)
    obsParamEstUse = transformParamVec(obsParamEst, paramIndices.namesObsParam, parameterData)
    nonDynParamEstUse = transformParamVec(nonDynParamEst, paramIndices.namesNonDynParam, parameterData)

    logLik = calcLogLik(dynamicParamEstUse, sdParamEstUse, obsParamEstUse, nonDynParamEstUse, peTabModel, simulationInfo, paramIndices, measurementData, parameterData, calcGradObsSdParam=calcGradObsSdParam)
    
    if priorInfo.hasPriors == true 
        logLik += evalPriors(sdParamEstUse, sdParamEst, paramEstIndices.namesSdParam, paramIndices, priorInfo)
        logLik += evalPriors(obsParamEstUse, obsParamEst, paramEstIndices.namesObsParam, paramIndices, priorInfo)
        logLik += evalPriors(nonDynParamEstUse, nonDynParamEst, nonDynParamEst, paramIndices, priorInfo)
    end

    return logLik
end


"""
    calcLogLik(dynamicParamEst::T1,
                    sdParamEst, 
                    obsPar, 
                    peTabModel::PeTabModel,
                    simulationInfo::SimulationInfo,
                    paramIndices::ParameterIndices,
                    measurementData::MeasurementData, 
                    parameterData::ParamData;
                    gradHessDynParam::Bool=false)::Real where T1<:Vector{<:Real}  

    Helper function computing the likelhood by given after solving the ODE-system using
    using the dynamic-parameters, sd-parameters and observable parameters. 

    Currently for Gaussian data log10 and non-transformed data is accepted.

    See also: [`calcCost`, `setUpCostGradHess`]
"""
function calcLogLik(dynamicParamEst,
                    sdParamEst, 
                    obsPar, 
                    nonDynParamEst,
                    peTabModel::PeTabModel,
                    simulationInfo::SimulationInfo,
                    paramIndices::ParameterIndices,
                    measurementData::MeasurementData, 
                    parameterData::ParamData;
                    calcHessDynParam::Bool=false, 
                    calcGradDynParam::Bool=false, 
                    calcGradObsSdParam::Bool=false)::Real 

    if calcHessDynParam == true || calcGradDynParam == true || calcGradObsSdParam == true
        odeSolArray = simulationInfo.solArrayGrad
    else
        odeSolArray = simulationInfo.solArray
    end

    # Compute yMod and sd-val by looping through all experimental conditons. At the end 
    # update the likelihood 
    logLik = 0.0
    for conditionID in keys(measurementData.iPerConditionId)
        # Extract the ODE-solution for specific condition ID
        whichForwardSol = findfirst(x -> x == conditionID, simulationInfo.conditionIdSol)
        odeSol = odeSolArray[whichForwardSol]   
        logLik += calcLogLikExpCond(odeSol, dynamicParamEst, sdParamEst, obsPar, 
                                    nonDynParamEst, peTabModel, conditionID, paramIndices,
                                    measurementData, parameterData, calcGradObsSdParam)

        if isinf(logLik)
            return Inf
        end
    end

    return logLik
end



function calcCostZygote(paramVecEst,
                        odeProb::ODEProblem,  
                        peTabModel::PeTabModel,
                        simulationInfo::SimulationInfo,
                        paramIndices::ParameterIndices,
                        measurementData::MeasurementData,
                        parameterData::ParamData,
                        changeModelParamUse::Function,
                        solveOdeModelAllCondZygoteUse::Function, 
                        priorInfo::PriorInfo)
    
    # Correctly map paramVecEst to dynmaic, observable and sd param. The vectors, 
    # e.g dynamicParamEst, are distinct copies so transforming them will not change 
    # paramVecEst.
    dynamicParamEst = paramVecEst[paramIndices.iDynParam]
    obsParEst = paramVecEst[paramIndices.iObsParam]
    sdParamEst = paramVecEst[paramIndices.iSdParam]
    nonDynamicParamEst = paramVecEst[paramIndices.iNonDynParam]

    logLik = calcLogLikZygote(dynamicParamEst,
                              sdParamEst,
                              obsParEst,
                              nonDynamicParamEst,
                              odeProb,
                              peTabModel,
                              simulationInfo,
                              paramIndices,
                              measurementData,
                              parameterData,
                              changeModelParamUse,
                              solveOdeModelAllCondZygoteUse, 
                              priorInfo)

    if priorInfo.hasPriors == true
        paramVecEstTransformed = transformParamVec(paramVecEst, paramIndices.namesParamEst, parameterData)
        logLik += evalPriors(paramVecEst, paramVecEstTransformed, paramIndices.namesParamEst, paramIndices, priorInfo)
    end                                  

    return logLik                          
end


function calcGradZygote!(grad::T1, 
                         paramVecEst::T2, 
                         odeProb::ODEProblem,  
                         peTabModel::PeTabModel,
                         simulationInfo::SimulationInfo,
                         paramIndices::ParameterIndices,
                         measurementData::MeasurementData,
                         parameterData::ParamData, 
                         changeModelParamUse::Function, 
                         solveOdeModelAllCondZygoteUse::Function, 
                         priorInfo::PriorInfo) where {T1<:Array{<:AbstractFloat, 1}, 
                                                      T2<:Vector{<:Real}}
    
    # Split input into observeble and dynamic parameters 
    dynamicParamEst = paramVecEst[paramIndices.iDynParam]
    obsParEst = paramVecEst[paramIndices.iObsParam]
    sdParamEst = paramVecEst[paramIndices.iSdParam]
    noneDynParamEst = paramVecEst[paramIndices.iNonDynParam]
    namesSdParam = paramIndices.namesSdParam
    namesObsParam = paramIndices.namesObsParam
    namesNonDynParam = paramVecEst[paramIndices.namesNonDynParam]
    namesSdObsNonDynPar = paramIndices.namesSdObsNonDynPar

    # Calculate gradient seperately for dynamic and non dynamic parameter. This seems to considerble help Zygote 
    # if the model is large enough (which I guess has to with how the derivatives are overloaded when calling sol)

    # I have tried to decrease run time here with chunking without success (deafult value performs best). Might be 
    # worth to look into a parellisation over the chunks (as for larger models each call takes relatively long time). 
    # Also parellisation of the chunks should be faster than paralellisation over experimental condtions.
    calcCostDyn = (x) -> calcLogLikZygote(x, sdParamEst, obsParEst, noneDynParamEst, odeProb, peTabModel, simulationInfo, paramIndices, measurementData, parameterData, changeModelParamUse, solveOdeModelAllCondZygoteUse, priorInfo, evalGradDyn=true)
    grad[paramIndices.iDynParam] .= Zygote.gradient(calcCostDyn, dynamicParamEst)[1]

    # Here it is crucial to account for that obs- and sd parameter can be overlapping. Thus, a name-map 
    # of both Sd and Obs param is used to account for this. This is not a worry for non-dynamic parameters.
    paramNotOdeSys = paramVecEst[paramIndices.iSdObsNonDynPar]
    iSdUse = [findfirst(x -> x == namesSdParam[i], namesSdObsNonDynPar) for i in eachindex(namesSdParam)]
    iObsUse = [findfirst(x -> x == namesObsParam[i],  namesSdObsNonDynPar) for i in eachindex(namesObsParam)]
    iNonDynUse = [findfirst(x -> x == namesNonDynParam[i],  namesSdObsNonDynPar) for i in eachindex(namesNonDynParam)]

    # This is subtle. By using Zygote-ignore when solving the ODE the ODE solution is stored in simulationInfo.solArray
    # which can be used to compute the gradient for the non-dynamic parameters without having to resolve the ODE system.
    calcCostNonDyn = (x) -> calcLogLikNotSolveODE(dynamicParamEst, x[iSdUse], x[iObsUse], x[iNonDynUse], peTabModel, simulationInfo, paramIndices, measurementData, parameterData, priorInfo, calcGradObsSdParam=false)
    @views ReverseDiff.gradient!(grad[paramIndices.iSdObsNonDynPar], calcCostNonDyn, paramNotOdeSys)
end


# Computes the likelihood in such a in a Zygote compatible way, which mainly means that no arrays are mutated.
function calcLogLikZygote(dynamicParamEst,
                          sdParamEst,
                          obsParEst,
                          nonDynamicParamEst,
                          odeProb::ODEProblem,  
                          peTabModel::PeTabModel,
                          simulationInfo::SimulationInfo,
                          paramIndices::ParameterIndices,
                          measurementData::MeasurementData,
                          parameterData::ParamData,
                          changeModelParamUse::Function,
                          solveOdeModelAllCondZygoteUse::Function, 
                          priorInfo::PriorInfo; 
                          evalGradDyn::Bool=false)::Real

    # Correctly transform parameter if, for example, they are on the log-scale.
    dynamicParamEstUse = transformParamVec(dynamicParamEst, paramIndices.namesDynParam, parameterData)
    sdParamEstUse = transformParamVec(sdParamEst, paramIndices.namesSdParam, parameterData)
    obsParEstUse = transformParamVec(obsParEst, paramIndices.namesObsParam, parameterData)
    nonDynamicParamEstUse = transformParamVec(nonDynamicParamEst, paramIndices.namesNonDynParam, parameterData)                          
                                                            
    pOdeSysUse, u0Use = changeModelParamUse(odeProb.p, dynamicParamEstUse)
    odeProbUse = remake(odeProb, p = convert.(eltype(dynamicParamEstUse), pOdeSysUse), u0 = convert.(eltype(dynamicParamEstUse), u0Use))
    
    # Compute yMod and sd-val by looping through all experimental conditons. At the end 
    # update the likelihood 
    logLik = convert(eltype(dynamicParamEstUse), 0.0)
    for conditionID in keys(measurementData.iPerConditionId)
        
        # Solve ODE system 
        whichTMax = findfirst(x -> x == conditionID, simulationInfo.conditionIdSol)
        tMax = simulationInfo.tMaxForwardSim[whichTMax]
        
        odeSol, success = solveOdeModelAllCondZygoteUse(odeProbUse, conditionID, dynamicParamEstUse, tMax)
        if success != true
            return Inf
        end

        logLik += calcLogLikExpCond(odeSol, dynamicParamEstUse, sdParamEstUse, obsParEstUse, 
                                    nonDynamicParamEstUse, peTabModel, conditionID, paramIndices,
                                    measurementData, parameterData, false)

        if isinf(logLik)
            return logLik
        end
    end

    if priorInfo.hasPriors == true && evalGradDyn == true
        logLik += evalPriors(dynamicParamEstUse, dynamicParamEst, paramEstIndices.namesDynParam, paramIndices, priorInfo)
    end

    return logLik
end


function calcLogLikExpCond(odeSol::ODESolution,
                           dynamicParamEst,
                           sdParamEst, 
                           obsPar, 
                           nonDynParamEst,
                           peTabModel::PeTabModel,
                           conditionID::String,
                           paramIndices::ParameterIndices,
                           measurementData::MeasurementData,
                           parameterData::ParamData, 
                           calcGradObsSdParam::Bool)::Real

    # Compute yMod and sd for all observations having id conditionID 
    logLik = 0.0
    for i in measurementData.iPerConditionId[conditionID]
        # Compute Y-mod value 
        t = measurementData.tObs[i]
        if calcGradObsSdParam == true
            odeSolAtT = dualVecToFloatVec(odeSol[:, measurementData.iTObs[i]])
        else
            odeSolAtT = odeSol[:, measurementData.iTObs[i]]
        end
        mapObsParam = paramIndices.mapArrayObsParam[paramIndices.indexObsParamMap[i]]
        yMod = peTabModel.evalYmod(odeSolAtT, t, dynamicParamEst, obsPar, nonDynParamEst, parameterData, measurementData.observebleID[i], mapObsParam) 

        # Compute associated SD-value or extract said number if it is known 
        if typeof(measurementData.sdParams[i]) <: AbstractFloat
            sdVal = measurementData.sdParams[i]
        else
            mapSdParam = paramIndices.mapArraySdParam[paramIndices.indexSdParamMap[i]]
            sdVal = peTabModel.evalSd!(odeSolAtT, t, sdParamEst, dynamicParamEst, nonDynParamEst, parameterData, measurementData.observebleID[i], mapSdParam)
        end

        # Transform yMod is necessary
        yModTrans = transformObsOrData(yMod, measurementData.transformData[i])
            
        # By default a positive ODE solution is not enforced (even though the user can provide it as option).
        # In case with transformations on the data the code can crash, hence Inf is returned in case the 
        # model data transformation can not be perfomred. 
        if isinf(yModTrans)
            return Inf
        end

        # Update log-likelihood 
        if measurementData.transformData[i] == :lin
            logLik += log(sdVal) + 0.5*log(2*pi) + 0.5*((yModTrans - measurementData.yObsNotTransformed[i]) / sdVal)^2
        elseif measurementData.transformData[i] == :log10
            logLik += log(sdVal) + 0.5*log(2*pi) + log(log(10)) + log(exp10(measurementData.yObsTransformed[i])) + 0.5*( ( log(exp10(yModTrans)) - log(exp10(measurementData.yObsTransformed[i])) ) / (log(10)*sdVal))^2
        else
            println("Transformation ", measurementData.transformData[i], "not yet supported.")
            return Inf
        end   
    end

    return logLik
end


# Evaluate contribution of potential prior to the final cost function value. Not mutating so works with both Zygote 
# and forwardiff.
function evalPriors(paramVecTransformed, 
                    paramVecNotTransformed,
                    namesParamVec::Array{String, 1}, 
                    paramEstIndices::ParameterIndices, 
                    priorInfo::PriorInfo)::Real

    if priorInfo.hasPriors == false
        return 0.0
    end

    k = 0
    priorContribution = 0.0
    for i in eachindex(paramVecNotTransformed)
        iParam = findfirst(x -> x == namesParamVec[i], paramEstIndices.namesParamEst)
        if priorInfo.priorOnParamScale[iParam] == true
            pInput = paramVecTransformed[i]
        else
            pInput = paramVecNotTransformed[i]
        end
        priorContribution += priorInfo.logpdf[iParam](pInput)

        if priorInfo.logpdf[iParam](pInput) != 0
            k += 1
        end
    end

    return priorContribution
end


"""
    changeModelParam!(paramVecOdeModel, 
                      stateVecOdeModel,
                      paramVecEst,
                      paramEstNames::Array{String, 1},
                      paramIndices::ParameterIndices,
                      peTabModel::PeTabModel)

    Change the ODE parameter vector (paramVecOdeModel) and initial value vector (stateVecOdeModel)
    values to the values in parameter vector used for parameter estimation paramVecEst. 
    Basically, map the parameter-estiamtion vector to the ODE model.

    The function can handle that paramVecEst is a Float64 vector or a vector of Duals for the 
    gradient calculations. This function is used when computing the cost, and everything 
    is set up by `setUpCostGradHess`. 
     
    See also: [`setUpCostGradHess`]
"""
function changeModelParam!(pOdeSys, 
                           u0,
                           paramVecEst,
                           paramIndices::ParameterIndices,
                           peTabModel::PeTabModel)

    mapDynParam = paramIndices.mapDynParEst
    pOdeSys[mapDynParam.iDynParamInSys] .= paramVecEst[mapDynParam.iDynParamInVecEst]
    peTabModel.evalU0!(u0, pOdeSys) 
    
    return nothing
end


function changeModelParam(pOdeSys, 
                          paramVecEst,
                          paramIndices::ParameterIndices,
                          peTabModel::PeTabModel)

    # Helper function to not-inplace map parameters 
    function mapParamToEst(iUse::Integer, mapDynParam::MapDynParEst)
        whichIndex = findfirst(x -> x == iUse, mapDynParam.iDynParamInSys)
        return mapDynParam.iDynParamInVecEst[whichIndex]
    end

    mapDynParam = paramIndices.mapDynParEst
    pOdeSysRet = [i ∈ mapDynParam.iDynParamInSys ? paramVecEst[mapParamToEst(i, mapDynParam)] : pOdeSys[i] for i in eachindex(pOdeSys)]
    u0Ret = peTabModel.evalU0(pOdeSysRet) 
    
    return pOdeSysRet, u0Ret   
end


"""
    transformParamVec!(paramVec, namesParam::Array{String, 1}, paramData::ParamData; revTransform::Bool=false)

    Helper function which transforms in-place a parameter vector with parameters specied in namesParam according to the 
    transformation for said parameter specifid in paramData.shouldTransform. In case revTransform is true 
    performs the inverse parameter transformation (e.g exp10 instead of log10)
"""
function transformParamVec!(paramVec, 
                            namesParam::Array{String, 1}, 
                            paramData::ParamData; 
                            revTransform::Bool=false) 
    
    @inbounds for i in eachindex(paramVec)
        iParam = findfirst(x -> x == namesParam[i], paramData.parameterID)
        if isnothing(iParam)
            println("Warning : Could not find paramID for $namesParam")
        end
        if paramData.logScale[iParam] == true && revTransform == false
            paramVec[i] = exp10(paramVec[i])
        elseif paramData.logScale[iParam] == true && revTransform == true
            paramVec[i] = log10(paramVec[i])
        end
    end
end


"""
    transformParamVec!(paramVec, namesParam::Array{String, 1}, paramData::ParamData; revTransform::Bool=false)

    Helper function which returns a transformed parameter vector with parameters specied in namesParam according to the 
    transformation for said parameter specifid in paramData.shouldTransform. In case revTransform is true 
    performs the inverse parameter transformation (e.g exp10 instead of log10). 

    The function is fully compatible with Zygote.
"""
function transformParamVec(paramVec, 
                           namesParam::Array{String, 1}, 
                           paramData::ParamData; 
                           revTransform::Bool=false)
    
    iParam = [findfirst(x -> x == namesParam[i], paramData.parameterID) for i in eachindex(namesParam)]
    shouldTransform = [paramData.logScale[i] == true ? true : false for i in iParam]
    shouldNotTransform = .!shouldTransform

    if revTransform == false
        return exp10.(paramVec) .* shouldTransform .+ paramVec .* shouldNotTransform
    else
        return log10.(paramVec) .* shouldTransform .+ paramVec .* shouldNotTransform
    end
end


function dualVecToFloatVec(dualVec::T1)::Vector{Float64} where T1<:Vector{<:ForwardDiff.Dual}
    return [dualVec[i].value for i in eachindex(dualVec)]
end
