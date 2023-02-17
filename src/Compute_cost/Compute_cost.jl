function computeCost(θ_est::AbstractVector,
                     odeProblem::ODEProblem,  
                     petabModel::PEtabModel,
                     simulationInfo::SimulationInfo,
                     θ_indices::ParameterIndices,
                     measurementInfo::MeasurementsInfo,
                     parameterInfo::ParametersInfo,
                     changeODEProblemParameters!::Function,
                     solveOdeModelAllConditions!::Function,
                     priorInfo::PriorInfo;
                     expIDSolve::Vector{Symbol} = [:all],
                     computeHessian::Bool=false, 
                     computeResiduals::Bool=false)::Real

    θ_dynamic, θ_observable, θ_sd, θ_nonDynamic = splitParameterVector(θ_est, θ_indices) 

    cost = computeCostSolveODE(θ_dynamic, θ_sd, θ_observable, θ_nonDynamic, odeProblem, petabModel, simulationInfo, 
                               θ_indices, measurementInfo, parameterInfo, changeODEProblemParameters!, solveOdeModelAllConditions!, 
                               computeHessian=computeHessian, computeResiduals=computeResiduals, expIDSolve=expIDSolve)

    if priorInfo.hasPriors == true && computeHessian == false  
        θ_estT = transformθ(θ_est, θ_indices.θ_estNames, θ_indices)
        cost += computePriors(θ_est, θ_estT, θ_indices.θ_estNames, priorInfo)
    end

    return cost
end


function computeCostSolveODE(θ_dynamic::AbstractVector,
                             θ_sd::AbstractVector,
                             θ_observable::AbstractVector,
                             θ_nonDynamic::AbstractVector,
                             odeProblem::ODEProblem,
                             petabModel::PEtabModel,
                             simulationInfo::SimulationInfo,
                             θ_indices::ParameterIndices,
                             measurementInfo::MeasurementsInfo,
                             parameterInfo::ParametersInfo,
                             changeODEProblemParameters!::Function,
                             solveOdeModelAllConditions!::Function;
                             computeHessian::Bool=false, 
                             computeGradientDynamicθ::Bool=false, 
                             computeResiduals::Bool=false,
                             expIDSolve::Vector{Symbol} = [:all])::Real

    θ_dynamicT = transformθ(θ_dynamic, θ_indices.θ_dynamicNames, θ_indices)
    θ_sdT = transformθ(θ_sd, θ_indices.θ_sdNames, θ_indices)
    θ_observableT = transformθ(θ_observable, θ_indices.θ_observableNames, θ_indices)
    θ_nonDynamicT = transformθ(θ_nonDynamic, θ_indices.θ_nonDynamicNames, θ_indices)

    if petabModel.convertTspan == false
        _odeProblem = remake(odeProblem, p = convert.(eltype(θ_dynamicT), odeProblem.p), u0 = convert.(eltype(θ_dynamicT), odeProblem.u0))
    else
        _odeProblem = remake(odeProblem, p = convert.(eltype(θ_dynamicT), odeProblem.p), u0 = convert.(eltype(θ_dynamicT), odeProblem.u0), tspan=convert.(eltype(θ_dynamicT), odeProblem.tspan))
    end
    changeODEProblemParameters!(_odeProblem.p, _odeProblem.u0, θ_dynamicT)
    
    # If computing hessian or gradient store ODE solution in arrary with dual numbers, else use 
    # solution array with floats
    if computeHessian == true || computeGradientDynamicθ == true
        success = solveOdeModelAllConditions!(simulationInfo.odeSolutionsDerivatives, _odeProblem, θ_dynamicT, expIDSolve)
    else
        success = solveOdeModelAllConditions!(simulationInfo.odeSolutions, _odeProblem, θ_dynamicT, expIDSolve)
    end
    if success != true
        println("Failed to solve ODE model")
        return Inf
    end

    cost = _computeCost(θ_sdT, θ_observableT, θ_nonDynamicT, petabModel, simulationInfo, θ_indices, measurementInfo, 
                        parameterInfo, expIDSolve, 
                        computeHessian=computeHessian, 
                        computeGradientDynamicθ=computeGradientDynamicθ, 
                        computeResiduals=computeResiduals)

    return cost
end


function computeCostNotSolveODE(θ_sd::AbstractVector,
                                θ_observable::AbstractVector,
                                θ_nonDynamic::AbstractVector,
                                petabModel::PEtabModel,
                                simulationInfo::SimulationInfo,
                                θ_indices::ParameterIndices,
                                measurementInfo::MeasurementsInfo,
                                parameterInfo::ParametersInfo;
                                computeGradientNotSolveAutoDiff::Bool=false,
                                computeGradientNotSolveAdjoint::Bool=false,
                                computeGradientNotSolveForward::Bool=false, 
                                expIDSolve::Vector{Symbol} = [:all])::Real 

    # To be able to use ReverseDiff sdParamEstUse and obsParamEstUse cannot be overwritten. 
    # Hence new vectors have to be created. Minimal overhead.
    θ_sdT = transformθ(θ_sd, θ_indices.θ_sdNames, θ_indices)
    θ_observableT = transformθ(θ_observable, θ_indices.θ_observableNames, θ_indices)
    θ_nonDynamicT = transformθ(θ_nonDynamic, θ_indices.θ_nonDynamicNames, θ_indices)

    cost = _computeCost(θ_sdT, θ_observableT, θ_nonDynamicT, petabModel, simulationInfo, θ_indices, 
                        measurementInfo, parameterInfo, expIDSolve, 
                        computeGradientNotSolveAutoDiff=computeGradientNotSolveAutoDiff, 
                        computeGradientNotSolveAdjoint=computeGradientNotSolveAdjoint, 
                        computeGradientNotSolveForward=computeGradientNotSolveForward)

    return cost
end


function _computeCost(θ_sd::AbstractVector, 
                      θ_observable::AbstractVector, 
                      θ_nonDynamic::AbstractVector,
                      petabModel::PEtabModel,
                      simulationInfo::SimulationInfo,
                      θ_indices::ParameterIndices,
                      measurementInfo::MeasurementsInfo, 
                      parameterInfo::ParametersInfo,
                      expIDSolve::Vector{Symbol} = [:all];
                      computeHessian::Bool=false, 
                      computeGradientDynamicθ::Bool=false, 
                      computeResiduals::Bool=false,
                      computeGradientNotSolveAdjoint::Bool=false, 
                      computeGradientNotSolveForward::Bool=false, 
                      computeGradientNotSolveAutoDiff::Bool=false)::Real 

    if computeHessian == true || computeGradientDynamicθ == true || computeGradientNotSolveAdjoint == true || computeGradientNotSolveForward == true || computeGradientNotSolveAutoDiff == true 
        odeSolutions = simulationInfo.odeSolutionsDerivatives
    else
        odeSolutions = simulationInfo.odeSolutions
    end

    cost = 0.0
    for experimentalConditionId in simulationInfo.experimentalConditionId
        
        if expIDSolve[1] != :all && experimentalConditionId ∉ expIDSolve
            continue
        end

        # Extract the ODE-solution for specific condition ID
        odeSolution = odeSolutions[experimentalConditionId]  
        cost += computeCostExpCond(odeSolution, Float64[], θ_sd, θ_observable, θ_nonDynamic, petabModel, 
                                   experimentalConditionId, θ_indices, measurementInfo, parameterInfo, simulationInfo,
                                   computeResiduals=computeResiduals,
                                   computeGradientNotSolveAdjoint=computeGradientNotSolveAdjoint, 
                                   computeGradientNotSolveForward=computeGradientNotSolveForward, 
                                   computeGradientNotSolveAutoDiff=computeGradientNotSolveAutoDiff)

        if isinf(cost)
            return Inf
        end
    end

    return cost
end


function computeCostExpCond(odeSolution::ODESolution,
                            pODEProblemZygote::AbstractVector,
                            θ_sd::AbstractVector, 
                            θ_observable::AbstractVector, 
                            θ_nonDynamic::AbstractVector,
                            petabModel::PEtabModel,
                            experimentalConditionId::Symbol,
                            θ_indices::ParameterIndices,
                            measurementInfo::MeasurementsInfo,
                            parameterInfo::ParametersInfo,
                            simulationInfo::SimulationInfo; 
                            computeResiduals::Bool=false,
                            computeGradientNotSolveAdjoint::Bool=false, 
                            computeGradientNotSolveForward::Bool=false, 
                            computeGradientNotSolveAutoDiff::Bool=false, 
                            computeGradientθDynamicZygote::Bool=false)::Real

    if !(odeSolution.retcode == ReturnCode.Success || odeSolution.retcode == ReturnCode.Terminated)
        return Inf
    end

    cost = 0.0
    for iMeasurement in simulationInfo.iMeasurements[experimentalConditionId]
        
        t = measurementInfo.time[iMeasurement]

        # In these cases we only save the ODE at observed time-points and we do not want 
        # to extract Dual ODE solution 
        if computeGradientNotSolveForward == true || computeGradientNotSolveAutoDiff == true
            nModelStates = length(petabModel.stateNames)
            u = dualToFloat.(odeSolution[1:nModelStates, simulationInfo.iTimeODESolution[iMeasurement]]) 
            p = dualToFloat.(odeSolution.prob.p)
        # For adjoint sensitivity analysis we have a dense-ode solution 
        elseif computeGradientNotSolveAdjoint == true
            # In case we only have sol.t = 0.0 (or similar) interpolation does not work
            u = length(odeSolution.t) > 1 ? odeSolution(t) : odeSolution[1]
            p = odeSolution.prob.p

        elseif computeGradientθDynamicZygote == true            
            u = odeSolution[:, simulationInfo.iTimeODESolution[iMeasurement]]
            p = pODEProblemZygote

        # When we want to extract dual number from the ODE solution 
        else
            u = odeSolution[:, simulationInfo.iTimeODESolution[iMeasurement]]
            p = odeSolution.prob.p
        end

        hTransformed = computehTransformed(u, t, p, θ_observable, θ_nonDynamic, petabModel, iMeasurement, measurementInfo, θ_indices, parameterInfo)
        σ = computeσ(u, t, p, θ_sd, θ_nonDynamic, petabModel, iMeasurement, measurementInfo, θ_indices, parameterInfo)
            
        # By default a positive ODE solution is not enforced (even though the user can provide it as option).
        # In case with transformations on the data the code can crash, hence Inf is returned in case the 
        # model data transformation can not be perfomred. 
        if isinf(hTransformed)
            return Inf
        end

        # Update log-likelihood. In case of guass newton approximation we are only interested in the residuals, and here 
        # we allow the residuals to be computed to test the gauss-newton implementation 
        if computeResiduals == false
            if measurementInfo.measurementTransformation[iMeasurement] == :lin
                cost += log(σ) + 0.5*log(2*pi) + 0.5*((hTransformed - measurementInfo.measurement[iMeasurement]) / σ)^2
            elseif measurementInfo.measurementTransformation[iMeasurement] == :log10
                cost += log(σ) + 0.5*log(2*pi) + log(log(10)) + log(10)*measurementInfo.measurementT[iMeasurement] + 0.5*((hTransformed - measurementInfo.measurementT[iMeasurement]) / σ)^2
            else
                println("Transformation ", measurementInfo.measurementTransformation[iMeasurement], "not yet supported.")
                return Inf
            end   
        elseif computeResiduals == true
            if measurementInfo.measurementTransformation[iMeasurement] == :lin
                cost += ((hTransformed - measurementInfo.measurement[iMeasurement]) / σ)
            elseif measurementInfo.measurementTransformation[iMeasurement] == :log10
                cost += ((hTransformed - measurementInfo.measurementT[iMeasurement]) / σ)
            else
                println("Transformation ", measurementInfo.transformData[i], "not yet supported.")
                return Inf
            end   
        end
    end

    return cost
end