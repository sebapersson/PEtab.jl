function compute_cost(θ_est::V,
                     odeProblem::ODEProblem,
                     ode_solver::ODESolver,
                     ss_solver::SteadyStateSolver,
                     petab_model::PEtabModel,
                     simulation_info::SimulationInfo,
                     θ_indices::ParameterIndices,
                     measurementInfo::MeasurementsInfo,
                     parameterInfo::ParametersInfo,
                     priorInfo::PriorInfo,
                     petabODECache::PEtabODEProblemCache,
                     petabODESolverCache::PEtabODESolverCache,
                     expIDSolve::Vector{Symbol},
                     compute_cost::Bool,
                     compute_hessian::Bool,
                     compute_residuals::Bool) where V

    θ_dynamic, θ_observable, θ_sd, θ_nonDynamic = splitParameterVector(θ_est, θ_indices)

    cost = compute_costSolveODE(θ_dynamic, θ_sd, θ_observable, θ_nonDynamic, odeProblem, ode_solver, ss_solver, petab_model,
                               simulation_info, θ_indices, measurementInfo, parameterInfo, petabODECache, petabODESolverCache,
                               compute_cost=compute_cost,
                               compute_hessian=compute_hessian,
                               compute_residuals=compute_residuals,
                               expIDSolve=expIDSolve)

    if priorInfo.hasPriors == true && compute_hessian == false
        θ_estT = transformθ(θ_est, θ_indices.θ_names, θ_indices)
        cost -= computePriors(θ_est, θ_estT, θ_indices.θ_names, priorInfo) # We work with -loglik
    end

    return cost
end


function compute_costSolveODE(θ_dynamic::AbstractVector,
                             θ_sd::AbstractVector,
                             θ_observable::AbstractVector,
                             θ_nonDynamic::AbstractVector,
                             odeProblem::ODEProblem,
                             ode_solver::ODESolver,
                             ss_solver::SteadyStateSolver,
                             petab_model::PEtabModel,
                             simulation_info::SimulationInfo,
                             θ_indices::ParameterIndices,
                             measurementInfo::MeasurementsInfo,
                             parameterInfo::ParametersInfo,
                             petabODECache::PEtabODEProblemCache,
                             petabODESolverCache::PEtabODESolverCache;
                             compute_cost::Bool=false,
                             compute_hessian::Bool=false,
                             compute_gradientDynamicθ::Bool=false,
                             compute_residuals::Bool=false,
                             expIDSolve::Vector{Symbol} = [:all])::Real

    if compute_gradientDynamicθ == true && petabODECache.nθ_dynamicEst[1] != length(θ_dynamic)
        _θ_dynamic = θ_dynamic[petabODECache.θ_dynamicOutputOrder]
        θ_dynamicT = transformθ(_θ_dynamic, θ_indices.θ_dynamicNames, θ_indices, :θ_dynamic, petabODECache)
    else
        θ_dynamicT = transformθ(θ_dynamic, θ_indices.θ_dynamicNames, θ_indices, :θ_dynamic, petabODECache)
    end

    θ_sdT = transformθ(θ_sd, θ_indices.θ_sdNames, θ_indices, :θ_sd, petabODECache)
    θ_observableT = transformθ(θ_observable, θ_indices.θ_observableNames, θ_indices, :θ_observable, petabODECache)
    θ_nonDynamicT = transformθ(θ_nonDynamic, θ_indices.θ_nonDynamicNames, θ_indices, :θ_nonDynamic, petabODECache)

    _odeProblem = remake(odeProblem, p = convert.(eltype(θ_dynamicT), odeProblem.p), u0 = convert.(eltype(θ_dynamicT), odeProblem.u0))
    changeODEProblemParameters!(_odeProblem.p, _odeProblem.u0, θ_dynamicT, θ_indices, petab_model)

    # If computing hessian or gradient store ODE solution in arrary with dual numbers, else use
    # solution array with floats
    if compute_hessian == true || compute_gradientDynamicθ == true
        success = solveODEAllExperimentalConditions!(simulation_info.odeSolutionsDerivatives, _odeProblem, petab_model, θ_dynamicT, petabODESolverCache, simulation_info, θ_indices, ode_solver, ss_solver, expIDSolve=expIDSolve, denseSolution=false, onlySaveAtObservedTimes=true)
    elseif compute_cost == true
        success = solveODEAllExperimentalConditions!(simulation_info.odeSolutions, _odeProblem, petab_model, θ_dynamicT, petabODESolverCache, simulation_info, θ_indices, ode_solver, ss_solver, expIDSolve=expIDSolve, denseSolution=false, onlySaveAtObservedTimes=true)
    end
    if success != true
        @warn "Failed to solve ODE model"
        return Inf
    end

    cost = _compute_cost(θ_sdT, θ_observableT, θ_nonDynamicT, petab_model, simulation_info, θ_indices, measurementInfo,
                        parameterInfo, expIDSolve,
                        compute_hessian=compute_hessian,
                        compute_gradientDynamicθ=compute_gradientDynamicθ,
                        compute_residuals=compute_residuals)

    return cost
end


function compute_costNotSolveODE(θ_sd::AbstractVector,
                                θ_observable::AbstractVector,
                                θ_nonDynamic::AbstractVector,
                                petab_model::PEtabModel,
                                simulation_info::SimulationInfo,
                                θ_indices::ParameterIndices,
                                measurementInfo::MeasurementsInfo,
                                parameterInfo::ParametersInfo,
                                petabODECache::PEtabODEProblemCache;
                                compute_gradientNotSolveAutoDiff::Bool=false,
                                compute_gradientNotSolveAdjoint::Bool=false,
                                compute_gradientNotSolveForward::Bool=false,
                                expIDSolve::Vector{Symbol} = [:all])::Real

    # To be able to use ReverseDiff sdParamEstUse and obsParamEstUse cannot be overwritten.
    # Hence new vectors have to be created. Minimal overhead.
    θ_sdT = transformθ(θ_sd, θ_indices.θ_sdNames, θ_indices, :θ_sd, petabODECache)
    θ_observableT = transformθ(θ_observable, θ_indices.θ_observableNames, θ_indices, :θ_observable, petabODECache)
    θ_nonDynamicT = transformθ(θ_nonDynamic, θ_indices.θ_nonDynamicNames, θ_indices, :θ_nonDynamic, petabODECache)

    cost = _compute_cost(θ_sdT, θ_observableT, θ_nonDynamicT, petab_model, simulation_info, θ_indices,
                        measurementInfo, parameterInfo, expIDSolve,
                        compute_gradientNotSolveAutoDiff=compute_gradientNotSolveAutoDiff,
                        compute_gradientNotSolveAdjoint=compute_gradientNotSolveAdjoint,
                        compute_gradientNotSolveForward=compute_gradientNotSolveForward)

    return cost
end


function _compute_cost(θ_sd::AbstractVector,
                      θ_observable::AbstractVector,
                      θ_nonDynamic::AbstractVector,
                      petab_model::PEtabModel,
                      simulation_info::SimulationInfo,
                      θ_indices::ParameterIndices,
                      measurementInfo::MeasurementsInfo,
                      parameterInfo::ParametersInfo,
                      expIDSolve::Vector{Symbol} = [:all];
                      compute_hessian::Bool=false,
                      compute_gradientDynamicθ::Bool=false,
                      compute_residuals::Bool=false,
                      compute_gradientNotSolveAdjoint::Bool=false,
                      compute_gradientNotSolveForward::Bool=false,
                      compute_gradientNotSolveAutoDiff::Bool=false)::Real

    if compute_hessian == true || compute_gradientDynamicθ == true || compute_gradientNotSolveAdjoint == true || compute_gradientNotSolveForward == true || compute_gradientNotSolveAutoDiff == true
        odeSolutions = simulation_info.odeSolutionsDerivatives
    else
        odeSolutions = simulation_info.odeSolutions
    end

    cost = 0.0
    for experimentalConditionId in simulation_info.experimentalConditionId

        if expIDSolve[1] != :all && experimentalConditionId ∉ expIDSolve
            continue
        end

        # Extract the ODE-solution for specific condition ID
        odeSolution = odeSolutions[experimentalConditionId]
        cost += compute_costExpCond(odeSolution, Float64[], θ_sd, θ_observable, θ_nonDynamic, petab_model,
                                   experimentalConditionId, θ_indices, measurementInfo, parameterInfo, simulation_info,
                                   compute_residuals=compute_residuals,
                                   compute_gradientNotSolveAdjoint=compute_gradientNotSolveAdjoint,
                                   compute_gradientNotSolveForward=compute_gradientNotSolveForward,
                                   compute_gradientNotSolveAutoDiff=compute_gradientNotSolveAutoDiff)

        if isinf(cost)
            return Inf
        end
    end

    return cost
end


function compute_costExpCond(odeSolution::ODESolution,
                            pODEProblemZygote::AbstractVector,
                            θ_sd::AbstractVector,
                            θ_observable::AbstractVector,
                            θ_nonDynamic::AbstractVector,
                            petab_model::PEtabModel,
                            experimentalConditionId::Symbol,
                            θ_indices::ParameterIndices,
                            measurementInfo::MeasurementsInfo,
                            parameterInfo::ParametersInfo,
                            simulation_info::SimulationInfo;
                            compute_residuals::Bool=false,
                            compute_gradientNotSolveAdjoint::Bool=false,
                            compute_gradientNotSolveForward::Bool=false,
                            compute_gradientNotSolveAutoDiff::Bool=false,
                            compute_gradientθDynamicZygote::Bool=false)::Real

    if !(odeSolution.retcode == ReturnCode.Success || odeSolution.retcode == ReturnCode.Terminated)
        return Inf
    end

    cost = 0.0
    for iMeasurement in simulation_info.iMeasurements[experimentalConditionId]

        t = measurementInfo.time[iMeasurement]

        # In these cases we only save the ODE at observed time-points and we do not want
        # to extract Dual ODE solution
        if compute_gradientNotSolveForward == true || compute_gradientNotSolveAutoDiff == true
            nModelStates = length(petab_model.state_names)
            u = dualToFloat.(odeSolution[1:nModelStates, simulation_info.iTimeODESolution[iMeasurement]])
            p = dualToFloat.(odeSolution.prob.p)
        # For adjoint sensitivity analysis we have a dense-ode solution
        elseif compute_gradientNotSolveAdjoint == true
            # In case we only have sol.t = 0.0 (or similar) interpolation does not work
            u = length(odeSolution.t) > 1 ? odeSolution(t) : odeSolution[1]
            p = odeSolution.prob.p

        elseif compute_gradientθDynamicZygote == true
            u = odeSolution.u[simulation_info.iTimeODESolution[iMeasurement], :][1]
            p = pODEProblemZygote

        # When we want to extract dual number from the ODE solution
        else
            u = odeSolution[:, simulation_info.iTimeODESolution[iMeasurement]]
            p = odeSolution.prob.p
        end

        h = computeh(u, t, p, θ_observable, θ_nonDynamic, petab_model, iMeasurement, measurementInfo, θ_indices, parameterInfo)
        hTransformed = transformMeasurementOrH(h, measurementInfo.measurementTransformation[iMeasurement])
        σ = computeσ(u, t, p, θ_sd, θ_nonDynamic, petab_model, iMeasurement, measurementInfo, θ_indices, parameterInfo)
        residual = (hTransformed - measurementInfo.measurementT[iMeasurement]) / σ

        # These values might be needed by different software, e.g. PyPesto, to assess things such as parameter uncertainity. By storing them in
        # measurementInfo they can easily be computed given a call to the cost function has been made.
        updateMeasurementInfo!(measurementInfo, h, hTransformed, σ, residual, iMeasurement)

        # By default a positive ODE solution is not enforced (even though the user can provide it as option).
        # In case with transformations on the data the code can crash, hence Inf is returned in case the
        # model data transformation can not be perfomred.
        if isinf(hTransformed)
            println("Warning - transformed observable is non-finite for measurement $iMeasurement")
            return Inf
        end

        # Update log-likelihood. In case of guass newton approximation we are only interested in the residuals, and here
        # we allow the residuals to be computed to test the gauss-newton implementation
        if compute_residuals == false
            if measurementInfo.measurementTransformation[iMeasurement] === :lin
                cost += log(σ) + 0.5*log(2*pi) + 0.5*residual^2
            elseif measurementInfo.measurementTransformation[iMeasurement] === :log10
                cost += log(σ) + 0.5*log(2*pi) + log(log(10)) + log(10)*measurementInfo.measurementT[iMeasurement] + 0.5*residual^2
            elseif measurementInfo.measurementTransformation[iMeasurement] === :log
                cost += log(σ) + 0.5*log(2*pi) + log(measurementInfo.measurement[iMeasurement]) + 0.5*residual^2
            else
                println("Transformation ", measurementInfo.measurementTransformation[iMeasurement], " not yet supported.")
                return Inf
            end
        elseif compute_residuals == true
            cost += residual
        end
    end
    return cost
end


function updateMeasurementInfo!(measurementInfo::MeasurementsInfo, h::T, hTransformed::T, σ::T, residual::T, iMeasurement) where {T<:AbstractFloat}
    ChainRulesCore.@ignore_derivatives begin
        measurementInfo.simulatedValues[iMeasurement] = h
        measurementInfo.chi2Values[iMeasurement] = (hTransformed - measurementInfo.measurementT[iMeasurement])^2 / σ^2
        measurementInfo.residuals[iMeasurement] = residual
    end
end
function updateMeasurementInfo!(measurementInfo::MeasurementsInfo, h, hTransformed, σ, residual, iMeasurement)
    return
end