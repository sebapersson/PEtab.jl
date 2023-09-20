function PEtab.setUpGradient(whichMethod::Symbol,
                             odeProblem::ODEProblem,
                             ode_solver::ODESolver,
                             ss_solver::SteadyStateSolver,
                             petabODECache::PEtab.PEtabODEProblemCache,
                             petabODESolverCache::PEtab.PEtabODESolverCache,
                             petab_model::PEtabModel,
                             simulation_info::PEtab.SimulationInfo,
                             θ_indices::PEtab.ParameterIndices,
                             measurementInfo::PEtab.MeasurementsInfo,
                             parameterInfo::PEtab.ParametersInfo,
                             sensealg::Union{InterpolatingAdjoint, QuadratureAdjoint},
                             priorInfo::PEtab.PriorInfo;
                             chunksize::Union{Nothing, Int64}=nothing,
                             sensealg_ss=nothing,
                             n_processes::Int64=1,
                             jobs=nothing,
                             results=nothing,
                             split_over_conditions::Bool=false)

    _sensealg_ss = isnothing(sensealg_ss) ? InterpolatingAdjoint(autojacvec=ReverseDiffVJP()) : sensealg_ss
    # Fast but numerically unstable method
    if simulation_info.haspreEquilibrationConditionId == true && typeof(_sensealg_ss) <: SteadyStateAdjoint
        @warn "If using adjoint sensitivity analysis for a model with PreEq-criteria the most the most efficient sensealg_ss is as provided SteadyStateAdjoint. However, SteadyStateAdjoint fails if the Jacobian is singular hence we recomend you check that the Jacobian is non-singular."
    end

    iθ_sd, iθ_observable, iθ_nonDynamic, iθ_notOdeSystem = PEtab.getIndicesParametersNotInODESystem(θ_indices)
    compute_costNotODESystemθ = (x) -> PEtab.compute_costNotSolveODE(x[iθ_sd], x[iθ_observable], x[iθ_nonDynamic],
        petab_model, simulation_info, θ_indices, measurementInfo, parameterInfo, petabODECache, expIDSolve=[:all],
        compute_gradientNotSolveAdjoint=true)

    _compute_gradient! = (gradient, θ_est) -> compute_gradientAdjointEquations!(gradient,
                                                                              θ_est,
                                                                              ode_solver,
                                                                              ss_solver,
                                                                              compute_costNotODESystemθ,
                                                                              sensealg,
                                                                              _sensealg_ss,
                                                                              odeProblem,
                                                                              petab_model,
                                                                              simulation_info,
                                                                              θ_indices,
                                                                              measurementInfo,
                                                                              parameterInfo,
                                                                              priorInfo,
                                                                              petabODECache,
                                                                              petabODESolverCache,
                                                                              expIDSolve=[:all])
    
    return _compute_gradient!
end


function PEtab.setSensealg(sensealg, ::Val{:Adjoint})

    if !isnothing(sensealg)
        @assert any(typeof(sensealg) .<: [InterpolatingAdjoint, QuadratureAdjoint]) "For gradient method :Adjoint allowed sensealg args are InterpolatingAdjoint, QuadratureAdjoint not $sensealg"
        return sensealg
    end

    return InterpolatingAdjoint(autojacvec=ReverseDiffVJP())
end
function PEtab.setSensealg(sensealg::Union{ForwardSensitivity, ForwardDiffSensitivity}, ::Val{:ForwardEquations})
    return sensealg
end


function PEtab.getCallbackSet(odeProblem::ODEProblem,
                              simulation_info::PEtab.SimulationInfo,
                              simulationConditionId::Symbol,
                              sensealg::Union{InterpolatingAdjoint, QuadratureAdjoint})::SciMLBase.DECallback

    cbSet = SciMLSensitivity.track_callbacks(simulation_info.callbacks[simulationConditionId], odeProblem.tspan[1],
                                                 odeProblem.u0, odeProblem.p, sensealg)
    simulation_info.trackedCallbacks[simulationConditionId] = cbSet
    return cbSet
end