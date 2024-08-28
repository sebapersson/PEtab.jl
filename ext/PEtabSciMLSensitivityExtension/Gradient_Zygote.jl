# Compute gradient using Zygote
function compute_gradient_zygote!(gradient::Vector{Float64},
                                  θ_est::Vector{Float64},
                                  ode_problem::ODEProblem,
                                  petab_model::PEtabModel,
                                  simulation_info::PEtab.SimulationInfo,
                                  θ_indices::PEtab.ParameterIndices,
                                  measurement_info::PEtab.MeasurementsInfo,
                                  parameter_info::PEtab.ParametersInfo,
                                  solve_ode_condition::Function,
                                  prior_info::PEtab.PriorInfo,
                                  cache::PEtab.PEtabODEProblemCache)

    # Split input into observeble and dynamic parameters
    xdynamic, xobservable, xnoise, xnondynamic = PEtab.splitθ(θ_est, θ_indices)

    # For Zygote the code must be out-of place. Hence a special likelihood funciton is needed.
    compute_gradient_zygote_xdynamic! = (x) -> _compute_cost_zygote(x, xnoise, xobservable,
                                                                     xnondynamic,
                                                                     ode_problem,
                                                                     petab_model,
                                                                     simulation_info,
                                                                     θ_indices,
                                                                     measurement_info,
                                                                     parameter_info,
                                                                     solve_ode_condition)
    gradient[θ_indices.xindices[:dynamic]] .= Zygote.gradient(compute_gradient_zygote_xdynamic!,
                                                              xdynamic)[1]

    # Compute gradient for parameters which are not in ODE-system. Important to keep in mind that Sd- and observable
    # parameters can overlap in θ_est.
    ixnoise, ixobservable, ixnondynamic, iθ_not_ode = PEtab.get_index_parameters_not_ODE(θ_indices)
    compute_cost_θ_not_ODE = (x) -> PEtab.cost_not_solve_ODE(x[ixnoise],
                                                                     x[ixobservable],
                                                                     x[ixnondynamic],
                                                                     petab_model,
                                                                     simulation_info,
                                                                     θ_indices,
                                                                     measurement_info,
                                                                     parameter_info,
                                                                     cache)
    @views ReverseDiff.gradient!(gradient[iθ_not_ode], compute_cost_θ_not_ODE,
                                 θ_est[iθ_not_ode])
end
