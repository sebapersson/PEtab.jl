#=
    Functions for computing forward-sensitivities with SciMLSensitivity
=#

function PEtab._get_odeproblem_gradient(ode_problem::ODEProblem, gradient_method::Symbol,
                                        sensealg_forward_equations::SciMLSensitivity.AbstractForwardSensitivityAlgorithm)::ODEProblem
    return ODEForwardSensitivityProblem(ode_problem.f, ode_problem.u0, ode_problem.tspan,
                                        ode_problem.p,
                                        sensealg = sensealg_forward_equations)
end

function PEtab.solve_sensitivites(ode_problem::ODEProblem,
                                  simulation_info::PEtab.SimulationInfo,
                                  θ_indices::PEtab.ParameterIndices,
                                  petab_model::PEtabModel,
                                  sensealg::SciMLSensitivity.AbstractForwardSensitivityAlgorithm,
                                  θ_dynamic::AbstractVector,
                                  _solve_ode_all_conditions!::Function,
                                  cfg::Nothing,
                                  petab_ODE_cache::PEtab.PEtabODEProblemCache,
                                  exp_id_solve::Vector{Symbol},
                                  split_over_conditions::Bool,
                                  isremade::Bool = false)::Bool
    success = _solve_ode_all_conditions!(θ_dynamic, exp_id_solve)
    return success
end

function PEtab.compute_gradient_forward_equations_condition!(gradient::Vector{Float64},
                                                             sol::ODESolution,
                                                             petab_ODE_cache::PEtab.PEtabODEProblemCache,
                                                             sensealg::SciMLSensitivity.AbstractForwardSensitivityAlgorithm,
                                                             θ_dynamic::Vector{Float64},
                                                             θ_sd::Vector{Float64},
                                                             θ_observable::Vector{Float64},
                                                             θ_non_dynamic::Vector{Float64},
                                                             experimental_condition_id::Symbol,
                                                             simulation_condition_id::Symbol,
                                                             simulation_info::PEtab.SimulationInfo,
                                                             petab_model::PEtabModel,
                                                             θ_indices::PEtab.ParameterIndices,
                                                             measurement_info::PEtab.MeasurementsInfo,
                                                             parameter_info::PEtab.ParametersInfo)::Nothing
    imeasurements_t = simulation_info.imeasurements_t[experimental_condition_id]
    time_observed = simulation_info.tsaves[experimental_condition_id]

    # To compute
    compute∂G∂u = (out, u, p, t, i) -> begin
        PEtab.compute∂G∂_(out, u, p, t, i, imeasurements_t,
                          measurement_info, parameter_info,
                          θ_indices, petab_model,
                          θ_sd, θ_observable, θ_non_dynamic,
                          petab_ODE_cache.∂h∂u, petab_ODE_cache.∂σ∂u, compute∂G∂U = true)
    end
    compute∂G∂p = (out, u, p, t, i) -> begin
        PEtab.compute∂G∂_(out, u, p, t, i, imeasurements_t,
                          measurement_info, parameter_info,
                          θ_indices, petab_model,
                          θ_sd, θ_observable, θ_non_dynamic,
                          petab_ODE_cache.∂h∂p, petab_ODE_cache.∂σ∂p, compute∂G∂U = false)
    end

    # Loop through solution and extract sensitivites
    p = sol.prob.p
    ∂G∂p, ∂G∂p_ = zeros(Float64, length(p)), zeros(Float64, length(p))
    ∂G∂u = zeros(Float64, length(states(petab_model.sys_mutated)))
    _gradient = zeros(Float64, length(p))
    for i in eachindex(time_observed)
        u, _S = extract_local_sensitivities(sol, i, true)
        compute∂G∂u(∂G∂u, u, p, time_observed[i], i)
        compute∂G∂p(∂G∂p_, u, p, time_observed[i], i)
        _gradient .+= transpose(_S) * ∂G∂u
        ∂G∂p .+= ∂G∂p_
    end

    # Thus far have have computed dY/dθ, but for parameters on the log-scale we want dY/dθ_log. We can adjust via;
    # dY/dθ_log = log(10) * θ * dY/dθ
    PEtab.adjust_gradient_θ_transformed!(gradient, _gradient, ∂G∂p, θ_dynamic, θ_indices,
                                         simulation_condition_id,
                                         autodiff_sensitivites = false)

    return nothing
end
