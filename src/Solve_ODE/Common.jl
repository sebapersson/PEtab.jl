#=
    Helper function for correctly solving (simulating) the ODE model
=#

function get_t_saveat(::Val{save_at_observed_t},
                      simulation_info::SimulationInfo,
                      condition_id::Symbol,
                      tmax::Float64,
                      n_timepoints_save::Int64)::Vector{Float64} where {save_at_observed_t}

    # Check if we want to only save the ODE at specific time-points
    if save_at_observed_t == true
        return simulation_info.tsaves[condition_id]
    elseif n_timepoints_save > 0
        return collect(LinRange(0.0, tmax, n_timepoints_save))
    else
        return Float64[]
    end
end

function should_save_dense_sol(::Val{save_at_observed_t},
                               n_timepoints_save::Int64,
                               dense_sol::Bool)::Bool where {save_at_observed_t}

    # Check if we want to only save the ODE at specific time-points
    if save_at_observed_t == true
        return false
    elseif n_timepoints_save > 0
        return false
    else
        return dense_sol
    end
end

function get_callbackset(ode_problem::ODEProblem,
                         simulation_info::SimulationInfo,
                         condition_id::Symbol,
                         sensealg)::SciMLBase.DECallback
    return simulation_info.callbacks[condition_id]
end

function get_tspan(ode_problem::ODEProblem,
                   tmax::Float64,
                   solver::SciMLAlgorithm,
                   convert_tspan::Bool)::ODEProblem

    # When tmax=Inf and a multistep BDF Julia method, e.g. QNDF, is used tmax must be inf, else if it is a large
    # number such as 1e8 the dt_min is set to a large value making the solver fail. Sundials solvers on the other
    # hand are not compatible with timespan = (0.0, Inf), hence for these we use timespan = (0.0, 1e8)
    tmax::Float64 = _get_tmax(tmax, solver)
    if convert_tspan == false
        return remake(ode_problem, tspan = (0.0, tmax))
    else
        return remake(ode_problem, tspan = convert.(eltype(ode_problem.p), (0.0, tmax)))
    end
end

function _get_tmax(tmax::Float64, solver::Union{CVODE_BDF, CVODE_Adams})::Float64
    return isinf(tmax) ? 1e8 : tmax
end
function _get_tmax(tmax::Float64, solver::Union{Vector{Symbol}, SciMLAlgorithm})::Float64
    return tmax
end

# Each soluation needs to have a unique vector associated with it such that the gradient
# is correct computed for non-dynamic parameters (condition specific parameters are mapped
# correctly) as these computations employ ode_problem.p
function set_ode_parameters(_ode_problem::ODEProblem,
                            petab_ODESolver_cache::PEtabODESolverCache,
                            condition_id::Symbol)::ODEProblem

    # Ensure parameters constant between conditions are set correctly
    p_ode_problem = get_tmp(petab_ODESolver_cache.p_ode_problem_cache[condition_id],
                            _ode_problem.p)
    u0 = get_tmp(petab_ODESolver_cache.u0_cache[condition_id], _ode_problem.p)
    p_ode_problem .= _ode_problem.p
    @views u0 .= _ode_problem.u0[1:length(u0)]

    return remake(_ode_problem, p = p_ode_problem, u0 = u0)
end
