#u[1] = x1, u[2] = observable_x2, u[3] = sigma_x2, u[4] = x2
#p_ode_problem[1] = default, p_ode_problem[2] = k3, p_ode_problem[3] = k1, p_ode_problem[4] = k2
#
function compute_∂h∂u!(u, t::Real, p_ode_problem::AbstractVector,
                       θ_observable::AbstractVector,
                       θ_non_dynamic::AbstractVector, observableId::Symbol,
                       parameter_map::θObsOrSdParameterMap, out)
    if observableId == :obs_x2
        out[4] = 1
        return nothing
    end
end

function compute_∂h∂p!(u, t::Real, p_ode_problem::AbstractVector,
                       θ_observable::AbstractVector,
                       θ_non_dynamic::AbstractVector, observableId::Symbol,
                       parameter_map::θObsOrSdParameterMap, out)
    if observableId == :obs_x2
        return nothing
    end
end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,
                        θ_non_dynamic::AbstractVector,
                        parameter_info::ParametersInfo, observableId::Symbol,
                        parameter_map::θObsOrSdParameterMap, out)
    if observableId == :obs_x2
        return nothing
    end
end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,
                        θ_non_dynamic::AbstractVector,
                        parameter_info::ParametersInfo, observableId::Symbol,
                        parameter_map::θObsOrSdParameterMap, out)
    if observableId == :obs_x2
        return nothing
    end
end
