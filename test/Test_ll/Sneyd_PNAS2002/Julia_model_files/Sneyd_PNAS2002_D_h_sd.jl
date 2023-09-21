#u[1] = IPR_S, u[2] = IPR_I2, u[3] = IPR_R, u[4] = IPR_O, u[5] = IPR_I1, u[6] = IPR_A
#p_ode_problem[1] = l_4, p_ode_problem[2] = k_4, p_ode_problem[3] = IP3, p_ode_problem[4] = k4, p_ode_problem[5] = k_2, p_ode_problem[6] = l2, p_ode_problem[7] = l_2, p_ode_problem[8] = default, p_ode_problem[9] = l_6, p_ode_problem[10] = k1, p_ode_problem[11] = k_3, p_ode_problem[12] = l6, p_ode_problem[13] = membrane, p_ode_problem[14] = k3, p_ode_problem[15] = l4, p_ode_problem[16] = Ca, p_ode_problem[17] = k2, p_ode_problem[18] = k_1
#
function compute_∂h∂u!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                       θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :open_probability 
		out[4] = 0.4((0.1u[4] + 0.9u[6])^3.0)
		out[6] = 3.6((0.1u[4] + 0.9u[6])^3.0)
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                       θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :open_probability 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                        parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :open_probability 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                        parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :open_probability 
		return nothing
	end

end

