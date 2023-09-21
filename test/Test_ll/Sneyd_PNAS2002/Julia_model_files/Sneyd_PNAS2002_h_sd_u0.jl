#u[1] = IPR_S, u[2] = IPR_I2, u[3] = IPR_R, u[4] = IPR_O, u[5] = IPR_I1, u[6] = IPR_A
#p_ode_problem_names[1] = l_4, p_ode_problem_names[2] = k_4, p_ode_problem_names[3] = IP3, p_ode_problem_names[4] = k4, p_ode_problem_names[5] = k_2, p_ode_problem_names[6] = l2, p_ode_problem_names[7] = l_2, p_ode_problem_names[8] = default, p_ode_problem_names[9] = l_6, p_ode_problem_names[10] = k1, p_ode_problem_names[11] = k_3, p_ode_problem_names[12] = l6, p_ode_problem_names[13] = membrane, p_ode_problem_names[14] = k3, p_ode_problem_names[15] = l4, p_ode_problem_names[16] = Ca, p_ode_problem_names[17] = k2, p_ode_problem_names[18] = k_1
#

function compute_h(u::AbstractVector, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                   θ_non_dynamic::AbstractVector, parameter_info::ParametersInfo, observableId::Symbol,
                      parameter_map::θObsOrSdParameterMap)::Real 
	if observableId === :open_probability 
		return ( 0.9 * u[6] + 0.1 * u[4] ) ^ 4.0 
	end

end

function compute_u0!(u0::AbstractVector, p_ode_problem::AbstractVector) 

	#p_ode_problem[1] = l_4, p_ode_problem[2] = k_4, p_ode_problem[3] = IP3, p_ode_problem[4] = k4, p_ode_problem[5] = k_2, p_ode_problem[6] = l2, p_ode_problem[7] = l_2, p_ode_problem[8] = default, p_ode_problem[9] = l_6, p_ode_problem[10] = k1, p_ode_problem[11] = k_3, p_ode_problem[12] = l6, p_ode_problem[13] = membrane, p_ode_problem[14] = k3, p_ode_problem[15] = l4, p_ode_problem[16] = Ca, p_ode_problem[17] = k2, p_ode_problem[18] = k_1

	t = 0.0 # u at time zero

	IPR_S = 0.0 
	IPR_I2 = 0.0 
	IPR_R = 1.0 
	IPR_O = 0.0 
	IPR_I1 = 0.0 
	IPR_A = 0.0 

	u0 .= IPR_S, IPR_I2, IPR_R, IPR_O, IPR_I1, IPR_A
end

function compute_u0(p_ode_problem::AbstractVector)::AbstractVector 

	#p_ode_problem[1] = l_4, p_ode_problem[2] = k_4, p_ode_problem[3] = IP3, p_ode_problem[4] = k4, p_ode_problem[5] = k_2, p_ode_problem[6] = l2, p_ode_problem[7] = l_2, p_ode_problem[8] = default, p_ode_problem[9] = l_6, p_ode_problem[10] = k1, p_ode_problem[11] = k_3, p_ode_problem[12] = l6, p_ode_problem[13] = membrane, p_ode_problem[14] = k3, p_ode_problem[15] = l4, p_ode_problem[16] = Ca, p_ode_problem[17] = k2, p_ode_problem[18] = k_1

	t = 0.0 # u at time zero

	IPR_S = 0.0 
	IPR_I2 = 0.0 
	IPR_R = 1.0 
	IPR_O = 0.0 
	IPR_I1 = 0.0 
	IPR_A = 0.0 

	 return [IPR_S, IPR_I2, IPR_R, IPR_O, IPR_I1, IPR_A]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                   parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap)::Real 
	if observableId === :open_probability 
		noiseParameter1_open_probability = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_open_probability 
	end


end

