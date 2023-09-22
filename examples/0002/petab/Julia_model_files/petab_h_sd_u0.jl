#u[1] = x1, u[2] = observable_x2, u[3] = sigma_x2, u[4] = x2
#p_ode_problem_names[1] = default, p_ode_problem_names[2] = k3, p_ode_problem_names[3] = k1, p_ode_problem_names[4] = k2
#

function compute_h(u::AbstractVector, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                   θ_non_dynamic::AbstractVector, parameter_info::ParametersInfo, observableId::Symbol,
                      parameter_map::θObsOrSdParameterMap)::Real 
	if observableId === :obs_x2 
		return u[4] 
	end

end

function compute_u0!(u0::AbstractVector, p_ode_problem::AbstractVector) 

	#p_ode_problem[1] = default, p_ode_problem[2] = k3, p_ode_problem[3] = k1, p_ode_problem[4] = k2

	t = 0.0 # u at time zero

	x1 = 0.0 
	observable_x2 = 0.0 
	sigma_x2 = 0.04 
	x2 = 0.0 

	u0 .= x1, observable_x2, sigma_x2, x2
end

function compute_u0(p_ode_problem::AbstractVector)::AbstractVector 

	#p_ode_problem[1] = default, p_ode_problem[2] = k3, p_ode_problem[3] = k1, p_ode_problem[4] = k2

	t = 0.0 # u at time zero

	x1 = 0.0 
	observable_x2 = 0.0 
	sigma_x2 = 0.04 
	x2 = 0.0 

	 return [x1, observable_x2, sigma_x2, x2]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                   parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap)::Real 
	if observableId === :obs_x2 
		noiseParameter1_obs_x2 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_obs_x2 
	end


end

