#u[1] = Naive, u[2] = Pathogen, u[3] = LateEffector, u[4] = EarlyEffector, u[5] = Memory
#p_ode_problem_names[1] = mu_LL, p_ode_problem_names[2] = delta_NE, p_ode_problem_names[3] = mu_PL, p_ode_problem_names[4] = mu_P, p_ode_problem_names[5] = delta_EL, p_ode_problem_names[6] = mu_PE, p_ode_problem_names[7] = mu_EE, p_ode_problem_names[8] = default, p_ode_problem_names[9] = mu_N, p_ode_problem_names[10] = rho_E, p_ode_problem_names[11] = delta_LM, p_ode_problem_names[12] = rho_P, p_ode_problem_names[13] = mu_LE
#

function compute_h(u::AbstractVector, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                   θ_non_dynamic::AbstractVector, parameter_info::ParametersInfo, observableId::Symbol,
                      parameter_map::θObsOrSdParameterMap)::Real 
	if observableId === :observable_EarlyEffector 
		return u[4] 
	end

	if observableId === :observable_LateEffector 
		return u[3] 
	end

	if observableId === :observable_Memory 
		return u[5] 
	end

	if observableId === :observable_Naive 
		return u[1] 
	end

end

function compute_u0!(u0::AbstractVector, p_ode_problem::AbstractVector) 

	#p_ode_problem[1] = mu_LL, p_ode_problem[2] = delta_NE, p_ode_problem[3] = mu_PL, p_ode_problem[4] = mu_P, p_ode_problem[5] = delta_EL, p_ode_problem[6] = mu_PE, p_ode_problem[7] = mu_EE, p_ode_problem[8] = default, p_ode_problem[9] = mu_N, p_ode_problem[10] = rho_E, p_ode_problem[11] = delta_LM, p_ode_problem[12] = rho_P, p_ode_problem[13] = mu_LE

	t = 0.0 # u at time zero

	Naive = 8090.0 
	Pathogen = 1.0 
	LateEffector = 0.0 
	EarlyEffector = 0.0 
	Memory = 0.0 

	u0 .= Naive, Pathogen, LateEffector, EarlyEffector, Memory
end

function compute_u0(p_ode_problem::AbstractVector)::AbstractVector 

	#p_ode_problem[1] = mu_LL, p_ode_problem[2] = delta_NE, p_ode_problem[3] = mu_PL, p_ode_problem[4] = mu_P, p_ode_problem[5] = delta_EL, p_ode_problem[6] = mu_PE, p_ode_problem[7] = mu_EE, p_ode_problem[8] = default, p_ode_problem[9] = mu_N, p_ode_problem[10] = rho_E, p_ode_problem[11] = delta_LM, p_ode_problem[12] = rho_P, p_ode_problem[13] = mu_LE

	t = 0.0 # u at time zero

	Naive = 8090.0 
	Pathogen = 1.0 
	LateEffector = 0.0 
	EarlyEffector = 0.0 
	Memory = 0.0 

	 return [Naive, Pathogen, LateEffector, EarlyEffector, Memory]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                   parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap)::Real 
	if observableId === :observable_EarlyEffector 
		noiseParameter1_observable_EarlyEffector = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_EarlyEffector 
	end

	if observableId === :observable_LateEffector 
		noiseParameter1_observable_LateEffector = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_LateEffector 
	end

	if observableId === :observable_Memory 
		noiseParameter1_observable_Memory = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_Memory 
	end

	if observableId === :observable_Naive 
		noiseParameter1_observable_Naive = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_Naive 
	end


end

