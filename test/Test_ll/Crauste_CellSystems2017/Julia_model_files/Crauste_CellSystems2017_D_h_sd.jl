#u[1] = Naive, u[2] = Pathogen, u[3] = LateEffector, u[4] = EarlyEffector, u[5] = Memory
#p_ode_problem[1] = mu_LL, p_ode_problem[2] = delta_NE, p_ode_problem[3] = mu_PL, p_ode_problem[4] = mu_P, p_ode_problem[5] = delta_EL, p_ode_problem[6] = mu_PE, p_ode_problem[7] = mu_EE, p_ode_problem[8] = default, p_ode_problem[9] = mu_N, p_ode_problem[10] = rho_E, p_ode_problem[11] = delta_LM, p_ode_problem[12] = rho_P, p_ode_problem[13] = mu_LE
#
function compute_∂h∂u!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                       θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :observable_EarlyEffector 
		out[4] = 1
		return nothing
	end

	if observableId == :observable_LateEffector 
		out[3] = 1
		return nothing
	end

	if observableId == :observable_Memory 
		out[5] = 1
		return nothing
	end

	if observableId == :observable_Naive 
		out[1] = 1
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                       θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :observable_EarlyEffector 
		return nothing
	end

	if observableId == :observable_LateEffector 
		return nothing
	end

	if observableId == :observable_Memory 
		return nothing
	end

	if observableId == :observable_Naive 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                        parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :observable_EarlyEffector 
		return nothing
	end

	if observableId == :observable_LateEffector 
		return nothing
	end

	if observableId == :observable_Memory 
		return nothing
	end

	if observableId == :observable_Naive 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                        parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :observable_EarlyEffector 
		return nothing
	end

	if observableId == :observable_LateEffector 
		return nothing
	end

	if observableId == :observable_Memory 
		return nothing
	end

	if observableId == :observable_Naive 
		return nothing
	end

end

