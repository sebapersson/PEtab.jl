#u[1] = IRp, u[2] = IR, u[3] = IRins, u[4] = IRiP, u[5] = IRS, u[6] = X, u[7] = IRi, u[8] = IRSiP, u[9] = Xp
#p_ode_problem[1] = k1c, p_ode_problem[2] = k21, p_ode_problem[3] = insulin_bool1, p_ode_problem[4] = k1g, p_ode_problem[5] = insulin_dose_2, p_ode_problem[6] = k1a, p_ode_problem[7] = insulin_dose_1, p_ode_problem[8] = k1aBasic, p_ode_problem[9] = k1d, p_ode_problem[10] = insulin_time_1, p_ode_problem[11] = insulin_time_2, p_ode_problem[12] = cyt, p_ode_problem[13] = k22, p_ode_problem[14] = insulin_bool2, p_ode_problem[15] = default, p_ode_problem[16] = k1r, p_ode_problem[17] = k1f, p_ode_problem[18] = k1b, p_ode_problem[19] = k3, p_ode_problem[20] = km2, p_ode_problem[21] = k1e, p_ode_problem[22] = k_IRSiP_DosR, p_ode_problem[23] = km3
#
function compute_∂h∂u!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,                    
                       θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :IR1_P 
		observableParameter1_IR1_P = get_obs_sd_parameter(θ_observable, parameter_map)
		out[1] = observableParameter1_IR1_P
		out[4] = observableParameter1_IR1_P
		return nothing
	end

	if observableId == :IRS1_P 
		observableParameter1_IRS1_P = get_obs_sd_parameter(θ_observable, parameter_map)
		out[8] = observableParameter1_IRS1_P
		return nothing
	end

	if observableId == :IRS1_P_DosR 
		observableParameter1_IRS1_P_DosR = get_obs_sd_parameter(θ_observable, parameter_map)
		out[8] = observableParameter1_IRS1_P_DosR
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                       θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :IR1_P 
		return nothing
	end

	if observableId == :IRS1_P 
		return nothing
	end

	if observableId == :IRS1_P_DosR 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector, 
                        parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :IR1_P 
		return nothing
	end

	if observableId == :IRS1_P 
		return nothing
	end

	if observableId == :IRS1_P_DosR 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector, 
                        parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :IR1_P 
		return nothing
	end

	if observableId == :IRS1_P 
		return nothing
	end

	if observableId == :IRS1_P_DosR 
		return nothing
	end

end

