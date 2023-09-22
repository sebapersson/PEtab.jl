#u[1] = pAkt_S6, u[2] = pAkt, u[3] = pS6, u[4] = EGFR, u[5] = pEGFR_Akt, u[6] = pEGFR, u[7] = Akt, u[8] = S6, u[9] = EGF_EGFR
#p_ode_problem[1] = EGF_end, p_ode_problem[2] = reaction_5_k1, p_ode_problem[3] = reaction_2_k2, p_ode_problem[4] = init_AKT, p_ode_problem[5] = init_EGFR, p_ode_problem[6] = EGF_bool1, p_ode_problem[7] = EGF_rate, p_ode_problem[8] = EGFR_turnover, p_ode_problem[9] = reaction_1_k1, p_ode_problem[10] = reaction_1_k2, p_ode_problem[11] = reaction_8_k1, p_ode_problem[12] = reaction_4_k1, p_ode_problem[13] = reaction_6_k1, p_ode_problem[14] = reaction_2_k1, p_ode_problem[15] = init_S6, p_ode_problem[16] = reaction_7_k1, p_ode_problem[17] = reaction_9_k1, p_ode_problem[18] = reaction_3_k1, p_ode_problem[19] = reaction_5_k2, p_ode_problem[20] = Cell, p_ode_problem[21] = EGF_0
#
function compute_∂h∂u!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                       θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :pAkt_tot 
		observableParameter1_pAkt_tot = get_obs_sd_parameter(θ_observable, parameter_map)
		out[1] = observableParameter1_pAkt_tot
		out[2] = observableParameter1_pAkt_tot
		return nothing
	end

	if observableId == :pEGFR_tot 
		observableParameter1_pEGFR_tot = get_obs_sd_parameter(θ_observable, parameter_map)
		out[5] = observableParameter1_pEGFR_tot
		out[6] = observableParameter1_pEGFR_tot
		return nothing
	end

	if observableId == :pS6_tot 
		observableParameter1_pS6_tot = get_obs_sd_parameter(θ_observable, parameter_map)
		out[3] = observableParameter1_pS6_tot
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                       θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :pAkt_tot 
		return nothing
	end

	if observableId == :pEGFR_tot 
		return nothing
	end

	if observableId == :pS6_tot 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                        parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :pAkt_tot 
		return nothing
	end

	if observableId == :pEGFR_tot 
		return nothing
	end

	if observableId == :pS6_tot 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                        parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :pAkt_tot 
		return nothing
	end

	if observableId == :pEGFR_tot 
		return nothing
	end

	if observableId == :pS6_tot 
		return nothing
	end

end

