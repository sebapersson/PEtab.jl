#u[1] = pAkt_S6, u[2] = pAkt, u[3] = pS6, u[4] = EGFR, u[5] = pEGFR_Akt, u[6] = pEGFR, u[7] = Akt, u[8] = S6, u[9] = EGF_EGFR
#p_ode_problem_names[1] = EGF_end, p_ode_problem_names[2] = reaction_5_k1, p_ode_problem_names[3] = reaction_2_k2, p_ode_problem_names[4] = init_AKT, p_ode_problem_names[5] = init_EGFR, p_ode_problem_names[6] = EGF_bool1, p_ode_problem_names[7] = EGF_rate, p_ode_problem_names[8] = EGFR_turnover, p_ode_problem_names[9] = reaction_1_k1, p_ode_problem_names[10] = reaction_1_k2, p_ode_problem_names[11] = reaction_8_k1, p_ode_problem_names[12] = reaction_4_k1, p_ode_problem_names[13] = reaction_6_k1, p_ode_problem_names[14] = reaction_2_k1, p_ode_problem_names[15] = init_S6, p_ode_problem_names[16] = reaction_7_k1, p_ode_problem_names[17] = reaction_9_k1, p_ode_problem_names[18] = reaction_3_k1, p_ode_problem_names[19] = reaction_5_k2, p_ode_problem_names[20] = Cell, p_ode_problem_names[21] = EGF_0
#

function compute_h(u::AbstractVector, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                   θ_non_dynamic::AbstractVector, parameter_info::ParametersInfo, observableId::Symbol,
                      parameter_map::θObsOrSdParameterMap)::Real 
	if observableId === :pAkt_tot 
		observableParameter1_pAkt_tot = get_obs_sd_parameter(θ_observable, parameter_map)
		return observableParameter1_pAkt_tot * ( u[2] + u[1] ) 
	end

	if observableId === :pEGFR_tot 
		observableParameter1_pEGFR_tot = get_obs_sd_parameter(θ_observable, parameter_map)
		return observableParameter1_pEGFR_tot * ( u[6] + u[5] ) 
	end

	if observableId === :pS6_tot 
		observableParameter1_pS6_tot = get_obs_sd_parameter(θ_observable, parameter_map)
		return u[3] * observableParameter1_pS6_tot 
	end

end

function compute_u0!(u0::AbstractVector, p_ode_problem::AbstractVector) 

	#p_ode_problem[1] = EGF_end, p_ode_problem[2] = reaction_5_k1, p_ode_problem[3] = reaction_2_k2, p_ode_problem[4] = init_AKT, p_ode_problem[5] = init_EGFR, p_ode_problem[6] = EGF_bool1, p_ode_problem[7] = EGF_rate, p_ode_problem[8] = EGFR_turnover, p_ode_problem[9] = reaction_1_k1, p_ode_problem[10] = reaction_1_k2, p_ode_problem[11] = reaction_8_k1, p_ode_problem[12] = reaction_4_k1, p_ode_problem[13] = reaction_6_k1, p_ode_problem[14] = reaction_2_k1, p_ode_problem[15] = init_S6, p_ode_problem[16] = reaction_7_k1, p_ode_problem[17] = reaction_9_k1, p_ode_problem[18] = reaction_3_k1, p_ode_problem[19] = reaction_5_k2, p_ode_problem[20] = Cell, p_ode_problem[21] = EGF_0

	t = 0.0 # u at time zero

	pAkt_S6 = 0.0 
	pAkt = 0.0 
	pS6 = 0.0 
	EGFR = p_ode_problem[5] 
	pEGFR_Akt = 0.0 
	pEGFR = 0.0 
	Akt = p_ode_problem[4] 
	S6 = p_ode_problem[15] 
	EGF_EGFR = 0.0 

	u0 .= pAkt_S6, pAkt, pS6, EGFR, pEGFR_Akt, pEGFR, Akt, S6, EGF_EGFR
end

function compute_u0(p_ode_problem::AbstractVector)::AbstractVector 

	#p_ode_problem[1] = EGF_end, p_ode_problem[2] = reaction_5_k1, p_ode_problem[3] = reaction_2_k2, p_ode_problem[4] = init_AKT, p_ode_problem[5] = init_EGFR, p_ode_problem[6] = EGF_bool1, p_ode_problem[7] = EGF_rate, p_ode_problem[8] = EGFR_turnover, p_ode_problem[9] = reaction_1_k1, p_ode_problem[10] = reaction_1_k2, p_ode_problem[11] = reaction_8_k1, p_ode_problem[12] = reaction_4_k1, p_ode_problem[13] = reaction_6_k1, p_ode_problem[14] = reaction_2_k1, p_ode_problem[15] = init_S6, p_ode_problem[16] = reaction_7_k1, p_ode_problem[17] = reaction_9_k1, p_ode_problem[18] = reaction_3_k1, p_ode_problem[19] = reaction_5_k2, p_ode_problem[20] = Cell, p_ode_problem[21] = EGF_0

	t = 0.0 # u at time zero

	pAkt_S6 = 0.0 
	pAkt = 0.0 
	pS6 = 0.0 
	EGFR = p_ode_problem[5] 
	pEGFR_Akt = 0.0 
	pEGFR = 0.0 
	Akt = p_ode_problem[4] 
	S6 = p_ode_problem[15] 
	EGF_EGFR = 0.0 

	 return [pAkt_S6, pAkt, pS6, EGFR, pEGFR_Akt, pEGFR, Akt, S6, EGF_EGFR]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                   parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap)::Real 
	if observableId === :pAkt_tot 
		noiseParameter1_pAkt_tot = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_pAkt_tot 
	end

	if observableId === :pEGFR_tot 
		noiseParameter1_pEGFR_tot = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_pEGFR_tot 
	end

	if observableId === :pS6_tot 
		noiseParameter1_pS6_tot = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_pS6_tot 
	end


end

