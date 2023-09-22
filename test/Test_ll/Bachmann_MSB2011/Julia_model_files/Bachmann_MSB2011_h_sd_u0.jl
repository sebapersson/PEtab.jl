#u[1] = p1EpoRpJAK2, u[2] = pSTAT5, u[3] = EpoRJAK2_CIS, u[4] = SOCS3nRNA4, u[5] = SOCS3RNA, u[6] = SHP1, u[7] = STAT5, u[8] = EpoRJAK2, u[9] = CISnRNA1, u[10] = SOCS3nRNA1, u[11] = SOCS3nRNA2, u[12] = CISnRNA3, u[13] = CISnRNA4, u[14] = SOCS3, u[15] = CISnRNA5, u[16] = SOCS3nRNA5, u[17] = SOCS3nRNA3, u[18] = SHP1Act, u[19] = npSTAT5, u[20] = p12EpoRpJAK2, u[21] = p2EpoRpJAK2, u[22] = CIS, u[23] = EpoRpJAK2, u[24] = CISnRNA2, u[25] = CISRNA
#p_ode_problem_names[1] = SOCS3RNATurn, p_ode_problem_names[2] = STAT5Imp, p_ode_problem_names[3] = SOCS3Eqc, p_ode_problem_names[4] = EpoRCISRemove, p_ode_problem_names[5] = STAT5ActEpoR, p_ode_problem_names[6] = SHP1ActEpoR, p_ode_problem_names[7] = JAK2EpoRDeaSHP1, p_ode_problem_names[8] = CISTurn, p_ode_problem_names[9] = SOCS3Turn, p_ode_problem_names[10] = init_EpoRJAK2_CIS, p_ode_problem_names[11] = SOCS3Inh, p_ode_problem_names[12] = ActD, p_ode_problem_names[13] = init_CIS_multiplier, p_ode_problem_names[14] = cyt, p_ode_problem_names[15] = CISRNAEqc, p_ode_problem_names[16] = JAK2ActEpo, p_ode_problem_names[17] = Epo, p_ode_problem_names[18] = SOCS3oe, p_ode_problem_names[19] = CISInh, p_ode_problem_names[20] = SHP1Dea, p_ode_problem_names[21] = SOCS3EqcOE, p_ode_problem_names[22] = CISRNADelay, p_ode_problem_names[23] = init_SHP1, p_ode_problem_names[24] = CISEqcOE, p_ode_problem_names[25] = EpoRActJAK2, p_ode_problem_names[26] = SOCS3RNAEqc, p_ode_problem_names[27] = CISEqc, p_ode_problem_names[28] = SHP1ProOE, p_ode_problem_names[29] = SOCS3RNADelay, p_ode_problem_names[30] = init_STAT5, p_ode_problem_names[31] = CISoe, p_ode_problem_names[32] = CISRNATurn, p_ode_problem_names[33] = init_SHP1_multiplier, p_ode_problem_names[34] = init_EpoRJAK2, p_ode_problem_names[35] = nuc, p_ode_problem_names[36] = EpoRCISInh, p_ode_problem_names[37] = STAT5ActJAK2, p_ode_problem_names[38] = STAT5Exp, p_ode_problem_names[39] = init_SOCS3_multiplier
##parameter_info.nominalValue[5] = CISRNAEqc_C 
#parameter_info.nominalValue[20] = SOCS3RNAEqc_C 


function compute_h(u::AbstractVector, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                   θ_non_dynamic::AbstractVector, parameter_info::ParametersInfo, observableId::Symbol,
                      parameter_map::θObsOrSdParameterMap)::Real 
	if observableId === :observable_CISRNA_foldA 
		observableParameter1_observable_CISRNA_foldA = get_obs_sd_parameter(θ_observable, parameter_map)
		return u[25] * observableParameter1_observable_CISRNA_foldA / p_ode_problem[15] + 1.0 
	end

	if observableId === :observable_CISRNA_foldB 
		observableParameter1_observable_CISRNA_foldB = get_obs_sd_parameter(θ_observable, parameter_map)
		return u[25] * observableParameter1_observable_CISRNA_foldB / p_ode_problem[15] + 1.0 
	end

	if observableId === :observable_CISRNA_foldC 
		observableParameter1_observable_CISRNA_foldC = get_obs_sd_parameter(θ_observable, parameter_map)
		return u[25] * observableParameter1_observable_CISRNA_foldC / p_ode_problem[15] + 1.0 
	end

	if observableId === :observable_CIS_abs 
		return u[22] 
	end

	if observableId === :observable_CIS_au 
		observableParameter1_observable_CIS_au, observableParameter2_observable_CIS_au = get_obs_sd_parameter(θ_observable, parameter_map)
		return observableParameter1_observable_CIS_au + u[22] * observableParameter2_observable_CIS_au / p_ode_problem[27] 
	end

	if observableId === :observable_CIS_au1 
		observableParameter1_observable_CIS_au1 = get_obs_sd_parameter(θ_observable, parameter_map)
		return u[22] * observableParameter1_observable_CIS_au1 / p_ode_problem[27] 
	end

	if observableId === :observable_CIS_au2 
		observableParameter1_observable_CIS_au2 = get_obs_sd_parameter(θ_observable, parameter_map)
		return u[22] * observableParameter1_observable_CIS_au2 / p_ode_problem[27] 
	end

	if observableId === :observable_SHP1_abs 
		return u[6] + u[18] 
	end

	if observableId === :observable_SOCS3RNA_foldA 
		observableParameter1_observable_SOCS3RNA_foldA = get_obs_sd_parameter(θ_observable, parameter_map)
		return u[5] * observableParameter1_observable_SOCS3RNA_foldA / p_ode_problem[26] + 1.0 
	end

	if observableId === :observable_SOCS3RNA_foldB 
		observableParameter1_observable_SOCS3RNA_foldB = get_obs_sd_parameter(θ_observable, parameter_map)
		return u[5] * observableParameter1_observable_SOCS3RNA_foldB / p_ode_problem[26] + 1.0 
	end

	if observableId === :observable_SOCS3RNA_foldC 
		observableParameter1_observable_SOCS3RNA_foldC = get_obs_sd_parameter(θ_observable, parameter_map)
		return u[5] * observableParameter1_observable_SOCS3RNA_foldC / p_ode_problem[26] + 1.0 
	end

	if observableId === :observable_SOCS3_abs 
		return u[14] 
	end

	if observableId === :observable_SOCS3_au 
		observableParameter1_observable_SOCS3_au, observableParameter2_observable_SOCS3_au = get_obs_sd_parameter(θ_observable, parameter_map)
		return observableParameter1_observable_SOCS3_au + u[14] * observableParameter2_observable_SOCS3_au / p_ode_problem[3] 
	end

	if observableId === :observable_STAT5_abs 
		return u[7] 
	end

	if observableId === :observable_pEpoR_au 
		observableParameter1_observable_pEpoR_au, observableParameter2_observable_pEpoR_au = get_obs_sd_parameter(θ_observable, parameter_map)
		return observableParameter1_observable_pEpoR_au + observableParameter2_observable_pEpoR_au * ( 16.0 * u[20] + 16.0 * u[1] + 16.0 * u[21] ) / p_ode_problem[34] 
	end

	if observableId === :observable_pJAK2_au 
		observableParameter1_observable_pJAK2_au, observableParameter2_observable_pJAK2_au = get_obs_sd_parameter(θ_observable, parameter_map)
		return observableParameter1_observable_pJAK2_au + observableParameter2_observable_pJAK2_au * ( 2.0 * u[23] + 2.0 * u[20] + 2.0 * u[1] + 2.0 * u[21] ) / p_ode_problem[34] 
	end

	if observableId === :observable_pSTAT5B_rel 
		observableParameter1_observable_pSTAT5B_rel = get_obs_sd_parameter(θ_observable, parameter_map)
		return observableParameter1_observable_pSTAT5B_rel + 100.0 * u[2] / ( u[7] + u[2] ) 
	end

	if observableId === :observable_pSTAT5_au 
		observableParameter1_observable_pSTAT5_au, observableParameter2_observable_pSTAT5_au = get_obs_sd_parameter(θ_observable, parameter_map)
		return observableParameter1_observable_pSTAT5_au + u[2] * observableParameter2_observable_pSTAT5_au / p_ode_problem[30] 
	end

	if observableId === :observable_tSHP1_au 
		observableParameter1_observable_tSHP1_au = get_obs_sd_parameter(θ_observable, parameter_map)
		return observableParameter1_observable_tSHP1_au * ( u[6] + u[18] ) / p_ode_problem[23] 
	end

	if observableId === :observable_tSTAT5_au 
		observableParameter1_observable_tSTAT5_au = get_obs_sd_parameter(θ_observable, parameter_map)
		return observableParameter1_observable_tSTAT5_au * ( u[7] + u[2] ) / p_ode_problem[30] 
	end

end

function compute_u0!(u0::AbstractVector, p_ode_problem::AbstractVector) 

	#p_ode_problem[1] = SOCS3RNATurn, p_ode_problem[2] = STAT5Imp, p_ode_problem[3] = SOCS3Eqc, p_ode_problem[4] = EpoRCISRemove, p_ode_problem[5] = STAT5ActEpoR, p_ode_problem[6] = SHP1ActEpoR, p_ode_problem[7] = JAK2EpoRDeaSHP1, p_ode_problem[8] = CISTurn, p_ode_problem[9] = SOCS3Turn, p_ode_problem[10] = init_EpoRJAK2_CIS, p_ode_problem[11] = SOCS3Inh, p_ode_problem[12] = ActD, p_ode_problem[13] = init_CIS_multiplier, p_ode_problem[14] = cyt, p_ode_problem[15] = CISRNAEqc, p_ode_problem[16] = JAK2ActEpo, p_ode_problem[17] = Epo, p_ode_problem[18] = SOCS3oe, p_ode_problem[19] = CISInh, p_ode_problem[20] = SHP1Dea, p_ode_problem[21] = SOCS3EqcOE, p_ode_problem[22] = CISRNADelay, p_ode_problem[23] = init_SHP1, p_ode_problem[24] = CISEqcOE, p_ode_problem[25] = EpoRActJAK2, p_ode_problem[26] = SOCS3RNAEqc, p_ode_problem[27] = CISEqc, p_ode_problem[28] = SHP1ProOE, p_ode_problem[29] = SOCS3RNADelay, p_ode_problem[30] = init_STAT5, p_ode_problem[31] = CISoe, p_ode_problem[32] = CISRNATurn, p_ode_problem[33] = init_SHP1_multiplier, p_ode_problem[34] = init_EpoRJAK2, p_ode_problem[35] = nuc, p_ode_problem[36] = EpoRCISInh, p_ode_problem[37] = STAT5ActJAK2, p_ode_problem[38] = STAT5Exp, p_ode_problem[39] = init_SOCS3_multiplier

	t = 0.0 # u at time zero

	p1EpoRpJAK2 = 0.0 
	pSTAT5 = 0.0 
	EpoRJAK2_CIS = p_ode_problem[10] 
	SOCS3nRNA4 = 0.0 
	SOCS3RNA = 0.0 
	SHP1 = p_ode_problem[23] * ( 1.0 + p_ode_problem[28] * p_ode_problem[33] ) 
	STAT5 = p_ode_problem[30] 
	EpoRJAK2 = p_ode_problem[34] 
	CISnRNA1 = 0.0 
	SOCS3nRNA1 = 0.0 
	SOCS3nRNA2 = 0.0 
	CISnRNA3 = 0.0 
	CISnRNA4 = 0.0 
	SOCS3 = p_ode_problem[3] * p_ode_problem[21] * p_ode_problem[39] 
	CISnRNA5 = 0.0 
	SOCS3nRNA5 = 0.0 
	SOCS3nRNA3 = 0.0 
	SHP1Act = 0.0 
	npSTAT5 = 0.0 
	p12EpoRpJAK2 = 0.0 
	p2EpoRpJAK2 = 0.0 
	CIS = p_ode_problem[27] * p_ode_problem[24] * p_ode_problem[13] 
	EpoRpJAK2 = 0.0 
	CISnRNA2 = 0.0 
	CISRNA = 0.0 

	u0 .= p1EpoRpJAK2, pSTAT5, EpoRJAK2_CIS, SOCS3nRNA4, SOCS3RNA, SHP1, STAT5, EpoRJAK2, CISnRNA1, SOCS3nRNA1, SOCS3nRNA2, CISnRNA3, CISnRNA4, SOCS3, CISnRNA5, SOCS3nRNA5, SOCS3nRNA3, SHP1Act, npSTAT5, p12EpoRpJAK2, p2EpoRpJAK2, CIS, EpoRpJAK2, CISnRNA2, CISRNA
end

function compute_u0(p_ode_problem::AbstractVector)::AbstractVector 

	#p_ode_problem[1] = SOCS3RNATurn, p_ode_problem[2] = STAT5Imp, p_ode_problem[3] = SOCS3Eqc, p_ode_problem[4] = EpoRCISRemove, p_ode_problem[5] = STAT5ActEpoR, p_ode_problem[6] = SHP1ActEpoR, p_ode_problem[7] = JAK2EpoRDeaSHP1, p_ode_problem[8] = CISTurn, p_ode_problem[9] = SOCS3Turn, p_ode_problem[10] = init_EpoRJAK2_CIS, p_ode_problem[11] = SOCS3Inh, p_ode_problem[12] = ActD, p_ode_problem[13] = init_CIS_multiplier, p_ode_problem[14] = cyt, p_ode_problem[15] = CISRNAEqc, p_ode_problem[16] = JAK2ActEpo, p_ode_problem[17] = Epo, p_ode_problem[18] = SOCS3oe, p_ode_problem[19] = CISInh, p_ode_problem[20] = SHP1Dea, p_ode_problem[21] = SOCS3EqcOE, p_ode_problem[22] = CISRNADelay, p_ode_problem[23] = init_SHP1, p_ode_problem[24] = CISEqcOE, p_ode_problem[25] = EpoRActJAK2, p_ode_problem[26] = SOCS3RNAEqc, p_ode_problem[27] = CISEqc, p_ode_problem[28] = SHP1ProOE, p_ode_problem[29] = SOCS3RNADelay, p_ode_problem[30] = init_STAT5, p_ode_problem[31] = CISoe, p_ode_problem[32] = CISRNATurn, p_ode_problem[33] = init_SHP1_multiplier, p_ode_problem[34] = init_EpoRJAK2, p_ode_problem[35] = nuc, p_ode_problem[36] = EpoRCISInh, p_ode_problem[37] = STAT5ActJAK2, p_ode_problem[38] = STAT5Exp, p_ode_problem[39] = init_SOCS3_multiplier

	t = 0.0 # u at time zero

	p1EpoRpJAK2 = 0.0 
	pSTAT5 = 0.0 
	EpoRJAK2_CIS = p_ode_problem[10] 
	SOCS3nRNA4 = 0.0 
	SOCS3RNA = 0.0 
	SHP1 = p_ode_problem[23] * ( 1.0 + p_ode_problem[28] * p_ode_problem[33] ) 
	STAT5 = p_ode_problem[30] 
	EpoRJAK2 = p_ode_problem[34] 
	CISnRNA1 = 0.0 
	SOCS3nRNA1 = 0.0 
	SOCS3nRNA2 = 0.0 
	CISnRNA3 = 0.0 
	CISnRNA4 = 0.0 
	SOCS3 = p_ode_problem[3] * p_ode_problem[21] * p_ode_problem[39] 
	CISnRNA5 = 0.0 
	SOCS3nRNA5 = 0.0 
	SOCS3nRNA3 = 0.0 
	SHP1Act = 0.0 
	npSTAT5 = 0.0 
	p12EpoRpJAK2 = 0.0 
	p2EpoRpJAK2 = 0.0 
	CIS = p_ode_problem[27] * p_ode_problem[24] * p_ode_problem[13] 
	EpoRpJAK2 = 0.0 
	CISnRNA2 = 0.0 
	CISRNA = 0.0 

	 return [p1EpoRpJAK2, pSTAT5, EpoRJAK2_CIS, SOCS3nRNA4, SOCS3RNA, SHP1, STAT5, EpoRJAK2, CISnRNA1, SOCS3nRNA1, SOCS3nRNA2, CISnRNA3, CISnRNA4, SOCS3, CISnRNA5, SOCS3nRNA5, SOCS3nRNA3, SHP1Act, npSTAT5, p12EpoRpJAK2, p2EpoRpJAK2, CIS, EpoRpJAK2, CISnRNA2, CISRNA]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                   parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap)::Real 
	if observableId === :observable_CISRNA_foldA 
		noiseParameter1_observable_CISRNA_foldA = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_CISRNA_foldA 
	end

	if observableId === :observable_CISRNA_foldB 
		noiseParameter1_observable_CISRNA_foldB = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_CISRNA_foldB 
	end

	if observableId === :observable_CISRNA_foldC 
		noiseParameter1_observable_CISRNA_foldC = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_CISRNA_foldC 
	end

	if observableId === :observable_CIS_abs 
		noiseParameter1_observable_CIS_abs = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_CIS_abs 
	end

	if observableId === :observable_CIS_au 
		noiseParameter1_observable_CIS_au = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_CIS_au 
	end

	if observableId === :observable_CIS_au1 
		noiseParameter1_observable_CIS_au1 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_CIS_au1 
	end

	if observableId === :observable_CIS_au2 
		noiseParameter1_observable_CIS_au2 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_CIS_au2 
	end

	if observableId === :observable_SHP1_abs 
		noiseParameter1_observable_SHP1_abs = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_SHP1_abs 
	end

	if observableId === :observable_SOCS3RNA_foldA 
		noiseParameter1_observable_SOCS3RNA_foldA = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_SOCS3RNA_foldA 
	end

	if observableId === :observable_SOCS3RNA_foldB 
		noiseParameter1_observable_SOCS3RNA_foldB = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_SOCS3RNA_foldB 
	end

	if observableId === :observable_SOCS3RNA_foldC 
		noiseParameter1_observable_SOCS3RNA_foldC = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_SOCS3RNA_foldC 
	end

	if observableId === :observable_SOCS3_abs 
		noiseParameter1_observable_SOCS3_abs = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_SOCS3_abs 
	end

	if observableId === :observable_SOCS3_au 
		noiseParameter1_observable_SOCS3_au = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_SOCS3_au 
	end

	if observableId === :observable_STAT5_abs 
		noiseParameter1_observable_STAT5_abs = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_STAT5_abs 
	end

	if observableId === :observable_pEpoR_au 
		noiseParameter1_observable_pEpoR_au = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_pEpoR_au 
	end

	if observableId === :observable_pJAK2_au 
		noiseParameter1_observable_pJAK2_au = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_pJAK2_au 
	end

	if observableId === :observable_pSTAT5B_rel 
		noiseParameter1_observable_pSTAT5B_rel = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_pSTAT5B_rel 
	end

	if observableId === :observable_pSTAT5_au 
		noiseParameter1_observable_pSTAT5_au, noiseParameter2_observable_pSTAT5_au = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_pSTAT5_au + noiseParameter2_observable_pSTAT5_au 
	end

	if observableId === :observable_tSHP1_au 
		noiseParameter1_observable_tSHP1_au = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_tSHP1_au 
	end

	if observableId === :observable_tSTAT5_au 
		noiseParameter1_observable_tSTAT5_au = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_tSTAT5_au 
	end


end

