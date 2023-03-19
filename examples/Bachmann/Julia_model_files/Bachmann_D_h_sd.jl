#u[1] = p1EpoRpJAK2, u[2] = pSTAT5, u[3] = EpoRJAK2_CIS, u[4] = SOCS3nRNA4, u[5] = SOCS3RNA, u[6] = SHP1, u[7] = STAT5, u[8] = EpoRJAK2, u[9] = CISnRNA1, u[10] = SOCS3nRNA1, u[11] = SOCS3nRNA2, u[12] = CISnRNA3, u[13] = CISnRNA4, u[14] = SOCS3, u[15] = CISnRNA5, u[16] = SOCS3nRNA5, u[17] = SOCS3nRNA3, u[18] = SHP1Act, u[19] = npSTAT5, u[20] = p12EpoRpJAK2, u[21] = p2EpoRpJAK2, u[22] = CIS, u[23] = EpoRpJAK2, u[24] = CISnRNA2, u[25] = CISRNA
#pODEProblem[1] = STAT5Exp, pODEProblem[2] = STAT5Imp, pODEProblem[3] = init_SOCS3_multiplier, pODEProblem[4] = EpoRCISRemove, pODEProblem[5] = STAT5ActEpoR, pODEProblem[6] = SHP1ActEpoR, pODEProblem[7] = JAK2EpoRDeaSHP1, pODEProblem[8] = CISTurn, pODEProblem[9] = SOCS3Turn, pODEProblem[10] = init_EpoRJAK2_CIS, pODEProblem[11] = SOCS3Inh, pODEProblem[12] = ActD, pODEProblem[13] = init_CIS_multiplier, pODEProblem[14] = cyt, pODEProblem[15] = CISRNAEqc, pODEProblem[16] = JAK2ActEpo, pODEProblem[17] = Epo, pODEProblem[18] = SOCS3oe, pODEProblem[19] = CISInh, pODEProblem[20] = SHP1Dea, pODEProblem[21] = SOCS3EqcOE, pODEProblem[22] = CISRNADelay, pODEProblem[23] = init_SHP1, pODEProblem[24] = CISEqcOE, pODEProblem[25] = EpoRActJAK2, pODEProblem[26] = SOCS3RNAEqc, pODEProblem[27] = CISEqc, pODEProblem[28] = SHP1ProOE, pODEProblem[29] = SOCS3RNADelay, pODEProblem[30] = init_STAT5, pODEProblem[31] = CISoe, pODEProblem[32] = CISRNATurn, pODEProblem[33] = init_SHP1_multiplier, pODEProblem[34] = init_EpoRJAK2, pODEProblem[35] = nuc, pODEProblem[36] = EpoRCISInh, pODEProblem[37] = STAT5ActJAK2, pODEProblem[38] = SOCS3RNATurn, pODEProblem[39] = SOCS3Eqc
#
function compute_∂h∂u!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,                    
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :observable_CISRNA_foldA 
		observableParameter1_observable_CISRNA_foldA = getObsOrSdParam(θ_observable, parameterMap)
		out[25] = observableParameter1_observable_CISRNA_foldA / pODEProblem[15]
		return nothing
	end

	if observableId == :observable_CISRNA_foldB 
		observableParameter1_observable_CISRNA_foldB = getObsOrSdParam(θ_observable, parameterMap)
		out[25] = observableParameter1_observable_CISRNA_foldB / pODEProblem[15]
		return nothing
	end

	if observableId == :observable_CISRNA_foldC 
		observableParameter1_observable_CISRNA_foldC = getObsOrSdParam(θ_observable, parameterMap)
		out[25] = observableParameter1_observable_CISRNA_foldC / pODEProblem[15]
		return nothing
	end

	if observableId == :observable_CIS_abs 
		out[22] = 1
		return nothing
	end

	if observableId == :observable_CIS_au 
		observableParameter1_observable_CIS_au, observableParameter2_observable_CIS_au = getObsOrSdParam(θ_observable, parameterMap)
		out[22] = observableParameter2_observable_CIS_au / pODEProblem[27]
		return nothing
	end

	if observableId == :observable_CIS_au1 
		observableParameter1_observable_CIS_au1 = getObsOrSdParam(θ_observable, parameterMap)
		out[22] = observableParameter1_observable_CIS_au1 / pODEProblem[27]
		return nothing
	end

	if observableId == :observable_CIS_au2 
		observableParameter1_observable_CIS_au2 = getObsOrSdParam(θ_observable, parameterMap)
		out[22] = observableParameter1_observable_CIS_au2 / pODEProblem[27]
		return nothing
	end

	if observableId == :observable_SHP1_abs 
		out[6] = 1
		out[18] = 1
		return nothing
	end

	if observableId == :observable_SOCS3RNA_foldA 
		observableParameter1_observable_SOCS3RNA_foldA = getObsOrSdParam(θ_observable, parameterMap)
		out[5] = observableParameter1_observable_SOCS3RNA_foldA / pODEProblem[26]
		return nothing
	end

	if observableId == :observable_SOCS3RNA_foldB 
		observableParameter1_observable_SOCS3RNA_foldB = getObsOrSdParam(θ_observable, parameterMap)
		out[5] = observableParameter1_observable_SOCS3RNA_foldB / pODEProblem[26]
		return nothing
	end

	if observableId == :observable_SOCS3RNA_foldC 
		observableParameter1_observable_SOCS3RNA_foldC = getObsOrSdParam(θ_observable, parameterMap)
		out[5] = observableParameter1_observable_SOCS3RNA_foldC / pODEProblem[26]
		return nothing
	end

	if observableId == :observable_SOCS3_abs 
		out[14] = 1
		return nothing
	end

	if observableId == :observable_SOCS3_au 
		observableParameter1_observable_SOCS3_au, observableParameter2_observable_SOCS3_au = getObsOrSdParam(θ_observable, parameterMap)
		out[14] = observableParameter2_observable_SOCS3_au / pODEProblem[39]
		return nothing
	end

	if observableId == :observable_STAT5_abs 
		out[7] = 1
		return nothing
	end

	if observableId == :observable_pEpoR_au 
		observableParameter1_observable_pEpoR_au, observableParameter2_observable_pEpoR_au = getObsOrSdParam(θ_observable, parameterMap)
		out[1] = (16observableParameter2_observable_pEpoR_au) / pODEProblem[34]
		out[20] = (16observableParameter2_observable_pEpoR_au) / pODEProblem[34]
		out[21] = (16observableParameter2_observable_pEpoR_au) / pODEProblem[34]
		return nothing
	end

	if observableId == :observable_pJAK2_au 
		observableParameter1_observable_pJAK2_au, observableParameter2_observable_pJAK2_au = getObsOrSdParam(θ_observable, parameterMap)
		out[1] = (2observableParameter2_observable_pJAK2_au) / pODEProblem[34]
		out[20] = (2observableParameter2_observable_pJAK2_au) / pODEProblem[34]
		out[21] = (2observableParameter2_observable_pJAK2_au) / pODEProblem[34]
		out[23] = (2observableParameter2_observable_pJAK2_au) / pODEProblem[34]
		return nothing
	end

	if observableId == :observable_pSTAT5B_rel 
		observableParameter1_observable_pSTAT5B_rel = getObsOrSdParam(θ_observable, parameterMap)
		out[2] = (100.0u[7]) / ((u[7] + u[2])^2)
		out[7] = (-100u[2]) / ((u[7] + u[2])^2)
		return nothing
	end

	if observableId == :observable_pSTAT5_au 
		observableParameter1_observable_pSTAT5_au, observableParameter2_observable_pSTAT5_au = getObsOrSdParam(θ_observable, parameterMap)
		out[2] = observableParameter2_observable_pSTAT5_au / pODEProblem[30]
		return nothing
	end

	if observableId == :observable_tSHP1_au 
		observableParameter1_observable_tSHP1_au = getObsOrSdParam(θ_observable, parameterMap)
		out[6] = observableParameter1_observable_tSHP1_au / pODEProblem[23]
		out[18] = observableParameter1_observable_tSHP1_au / pODEProblem[23]
		return nothing
	end

	if observableId == :observable_tSTAT5_au 
		observableParameter1_observable_tSTAT5_au = getObsOrSdParam(θ_observable, parameterMap)
		out[2] = observableParameter1_observable_tSTAT5_au / pODEProblem[30]
		out[7] = observableParameter1_observable_tSTAT5_au / pODEProblem[30]
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :observable_CISRNA_foldA 
		observableParameter1_observable_CISRNA_foldA = getObsOrSdParam(θ_observable, parameterMap)
		out[15] = (-u[25]*observableParameter1_observable_CISRNA_foldA) / (pODEProblem[15]^2)
		return nothing
	end

	if observableId == :observable_CISRNA_foldB 
		observableParameter1_observable_CISRNA_foldB = getObsOrSdParam(θ_observable, parameterMap)
		out[15] = (-u[25]*observableParameter1_observable_CISRNA_foldB) / (pODEProblem[15]^2)
		return nothing
	end

	if observableId == :observable_CISRNA_foldC 
		observableParameter1_observable_CISRNA_foldC = getObsOrSdParam(θ_observable, parameterMap)
		out[15] = (-u[25]*observableParameter1_observable_CISRNA_foldC) / (pODEProblem[15]^2)
		return nothing
	end

	if observableId == :observable_CIS_abs 
		return nothing
	end

	if observableId == :observable_CIS_au 
		observableParameter1_observable_CIS_au, observableParameter2_observable_CIS_au = getObsOrSdParam(θ_observable, parameterMap)
		out[27] = (-u[22]*observableParameter2_observable_CIS_au) / (pODEProblem[27]^2)
		return nothing
	end

	if observableId == :observable_CIS_au1 
		observableParameter1_observable_CIS_au1 = getObsOrSdParam(θ_observable, parameterMap)
		out[27] = (-u[22]*observableParameter1_observable_CIS_au1) / (pODEProblem[27]^2)
		return nothing
	end

	if observableId == :observable_CIS_au2 
		observableParameter1_observable_CIS_au2 = getObsOrSdParam(θ_observable, parameterMap)
		out[27] = (-u[22]*observableParameter1_observable_CIS_au2) / (pODEProblem[27]^2)
		return nothing
	end

	if observableId == :observable_SHP1_abs 
		return nothing
	end

	if observableId == :observable_SOCS3RNA_foldA 
		observableParameter1_observable_SOCS3RNA_foldA = getObsOrSdParam(θ_observable, parameterMap)
		out[26] = (-u[5]*observableParameter1_observable_SOCS3RNA_foldA) / (pODEProblem[26]^2)
		return nothing
	end

	if observableId == :observable_SOCS3RNA_foldB 
		observableParameter1_observable_SOCS3RNA_foldB = getObsOrSdParam(θ_observable, parameterMap)
		out[26] = (-u[5]*observableParameter1_observable_SOCS3RNA_foldB) / (pODEProblem[26]^2)
		return nothing
	end

	if observableId == :observable_SOCS3RNA_foldC 
		observableParameter1_observable_SOCS3RNA_foldC = getObsOrSdParam(θ_observable, parameterMap)
		out[26] = (-u[5]*observableParameter1_observable_SOCS3RNA_foldC) / (pODEProblem[26]^2)
		return nothing
	end

	if observableId == :observable_SOCS3_abs 
		return nothing
	end

	if observableId == :observable_SOCS3_au 
		observableParameter1_observable_SOCS3_au, observableParameter2_observable_SOCS3_au = getObsOrSdParam(θ_observable, parameterMap)
		out[39] = (-u[14]*observableParameter2_observable_SOCS3_au) / (pODEProblem[39]^2)
		return nothing
	end

	if observableId == :observable_STAT5_abs 
		return nothing
	end

	if observableId == :observable_pEpoR_au 
		observableParameter1_observable_pEpoR_au, observableParameter2_observable_pEpoR_au = getObsOrSdParam(θ_observable, parameterMap)
		out[34] = (observableParameter2_observable_pEpoR_au*(-16u[20] - 16u[1] - 16u[21])) / (pODEProblem[34]^2)
		return nothing
	end

	if observableId == :observable_pJAK2_au 
		observableParameter1_observable_pJAK2_au, observableParameter2_observable_pJAK2_au = getObsOrSdParam(θ_observable, parameterMap)
		out[34] = (-observableParameter2_observable_pJAK2_au*(2u[23] + 2u[20] + 2u[1] + 2u[21])) / (pODEProblem[34]^2)
		return nothing
	end

	if observableId == :observable_pSTAT5B_rel 
		return nothing
	end

	if observableId == :observable_pSTAT5_au 
		observableParameter1_observable_pSTAT5_au, observableParameter2_observable_pSTAT5_au = getObsOrSdParam(θ_observable, parameterMap)
		out[30] = (-observableParameter2_observable_pSTAT5_au*u[2]) / (pODEProblem[30]^2)
		return nothing
	end

	if observableId == :observable_tSHP1_au 
		observableParameter1_observable_tSHP1_au = getObsOrSdParam(θ_observable, parameterMap)
		out[23] = (-observableParameter1_observable_tSHP1_au*(u[6] + u[18])) / (pODEProblem[23]^2)
		return nothing
	end

	if observableId == :observable_tSTAT5_au 
		observableParameter1_observable_tSTAT5_au = getObsOrSdParam(θ_observable, parameterMap)
		out[30] = (observableParameter1_observable_tSTAT5_au*(-u[7] - u[2])) / (pODEProblem[30]^2)
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector, 
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :observable_CISRNA_foldA 
		return nothing
	end

	if observableId == :observable_CISRNA_foldB 
		return nothing
	end

	if observableId == :observable_CISRNA_foldC 
		return nothing
	end

	if observableId == :observable_CIS_abs 
		return nothing
	end

	if observableId == :observable_CIS_au 
		return nothing
	end

	if observableId == :observable_CIS_au1 
		return nothing
	end

	if observableId == :observable_CIS_au2 
		return nothing
	end

	if observableId == :observable_SHP1_abs 
		return nothing
	end

	if observableId == :observable_SOCS3RNA_foldA 
		return nothing
	end

	if observableId == :observable_SOCS3RNA_foldB 
		return nothing
	end

	if observableId == :observable_SOCS3RNA_foldC 
		return nothing
	end

	if observableId == :observable_SOCS3_abs 
		return nothing
	end

	if observableId == :observable_SOCS3_au 
		return nothing
	end

	if observableId == :observable_STAT5_abs 
		return nothing
	end

	if observableId == :observable_pEpoR_au 
		return nothing
	end

	if observableId == :observable_pJAK2_au 
		return nothing
	end

	if observableId == :observable_pSTAT5B_rel 
		return nothing
	end

	if observableId == :observable_pSTAT5_au 
		return nothing
	end

	if observableId == :observable_tSHP1_au 
		return nothing
	end

	if observableId == :observable_tSTAT5_au 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector, 
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :observable_CISRNA_foldA 
		return nothing
	end

	if observableId == :observable_CISRNA_foldB 
		return nothing
	end

	if observableId == :observable_CISRNA_foldC 
		return nothing
	end

	if observableId == :observable_CIS_abs 
		return nothing
	end

	if observableId == :observable_CIS_au 
		return nothing
	end

	if observableId == :observable_CIS_au1 
		return nothing
	end

	if observableId == :observable_CIS_au2 
		return nothing
	end

	if observableId == :observable_SHP1_abs 
		return nothing
	end

	if observableId == :observable_SOCS3RNA_foldA 
		return nothing
	end

	if observableId == :observable_SOCS3RNA_foldB 
		return nothing
	end

	if observableId == :observable_SOCS3RNA_foldC 
		return nothing
	end

	if observableId == :observable_SOCS3_abs 
		return nothing
	end

	if observableId == :observable_SOCS3_au 
		return nothing
	end

	if observableId == :observable_STAT5_abs 
		return nothing
	end

	if observableId == :observable_pEpoR_au 
		return nothing
	end

	if observableId == :observable_pJAK2_au 
		return nothing
	end

	if observableId == :observable_pSTAT5B_rel 
		return nothing
	end

	if observableId == :observable_pSTAT5_au 
		return nothing
	end

	if observableId == :observable_tSHP1_au 
		return nothing
	end

	if observableId == :observable_tSTAT5_au 
		return nothing
	end

end

