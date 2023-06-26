#u[1] = K27me0K36me0, u[2] = K27me2K36me3, u[3] = K27me2K36me0, u[4] = K27me0K36me1, u[5] = K27me2K36me1, u[6] = K27me0K36me2, u[7] = K27me1K36me2, u[8] = K27me3K36me2, u[9] = K27me1K36me1, u[10] = K27me1K36me3, u[11] = K27me2K36me2, u[12] = K27me0K36me3, u[13] = K27me3K36me1, u[14] = K27me3K36me0, u[15] = K27me1K36me0
#pODEProblemNames[1] = k20_10, pODEProblemNames[2] = k11_12, pODEProblemNames[3] = k22_32, pODEProblemNames[4] = k21_11, pODEProblemNames[5] = k13_12, pODEProblemNames[6] = k01_00, pODEProblemNames[7] = k11_01, pODEProblemNames[8] = k22_21, pODEProblemNames[9] = k22_12, pODEProblemNames[10] = k11_10, pODEProblemNames[11] = k12_11, pODEProblemNames[12] = k21_31, pODEProblemNames[13] = k02_12, pODEProblemNames[14] = k30_20, pODEProblemNames[15] = dilution, pODEProblemNames[16] = k31_21, pODEProblemNames[17] = k23_22, pODEProblemNames[18] = k02_01, pODEProblemNames[19] = k03_13, pODEProblemNames[20] = k10_11, pODEProblemNames[21] = k21_20, pODEProblemNames[22] = k10_00, pODEProblemNames[23] = k30_31, pODEProblemNames[24] = k20_30, pODEProblemNames[25] = k20_21, pODEProblemNames[26] = k00_10, pODEProblemNames[27] = k12_13, pODEProblemNames[28] = k01_11, pODEProblemNames[29] = k02_03, pODEProblemNames[30] = k00_01, pODEProblemNames[31] = k03_02, pODEProblemNames[32] = k23_13, pODEProblemNames[33] = k32_31, pODEProblemNames[34] = default, pODEProblemNames[35] = k13_03, pODEProblemNames[36] = k31_30, pODEProblemNames[37] = k12_02, pODEProblemNames[38] = k31_32, pODEProblemNames[39] = k01_02, pODEProblemNames[40] = k13_23, pODEProblemNames[41] = k21_22, pODEProblemNames[42] = k12_22, pODEProblemNames[43] = k11_21, pODEProblemNames[44] = k22_23, pODEProblemNames[45] = k10_20, pODEProblemNames[46] = inflowp, pODEProblemNames[47] = k32_22
#

function compute_h(u::AbstractVector, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                   θ_nonDynamic::AbstractVector, parameterInfo::ParametersInfo, observableId::Symbol,
                      parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :observable_K27me0K36me0 
		return u[1] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId === :observable_K27me0K36me1 
		return u[4] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId === :observable_K27me0K36me2 
		return u[6] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId === :observable_K27me0K36me3 
		return u[12] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId === :observable_K27me1K36me0 
		return u[15] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId === :observable_K27me1K36me1 
		return u[9] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId === :observable_K27me1K36me2 
		return u[7] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId === :observable_K27me1K36me3 
		return u[10] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId === :observable_K27me2K36me0 
		return u[3] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId === :observable_K27me2K36me1 
		return u[5] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId === :observable_K27me2K36me2 
		return u[11] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId === :observable_K27me2K36me3 
		return u[2] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId === :observable_K27me3K36me0 
		return u[14] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId === :observable_K27me3K36me1 
		return u[13] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId === :observable_K27me3K36me2 
		return u[8] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

end

function compute_u0!(u0::AbstractVector, pODEProblem::AbstractVector) 

	#pODEProblem[1] = k20_10, pODEProblem[2] = k11_12, pODEProblem[3] = k22_32, pODEProblem[4] = k21_11, pODEProblem[5] = k13_12, pODEProblem[6] = k01_00, pODEProblem[7] = k11_01, pODEProblem[8] = k22_21, pODEProblem[9] = k22_12, pODEProblem[10] = k11_10, pODEProblem[11] = k12_11, pODEProblem[12] = k21_31, pODEProblem[13] = k02_12, pODEProblem[14] = k30_20, pODEProblem[15] = dilution, pODEProblem[16] = k31_21, pODEProblem[17] = k23_22, pODEProblem[18] = k02_01, pODEProblem[19] = k03_13, pODEProblem[20] = k10_11, pODEProblem[21] = k21_20, pODEProblem[22] = k10_00, pODEProblem[23] = k30_31, pODEProblem[24] = k20_30, pODEProblem[25] = k20_21, pODEProblem[26] = k00_10, pODEProblem[27] = k12_13, pODEProblem[28] = k01_11, pODEProblem[29] = k02_03, pODEProblem[30] = k00_01, pODEProblem[31] = k03_02, pODEProblem[32] = k23_13, pODEProblem[33] = k32_31, pODEProblem[34] = default, pODEProblem[35] = k13_03, pODEProblem[36] = k31_30, pODEProblem[37] = k12_02, pODEProblem[38] = k31_32, pODEProblem[39] = k01_02, pODEProblem[40] = k13_23, pODEProblem[41] = k21_22, pODEProblem[42] = k12_22, pODEProblem[43] = k11_21, pODEProblem[44] = k22_23, pODEProblem[45] = k10_20, pODEProblem[46] = inflowp, pODEProblem[47] = k32_22

	K27me0K36me0 = 0.00417724976345759 
	K27me2K36me3 = 0.00471831436002134 
	K27me2K36me0 = 0.00632744816295157 
	K27me0K36me1 = 0.0102104668587641 
	K27me2K36me1 = 0.0143896310177379 
	K27me0K36me2 = 0.169690316239546 
	K27me1K36me2 = 0.594249755169037 
	K27me3K36me2 = 0.00136041631795562 
	K27me1K36me1 = 0.0078328187288069 
	K27me1K36me3 = 0.102748675077958 
	K27me2K36me2 = 0.0263372634996529 
	K27me0K36me3 = 0.0504935214807544 
	K27me3K36me1 = 0.00250831034920277 
	K27me3K36me0 = 0.00330168411604165 
	K27me1K36me0 = 0.00165412810279407 

	u0 .= K27me0K36me0, K27me2K36me3, K27me2K36me0, K27me0K36me1, K27me2K36me1, K27me0K36me2, K27me1K36me2, K27me3K36me2, K27me1K36me1, K27me1K36me3, K27me2K36me2, K27me0K36me3, K27me3K36me1, K27me3K36me0, K27me1K36me0
end

function compute_u0(pODEProblem::AbstractVector)::AbstractVector 

	#pODEProblem[1] = k20_10, pODEProblem[2] = k11_12, pODEProblem[3] = k22_32, pODEProblem[4] = k21_11, pODEProblem[5] = k13_12, pODEProblem[6] = k01_00, pODEProblem[7] = k11_01, pODEProblem[8] = k22_21, pODEProblem[9] = k22_12, pODEProblem[10] = k11_10, pODEProblem[11] = k12_11, pODEProblem[12] = k21_31, pODEProblem[13] = k02_12, pODEProblem[14] = k30_20, pODEProblem[15] = dilution, pODEProblem[16] = k31_21, pODEProblem[17] = k23_22, pODEProblem[18] = k02_01, pODEProblem[19] = k03_13, pODEProblem[20] = k10_11, pODEProblem[21] = k21_20, pODEProblem[22] = k10_00, pODEProblem[23] = k30_31, pODEProblem[24] = k20_30, pODEProblem[25] = k20_21, pODEProblem[26] = k00_10, pODEProblem[27] = k12_13, pODEProblem[28] = k01_11, pODEProblem[29] = k02_03, pODEProblem[30] = k00_01, pODEProblem[31] = k03_02, pODEProblem[32] = k23_13, pODEProblem[33] = k32_31, pODEProblem[34] = default, pODEProblem[35] = k13_03, pODEProblem[36] = k31_30, pODEProblem[37] = k12_02, pODEProblem[38] = k31_32, pODEProblem[39] = k01_02, pODEProblem[40] = k13_23, pODEProblem[41] = k21_22, pODEProblem[42] = k12_22, pODEProblem[43] = k11_21, pODEProblem[44] = k22_23, pODEProblem[45] = k10_20, pODEProblem[46] = inflowp, pODEProblem[47] = k32_22

	K27me0K36me0 = 0.00417724976345759 
	K27me2K36me3 = 0.00471831436002134 
	K27me2K36me0 = 0.00632744816295157 
	K27me0K36me1 = 0.0102104668587641 
	K27me2K36me1 = 0.0143896310177379 
	K27me0K36me2 = 0.169690316239546 
	K27me1K36me2 = 0.594249755169037 
	K27me3K36me2 = 0.00136041631795562 
	K27me1K36me1 = 0.0078328187288069 
	K27me1K36me3 = 0.102748675077958 
	K27me2K36me2 = 0.0263372634996529 
	K27me0K36me3 = 0.0504935214807544 
	K27me3K36me1 = 0.00250831034920277 
	K27me3K36me0 = 0.00330168411604165 
	K27me1K36me0 = 0.00165412810279407 

	 return [K27me0K36me0, K27me2K36me3, K27me2K36me0, K27me0K36me1, K27me2K36me1, K27me0K36me2, K27me1K36me2, K27me3K36me2, K27me1K36me1, K27me1K36me3, K27me2K36me2, K27me0K36me3, K27me3K36me1, K27me3K36me0, K27me1K36me0]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                   parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :observable_K27me0K36me0 
		noiseParameter1_observable_K27me0K36me0 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_K27me0K36me0 
	end

	if observableId === :observable_K27me0K36me1 
		noiseParameter1_observable_K27me0K36me1 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_K27me0K36me1 
	end

	if observableId === :observable_K27me0K36me2 
		noiseParameter1_observable_K27me0K36me2 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_K27me0K36me2 
	end

	if observableId === :observable_K27me0K36me3 
		noiseParameter1_observable_K27me0K36me3 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_K27me0K36me3 
	end

	if observableId === :observable_K27me1K36me0 
		noiseParameter1_observable_K27me1K36me0 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_K27me1K36me0 
	end

	if observableId === :observable_K27me1K36me1 
		noiseParameter1_observable_K27me1K36me1 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_K27me1K36me1 
	end

	if observableId === :observable_K27me1K36me2 
		noiseParameter1_observable_K27me1K36me2 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_K27me1K36me2 
	end

	if observableId === :observable_K27me1K36me3 
		noiseParameter1_observable_K27me1K36me3 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_K27me1K36me3 
	end

	if observableId === :observable_K27me2K36me0 
		noiseParameter1_observable_K27me2K36me0 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_K27me2K36me0 
	end

	if observableId === :observable_K27me2K36me1 
		noiseParameter1_observable_K27me2K36me1 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_K27me2K36me1 
	end

	if observableId === :observable_K27me2K36me2 
		noiseParameter1_observable_K27me2K36me2 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_K27me2K36me2 
	end

	if observableId === :observable_K27me2K36me3 
		noiseParameter1_observable_K27me2K36me3 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_K27me2K36me3 
	end

	if observableId === :observable_K27me3K36me0 
		noiseParameter1_observable_K27me3K36me0 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_K27me3K36me0 
	end

	if observableId === :observable_K27me3K36me1 
		noiseParameter1_observable_K27me3K36me1 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_K27me3K36me1 
	end

	if observableId === :observable_K27me3K36me2 
		noiseParameter1_observable_K27me3K36me2 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_K27me3K36me2 
	end

end