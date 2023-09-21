#u[1] = K27me0K36me0, u[2] = K27me2K36me3, u[3] = K27me2K36me0, u[4] = K27me0K36me1, u[5] = K27me2K36me1, u[6] = K27me0K36me2, u[7] = K27me1K36me2, u[8] = K27me3K36me2, u[9] = K27me1K36me1, u[10] = K27me1K36me3, u[11] = K27me2K36me2, u[12] = K27me0K36me3, u[13] = K27me3K36me1, u[14] = K27me3K36me0, u[15] = K27me1K36me0
#p_ode_problem_names[1] = k20_10, p_ode_problem_names[2] = k11_12, p_ode_problem_names[3] = k22_32, p_ode_problem_names[4] = k21_11, p_ode_problem_names[5] = k13_12, p_ode_problem_names[6] = k01_00, p_ode_problem_names[7] = k11_01, p_ode_problem_names[8] = k22_21, p_ode_problem_names[9] = k22_12, p_ode_problem_names[10] = k11_10, p_ode_problem_names[11] = k12_11, p_ode_problem_names[12] = k21_31, p_ode_problem_names[13] = k02_12, p_ode_problem_names[14] = k30_20, p_ode_problem_names[15] = dilution, p_ode_problem_names[16] = k31_21, p_ode_problem_names[17] = k23_22, p_ode_problem_names[18] = k02_01, p_ode_problem_names[19] = k03_13, p_ode_problem_names[20] = k10_11, p_ode_problem_names[21] = k21_20, p_ode_problem_names[22] = k10_00, p_ode_problem_names[23] = k30_31, p_ode_problem_names[24] = k20_30, p_ode_problem_names[25] = k20_21, p_ode_problem_names[26] = k00_10, p_ode_problem_names[27] = k12_13, p_ode_problem_names[28] = k01_11, p_ode_problem_names[29] = k02_03, p_ode_problem_names[30] = k00_01, p_ode_problem_names[31] = k03_02, p_ode_problem_names[32] = k23_13, p_ode_problem_names[33] = k32_31, p_ode_problem_names[34] = default, p_ode_problem_names[35] = k13_03, p_ode_problem_names[36] = k31_30, p_ode_problem_names[37] = k12_02, p_ode_problem_names[38] = k31_32, p_ode_problem_names[39] = k01_02, p_ode_problem_names[40] = k13_23, p_ode_problem_names[41] = k21_22, p_ode_problem_names[42] = k12_22, p_ode_problem_names[43] = k11_21, p_ode_problem_names[44] = k22_23, p_ode_problem_names[45] = k10_20, p_ode_problem_names[46] = inflowp, p_ode_problem_names[47] = k32_22
#

function compute_h(u::AbstractVector, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                   θ_non_dynamic::AbstractVector, parameter_info::ParametersInfo, observableId::Symbol, 
                      parameter_map::θObsOrSdParameterMap)::Real 
	if observableId == :observable_K27me0K36me0 
		return u[1] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId == :observable_K27me0K36me1 
		return u[4] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId == :observable_K27me0K36me2 
		return u[6] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId == :observable_K27me0K36me3 
		return u[12] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId == :observable_K27me1K36me0 
		return u[15] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId == :observable_K27me1K36me1 
		return u[9] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId == :observable_K27me1K36me2 
		return u[7] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId == :observable_K27me1K36me3 
		return u[10] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId == :observable_K27me2K36me0 
		return u[3] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId == :observable_K27me2K36me1 
		return u[5] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId == :observable_K27me2K36me2 
		return u[11] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId == :observable_K27me2K36me3 
		return u[2] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId == :observable_K27me3K36me0 
		return u[14] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId == :observable_K27me3K36me1 
		return u[13] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

	if observableId == :observable_K27me3K36me2 
		return u[8] / ( u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8] ) 
	end

end

function compute_u0!(u0::AbstractVector, p_ode_problem::AbstractVector) 

	#p_ode_problem[1] = k20_10, p_ode_problem[2] = k11_12, p_ode_problem[3] = k22_32, p_ode_problem[4] = k21_11, p_ode_problem[5] = k13_12, p_ode_problem[6] = k01_00, p_ode_problem[7] = k11_01, p_ode_problem[8] = k22_21, p_ode_problem[9] = k22_12, p_ode_problem[10] = k11_10, p_ode_problem[11] = k12_11, p_ode_problem[12] = k21_31, p_ode_problem[13] = k02_12, p_ode_problem[14] = k30_20, p_ode_problem[15] = dilution, p_ode_problem[16] = k31_21, p_ode_problem[17] = k23_22, p_ode_problem[18] = k02_01, p_ode_problem[19] = k03_13, p_ode_problem[20] = k10_11, p_ode_problem[21] = k21_20, p_ode_problem[22] = k10_00, p_ode_problem[23] = k30_31, p_ode_problem[24] = k20_30, p_ode_problem[25] = k20_21, p_ode_problem[26] = k00_10, p_ode_problem[27] = k12_13, p_ode_problem[28] = k01_11, p_ode_problem[29] = k02_03, p_ode_problem[30] = k00_01, p_ode_problem[31] = k03_02, p_ode_problem[32] = k23_13, p_ode_problem[33] = k32_31, p_ode_problem[34] = default, p_ode_problem[35] = k13_03, p_ode_problem[36] = k31_30, p_ode_problem[37] = k12_02, p_ode_problem[38] = k31_32, p_ode_problem[39] = k01_02, p_ode_problem[40] = k13_23, p_ode_problem[41] = k21_22, p_ode_problem[42] = k12_22, p_ode_problem[43] = k11_21, p_ode_problem[44] = k22_23, p_ode_problem[45] = k10_20, p_ode_problem[46] = inflowp, p_ode_problem[47] = k32_22

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

function compute_u0(p_ode_problem::AbstractVector)::AbstractVector 

	#p_ode_problem[1] = k20_10, p_ode_problem[2] = k11_12, p_ode_problem[3] = k22_32, p_ode_problem[4] = k21_11, p_ode_problem[5] = k13_12, p_ode_problem[6] = k01_00, p_ode_problem[7] = k11_01, p_ode_problem[8] = k22_21, p_ode_problem[9] = k22_12, p_ode_problem[10] = k11_10, p_ode_problem[11] = k12_11, p_ode_problem[12] = k21_31, p_ode_problem[13] = k02_12, p_ode_problem[14] = k30_20, p_ode_problem[15] = dilution, p_ode_problem[16] = k31_21, p_ode_problem[17] = k23_22, p_ode_problem[18] = k02_01, p_ode_problem[19] = k03_13, p_ode_problem[20] = k10_11, p_ode_problem[21] = k21_20, p_ode_problem[22] = k10_00, p_ode_problem[23] = k30_31, p_ode_problem[24] = k20_30, p_ode_problem[25] = k20_21, p_ode_problem[26] = k00_10, p_ode_problem[27] = k12_13, p_ode_problem[28] = k01_11, p_ode_problem[29] = k02_03, p_ode_problem[30] = k00_01, p_ode_problem[31] = k03_02, p_ode_problem[32] = k23_13, p_ode_problem[33] = k32_31, p_ode_problem[34] = default, p_ode_problem[35] = k13_03, p_ode_problem[36] = k31_30, p_ode_problem[37] = k12_02, p_ode_problem[38] = k31_32, p_ode_problem[39] = k01_02, p_ode_problem[40] = k13_23, p_ode_problem[41] = k21_22, p_ode_problem[42] = k12_22, p_ode_problem[43] = k11_21, p_ode_problem[44] = k22_23, p_ode_problem[45] = k10_20, p_ode_problem[46] = inflowp, p_ode_problem[47] = k32_22

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

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector, 
                   parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap)::Real 
	if observableId == :observable_K27me0K36me0 
		noiseParameter1_observable_K27me0K36me0 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_K27me0K36me0 
	end

	if observableId == :observable_K27me0K36me1 
		noiseParameter1_observable_K27me0K36me1 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_K27me0K36me1 
	end

	if observableId == :observable_K27me0K36me2 
		noiseParameter1_observable_K27me0K36me2 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_K27me0K36me2 
	end

	if observableId == :observable_K27me0K36me3 
		noiseParameter1_observable_K27me0K36me3 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_K27me0K36me3 
	end

	if observableId == :observable_K27me1K36me0 
		noiseParameter1_observable_K27me1K36me0 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_K27me1K36me0 
	end

	if observableId == :observable_K27me1K36me1 
		noiseParameter1_observable_K27me1K36me1 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_K27me1K36me1 
	end

	if observableId == :observable_K27me1K36me2 
		noiseParameter1_observable_K27me1K36me2 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_K27me1K36me2 
	end

	if observableId == :observable_K27me1K36me3 
		noiseParameter1_observable_K27me1K36me3 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_K27me1K36me3 
	end

	if observableId == :observable_K27me2K36me0 
		noiseParameter1_observable_K27me2K36me0 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_K27me2K36me0 
	end

	if observableId == :observable_K27me2K36me1 
		noiseParameter1_observable_K27me2K36me1 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_K27me2K36me1 
	end

	if observableId == :observable_K27me2K36me2 
		noiseParameter1_observable_K27me2K36me2 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_K27me2K36me2 
	end

	if observableId == :observable_K27me2K36me3 
		noiseParameter1_observable_K27me2K36me3 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_K27me2K36me3 
	end

	if observableId == :observable_K27me3K36me0 
		noiseParameter1_observable_K27me3K36me0 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_K27me3K36me0 
	end

	if observableId == :observable_K27me3K36me1 
		noiseParameter1_observable_K27me3K36me1 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_K27me3K36me1 
	end

	if observableId == :observable_K27me3K36me2 
		noiseParameter1_observable_K27me3K36me2 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_K27me3K36me2 
	end

end