#u[1] = K27me0K36me0, u[2] = K27me2K36me3, u[3] = K27me2K36me0, u[4] = K27me0K36me1, u[5] = K27me2K36me1, u[6] = K27me0K36me2, u[7] = K27me1K36me2, u[8] = K27me3K36me2, u[9] = K27me1K36me1, u[10] = K27me1K36me3, u[11] = K27me2K36me2, u[12] = K27me0K36me3, u[13] = K27me3K36me1, u[14] = K27me3K36me0, u[15] = K27me1K36me0
#p_ode_problem[1] = k20_10, p_ode_problem[2] = k11_12, p_ode_problem[3] = k22_32, p_ode_problem[4] = k21_11, p_ode_problem[5] = k13_12, p_ode_problem[6] = k01_00, p_ode_problem[7] = k11_01, p_ode_problem[8] = k22_21, p_ode_problem[9] = k22_12, p_ode_problem[10] = k11_10, p_ode_problem[11] = k12_11, p_ode_problem[12] = k21_31, p_ode_problem[13] = k02_12, p_ode_problem[14] = k30_20, p_ode_problem[15] = dilution, p_ode_problem[16] = k31_21, p_ode_problem[17] = k23_22, p_ode_problem[18] = k02_01, p_ode_problem[19] = k03_13, p_ode_problem[20] = k10_11, p_ode_problem[21] = k21_20, p_ode_problem[22] = k10_00, p_ode_problem[23] = k30_31, p_ode_problem[24] = k20_30, p_ode_problem[25] = k20_21, p_ode_problem[26] = k00_10, p_ode_problem[27] = k12_13, p_ode_problem[28] = k01_11, p_ode_problem[29] = k02_03, p_ode_problem[30] = k00_01, p_ode_problem[31] = k03_02, p_ode_problem[32] = k23_13, p_ode_problem[33] = k32_31, p_ode_problem[34] = default, p_ode_problem[35] = k13_03, p_ode_problem[36] = k31_30, p_ode_problem[37] = k12_02, p_ode_problem[38] = k31_32, p_ode_problem[39] = k01_02, p_ode_problem[40] = k13_23, p_ode_problem[41] = k21_22, p_ode_problem[42] = k12_22, p_ode_problem[43] = k11_21, p_ode_problem[44] = k22_23, p_ode_problem[45] = k10_20, p_ode_problem[46] = inflowp, p_ode_problem[47] = k32_22
#
function compute_∂h∂u!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                       θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :observable_K27me0K36me0 
		out[1] = (u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[2] = (-u[1]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[3] = (-u[1]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[4] = (-u[1]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[5] = (-u[1]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[6] = (-u[1]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[7] = (-u[1]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[8] = (-u[1]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[9] = (-u[1]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[10] = (-u[1]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[11] = (-u[1]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[12] = (-u[1]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[13] = (-u[1]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[14] = (-u[1]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[15] = (-u[1]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		return nothing
	end

	if observableId == :observable_K27me0K36me1 
		out[1] = (-u[4]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[2] = (-u[4]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[3] = (-u[4]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[4] = (u[1] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[5] = (-u[4]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[6] = (-u[4]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[7] = (-u[4]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[8] = (-u[4]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[9] = (-u[4]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[10] = (-u[4]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[11] = (-u[4]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[12] = (-u[4]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[13] = (-u[4]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[14] = (-u[4]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[15] = (-u[4]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		return nothing
	end

	if observableId == :observable_K27me0K36me2 
		out[1] = (-u[6]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[2] = (-u[6]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[3] = (-u[6]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[4] = (-u[6]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[5] = (-u[6]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[6] = (u[1] + u[4] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[7] = (-u[6]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[8] = (-u[6]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[9] = (-u[6]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[10] = (-u[6]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[11] = (-u[6]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[12] = (-u[6]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[13] = (-u[6]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[14] = (-u[6]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[15] = (-u[6]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		return nothing
	end

	if observableId == :observable_K27me0K36me3 
		out[1] = (-u[12]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[2] = (-u[12]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[3] = (-u[12]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[4] = (-u[12]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[5] = (-u[12]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[6] = (-u[12]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[7] = (-u[12]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[8] = (-u[12]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[9] = (-u[12]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[10] = (-u[12]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[11] = (-u[12]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[12] = (u[1] + u[4] + u[6] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[13] = (-u[12]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[14] = (-u[12]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[15] = (-u[12]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		return nothing
	end

	if observableId == :observable_K27me1K36me0 
		out[1] = (-u[15]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[2] = (-u[15]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[3] = (-u[15]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[4] = (-u[15]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[5] = (-u[15]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[6] = (-u[15]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[7] = (-u[15]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[8] = (-u[15]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[9] = (-u[15]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[10] = (-u[15]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[11] = (-u[15]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[12] = (-u[15]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[13] = (-u[15]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[14] = (-u[15]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[15] = (u[1] + u[4] + u[6] + u[12] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		return nothing
	end

	if observableId == :observable_K27me1K36me1 
		out[1] = (-u[9]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[2] = (-u[9]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[3] = (-u[9]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[4] = (-u[9]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[5] = (-u[9]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[6] = (-u[9]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[7] = (-u[9]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[8] = (-u[9]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[9] = (u[1] + u[4] + u[6] + u[12] + u[15] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[10] = (-u[9]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[11] = (-u[9]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[12] = (-u[9]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[13] = (-u[9]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[14] = (-u[9]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[15] = (-u[9]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		return nothing
	end

	if observableId == :observable_K27me1K36me2 
		out[1] = (-u[7]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[2] = (-u[7]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[3] = (-u[7]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[4] = (-u[7]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[5] = (-u[7]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[6] = (-u[7]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[7] = (u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[8] = (-u[7]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[9] = (-u[7]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[10] = (-u[7]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[11] = (-u[7]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[12] = (-u[7]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[13] = (-u[7]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[14] = (-u[7]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[15] = (-u[7]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		return nothing
	end

	if observableId == :observable_K27me1K36me3 
		out[1] = (-u[10]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[2] = (-u[10]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[3] = (-u[10]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[4] = (-u[10]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[5] = (-u[10]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[6] = (-u[10]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[7] = (-u[10]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[8] = (-u[10]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[9] = (-u[10]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[10] = (u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[11] = (-u[10]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[12] = (-u[10]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[13] = (-u[10]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[14] = (-u[10]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[15] = (-u[10]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		return nothing
	end

	if observableId == :observable_K27me2K36me0 
		out[1] = (-u[3]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[2] = (-u[3]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[3] = (u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[4] = (-u[3]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[5] = (-u[3]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[6] = (-u[3]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[7] = (-u[3]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[8] = (-u[3]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[9] = (-u[3]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[10] = (-u[3]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[11] = (-u[3]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[12] = (-u[3]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[13] = (-u[3]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[14] = (-u[3]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[15] = (-u[3]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		return nothing
	end

	if observableId == :observable_K27me2K36me1 
		out[1] = (-u[5]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[2] = (-u[5]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[3] = (-u[5]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[4] = (-u[5]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[5] = (u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[11] + u[2] + u[14] + u[13] + u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[6] = (-u[5]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[7] = (-u[5]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[8] = (-u[5]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[9] = (-u[5]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[10] = (-u[5]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[11] = (-u[5]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[12] = (-u[5]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[13] = (-u[5]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[14] = (-u[5]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[15] = (-u[5]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		return nothing
	end

	if observableId == :observable_K27me2K36me2 
		out[1] = (-u[11]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[2] = (-u[11]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[3] = (-u[11]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[4] = (-u[11]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[5] = (-u[11]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[6] = (-u[11]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[7] = (-u[11]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[8] = (-u[11]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[9] = (-u[11]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[10] = (-u[11]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[11] = (u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[2] + u[14] + u[13] + u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[12] = (-u[11]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[13] = (-u[11]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[14] = (-u[11]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[15] = (-u[11]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		return nothing
	end

	if observableId == :observable_K27me2K36me3 
		out[1] = (-u[2]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[2] = (u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[14] + u[13] + u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[3] = (-u[2]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[4] = (-u[2]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[5] = (-u[2]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[6] = (-u[2]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[7] = (-u[2]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[8] = (-u[2]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[9] = (-u[2]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[10] = (-u[2]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[11] = (-u[2]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[12] = (-u[2]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[13] = (-u[2]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[14] = (-u[2]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[15] = (-u[2]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		return nothing
	end

	if observableId == :observable_K27me3K36me0 
		out[1] = (-u[14]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[2] = (-u[14]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[3] = (-u[14]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[4] = (-u[14]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[5] = (-u[14]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[6] = (-u[14]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[7] = (-u[14]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[8] = (-u[14]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[9] = (-u[14]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[10] = (-u[14]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[11] = (-u[14]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[12] = (-u[14]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[13] = (-u[14]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[14] = (u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[13] + u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[15] = (-u[14]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		return nothing
	end

	if observableId == :observable_K27me3K36me1 
		out[1] = (-u[13]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[2] = (-u[13]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[3] = (-u[13]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[4] = (-u[13]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[5] = (-u[13]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[6] = (-u[13]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[7] = (-u[13]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[8] = (-u[13]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[9] = (-u[13]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[10] = (-u[13]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[11] = (-u[13]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[12] = (-u[13]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[13] = (u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[14] = (-u[13]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[15] = (-u[13]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		return nothing
	end

	if observableId == :observable_K27me3K36me2 
		out[1] = (-u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[2] = (-u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[3] = (-u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[4] = (-u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[5] = (-u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[6] = (-u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[7] = (-u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[8] = (u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[9] = (-u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[10] = (-u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[11] = (-u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[12] = (-u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[13] = (-u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[14] = (-u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		out[15] = (-u[8]) / ((u[1] + u[4] + u[6] + u[12] + u[15] + u[9] + u[7] + u[10] + u[3] + u[5] + u[11] + u[2] + u[14] + u[13] + u[8])^2)
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                       θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :observable_K27me0K36me0 
		return nothing
	end

	if observableId == :observable_K27me0K36me1 
		return nothing
	end

	if observableId == :observable_K27me0K36me2 
		return nothing
	end

	if observableId == :observable_K27me0K36me3 
		return nothing
	end

	if observableId == :observable_K27me1K36me0 
		return nothing
	end

	if observableId == :observable_K27me1K36me1 
		return nothing
	end

	if observableId == :observable_K27me1K36me2 
		return nothing
	end

	if observableId == :observable_K27me1K36me3 
		return nothing
	end

	if observableId == :observable_K27me2K36me0 
		return nothing
	end

	if observableId == :observable_K27me2K36me1 
		return nothing
	end

	if observableId == :observable_K27me2K36me2 
		return nothing
	end

	if observableId == :observable_K27me2K36me3 
		return nothing
	end

	if observableId == :observable_K27me3K36me0 
		return nothing
	end

	if observableId == :observable_K27me3K36me1 
		return nothing
	end

	if observableId == :observable_K27me3K36me2 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                        parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :observable_K27me0K36me0 
		return nothing
	end

	if observableId == :observable_K27me0K36me1 
		return nothing
	end

	if observableId == :observable_K27me0K36me2 
		return nothing
	end

	if observableId == :observable_K27me0K36me3 
		return nothing
	end

	if observableId == :observable_K27me1K36me0 
		return nothing
	end

	if observableId == :observable_K27me1K36me1 
		return nothing
	end

	if observableId == :observable_K27me1K36me2 
		return nothing
	end

	if observableId == :observable_K27me1K36me3 
		return nothing
	end

	if observableId == :observable_K27me2K36me0 
		return nothing
	end

	if observableId == :observable_K27me2K36me1 
		return nothing
	end

	if observableId == :observable_K27me2K36me2 
		return nothing
	end

	if observableId == :observable_K27me2K36me3 
		return nothing
	end

	if observableId == :observable_K27me3K36me0 
		return nothing
	end

	if observableId == :observable_K27me3K36me1 
		return nothing
	end

	if observableId == :observable_K27me3K36me2 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                        parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :observable_K27me0K36me0 
		return nothing
	end

	if observableId == :observable_K27me0K36me1 
		return nothing
	end

	if observableId == :observable_K27me0K36me2 
		return nothing
	end

	if observableId == :observable_K27me0K36me3 
		return nothing
	end

	if observableId == :observable_K27me1K36me0 
		return nothing
	end

	if observableId == :observable_K27me1K36me1 
		return nothing
	end

	if observableId == :observable_K27me1K36me2 
		return nothing
	end

	if observableId == :observable_K27me1K36me3 
		return nothing
	end

	if observableId == :observable_K27me2K36me0 
		return nothing
	end

	if observableId == :observable_K27me2K36me1 
		return nothing
	end

	if observableId == :observable_K27me2K36me2 
		return nothing
	end

	if observableId == :observable_K27me2K36me3 
		return nothing
	end

	if observableId == :observable_K27me3K36me0 
		return nothing
	end

	if observableId == :observable_K27me3K36me1 
		return nothing
	end

	if observableId == :observable_K27me3K36me2 
		return nothing
	end

end

