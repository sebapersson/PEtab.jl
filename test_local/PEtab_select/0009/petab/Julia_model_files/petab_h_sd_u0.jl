#u[1] = x_k05k16, u[2] = x_k05k08k16, u[3] = x_k05k12k16, u[4] = x_k16, u[5] = x_0ac, u[6] = x_k12, u[7] = x_k12k16, u[8] = x_k05k12, u[9] = x_k08, u[10] = x_k05k08, u[11] = x_k08k16, u[12] = x_k08k12, u[13] = x_k05k08k12, u[14] = x_k05, u[15] = x_4ac, u[16] = x_k08k12k16
#pODEProblemNames[1] = a_k05_k05k12, pODEProblemNames[2] = a_0ac_k16, pODEProblemNames[3] = a_k08_k08k12, pODEProblemNames[4] = a_k16_k08k16, pODEProblemNames[5] = a_k05k08k16_4ac, pODEProblemNames[6] = a_k05k16_k05k08k16, pODEProblemNames[7] = a_k05k16_k05k12k16, pODEProblemNames[8] = a_k05k08_k05k08k16, pODEProblemNames[9] = a_k08k12_k08k12k16, pODEProblemNames[10] = a_k12k16_k05k12k16, pODEProblemNames[11] = a_k12_k12k16, pODEProblemNames[12] = a_0ac_k12, pODEProblemNames[13] = a_b, pODEProblemNames[14] = a_k05_k05k08, pODEProblemNames[15] = a_k08k16_k05k08k16, pODEProblemNames[16] = a_k08k12k16_4ac, pODEProblemNames[17] = a_k05k08k12_4ac, pODEProblemNames[18] = a_k05k08_k05k08k12, pODEProblemNames[19] = a_k08k16_k08k12k16, pODEProblemNames[20] = a_k05k12k16_4ac, pODEProblemNames[21] = a_k05k12_k05k12k16, pODEProblemNames[22] = a_k08_k05k08, pODEProblemNames[23] = a_k08_k08k16, pODEProblemNames[24] = a_k05k12_k05k08k12, pODEProblemNames[25] = a_k12k16_k08k12k16, pODEProblemNames[26] = a_k12_k05k12, pODEProblemNames[27] = da_b, pODEProblemNames[28] = compartment, pODEProblemNames[29] = a_0ac_k08, pODEProblemNames[30] = a_0ac_k05, pODEProblemNames[31] = a_k16_k05k16, pODEProblemNames[32] = a_k08k12_k05k08k12, pODEProblemNames[33] = a_k05_k05k16, pODEProblemNames[34] = a_k12_k08k12, pODEProblemNames[35] = a_k16_k12k16
##parameterInfo.nominalValue[1] = a_0ac_k05_C 
#parameterInfo.nominalValue[2] = a_0ac_k08_C 
#parameterInfo.nominalValue[3] = a_0ac_k12_C 
#parameterInfo.nominalValue[4] = a_0ac_k16_C 
#parameterInfo.nominalValue[5] = a_k05_k05k08_C 
#parameterInfo.nominalValue[6] = a_k05_k05k12_C 
#parameterInfo.nominalValue[7] = a_k05_k05k16_C 
#parameterInfo.nominalValue[9] = a_k08_k08k12_C 
#parameterInfo.nominalValue[10] = a_k08_k08k16_C 
#parameterInfo.nominalValue[11] = a_k12_k05k12_C 
#parameterInfo.nominalValue[12] = a_k12_k08k12_C 
#parameterInfo.nominalValue[13] = a_k12_k12k16_C 
#parameterInfo.nominalValue[14] = a_k16_k05k16_C 
#parameterInfo.nominalValue[15] = a_k16_k08k16_C 
#parameterInfo.nominalValue[16] = a_k16_k12k16_C 
#parameterInfo.nominalValue[17] = a_k05k08_k05k08k12_C 
#parameterInfo.nominalValue[18] = a_k05k08_k05k08k16_C 
#parameterInfo.nominalValue[19] = a_k05k12_k05k08k12_C 
#parameterInfo.nominalValue[20] = a_k05k12_k05k12k16_C 
#parameterInfo.nominalValue[21] = a_k05k16_k05k08k16_C 
#parameterInfo.nominalValue[22] = a_k05k16_k05k12k16_C 
#parameterInfo.nominalValue[23] = a_k08k12_k05k08k12_C 
#parameterInfo.nominalValue[24] = a_k08k12_k08k12k16_C 
#parameterInfo.nominalValue[25] = a_k08k16_k05k08k16_C 
#parameterInfo.nominalValue[26] = a_k08k16_k08k12k16_C 
#parameterInfo.nominalValue[27] = a_k12k16_k05k12k16_C 
#parameterInfo.nominalValue[28] = a_k12k16_k08k12k16_C 
#parameterInfo.nominalValue[29] = a_k05k08k12_4ac_C 
#parameterInfo.nominalValue[30] = a_k05k08k16_4ac_C 
#parameterInfo.nominalValue[31] = a_k05k12k16_4ac_C 
#parameterInfo.nominalValue[32] = a_k08k12k16_4ac_C 
#parameterInfo.nominalValue[34] = da_b_C 
#parameterInfo.nominalValue[35] = sigma__C 


function compute_h(u::AbstractVector, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                   θ_nonDynamic::AbstractVector, parameterInfo::ParametersInfo, observableId::Symbol,
                      parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :observable_x_0ac 
		return u[5] 
	end

	if observableId === :observable_x_4ac 
		return u[15] 
	end

	if observableId === :observable_x_k12 
		return u[6] 
	end

	if observableId === :observable_x_k12k16 
		return u[7] 
	end

	if observableId === :observable_x_k16 
		return u[4] 
	end

	if observableId === :observable_x_k05 
		return u[14] 
	end

	if observableId === :observable_x_k05k12 
		return u[8] 
	end

	if observableId === :observable_x_k05k12k16 
		return u[3] 
	end

	if observableId === :observable_x_k05k08 
		return u[10] 
	end

	if observableId === :observable_x_k05k08k12 
		return u[13] 
	end

	if observableId === :observable_x_k05k08k16 
		return u[2] 
	end

	if observableId === :observable_x_k08 
		return u[9] 
	end

	if observableId === :observable_x_k08k12 
		return u[12] 
	end

	if observableId === :observable_x_k08k12k16 
		return u[16] 
	end

	if observableId === :observable_x_k08k16 
		return u[11] 
	end

end

function compute_u0!(u0::AbstractVector, pODEProblem::AbstractVector) 

	#pODEProblem[1] = a_k05_k05k12, pODEProblem[2] = a_0ac_k16, pODEProblem[3] = a_k08_k08k12, pODEProblem[4] = a_k16_k08k16, pODEProblem[5] = a_k05k08k16_4ac, pODEProblem[6] = a_k05k16_k05k08k16, pODEProblem[7] = a_k05k16_k05k12k16, pODEProblem[8] = a_k05k08_k05k08k16, pODEProblem[9] = a_k08k12_k08k12k16, pODEProblem[10] = a_k12k16_k05k12k16, pODEProblem[11] = a_k12_k12k16, pODEProblem[12] = a_0ac_k12, pODEProblem[13] = a_b, pODEProblem[14] = a_k05_k05k08, pODEProblem[15] = a_k08k16_k05k08k16, pODEProblem[16] = a_k08k12k16_4ac, pODEProblem[17] = a_k05k08k12_4ac, pODEProblem[18] = a_k05k08_k05k08k12, pODEProblem[19] = a_k08k16_k08k12k16, pODEProblem[20] = a_k05k12k16_4ac, pODEProblem[21] = a_k05k12_k05k12k16, pODEProblem[22] = a_k08_k05k08, pODEProblem[23] = a_k08_k08k16, pODEProblem[24] = a_k05k12_k05k08k12, pODEProblem[25] = a_k12k16_k08k12k16, pODEProblem[26] = a_k12_k05k12, pODEProblem[27] = da_b, pODEProblem[28] = compartment, pODEProblem[29] = a_0ac_k08, pODEProblem[30] = a_0ac_k05, pODEProblem[31] = a_k16_k05k16, pODEProblem[32] = a_k08k12_k05k08k12, pODEProblem[33] = a_k05_k05k16, pODEProblem[34] = a_k12_k08k12, pODEProblem[35] = a_k16_k12k16

	x_k05k16 = 0.0 
	x_k05k08k16 = 0.0 
	x_k05k12k16 = 0.0 
	x_k16 = 0.0 
	x_0ac = 1.0 
	x_k12 = 0.0 
	x_k12k16 = 0.0 
	x_k05k12 = 0.0 
	x_k08 = 0.0 
	x_k05k08 = 0.0 
	x_k08k16 = 0.0 
	x_k08k12 = 0.0 
	x_k05k08k12 = 0.0 
	x_k05 = 0.0 
	x_4ac = 0.0 
	x_k08k12k16 = 0.0 

	u0 .= x_k05k16, x_k05k08k16, x_k05k12k16, x_k16, x_0ac, x_k12, x_k12k16, x_k05k12, x_k08, x_k05k08, x_k08k16, x_k08k12, x_k05k08k12, x_k05, x_4ac, x_k08k12k16
end

function compute_u0(pODEProblem::AbstractVector)::AbstractVector 

	#pODEProblem[1] = a_k05_k05k12, pODEProblem[2] = a_0ac_k16, pODEProblem[3] = a_k08_k08k12, pODEProblem[4] = a_k16_k08k16, pODEProblem[5] = a_k05k08k16_4ac, pODEProblem[6] = a_k05k16_k05k08k16, pODEProblem[7] = a_k05k16_k05k12k16, pODEProblem[8] = a_k05k08_k05k08k16, pODEProblem[9] = a_k08k12_k08k12k16, pODEProblem[10] = a_k12k16_k05k12k16, pODEProblem[11] = a_k12_k12k16, pODEProblem[12] = a_0ac_k12, pODEProblem[13] = a_b, pODEProblem[14] = a_k05_k05k08, pODEProblem[15] = a_k08k16_k05k08k16, pODEProblem[16] = a_k08k12k16_4ac, pODEProblem[17] = a_k05k08k12_4ac, pODEProblem[18] = a_k05k08_k05k08k12, pODEProblem[19] = a_k08k16_k08k12k16, pODEProblem[20] = a_k05k12k16_4ac, pODEProblem[21] = a_k05k12_k05k12k16, pODEProblem[22] = a_k08_k05k08, pODEProblem[23] = a_k08_k08k16, pODEProblem[24] = a_k05k12_k05k08k12, pODEProblem[25] = a_k12k16_k08k12k16, pODEProblem[26] = a_k12_k05k12, pODEProblem[27] = da_b, pODEProblem[28] = compartment, pODEProblem[29] = a_0ac_k08, pODEProblem[30] = a_0ac_k05, pODEProblem[31] = a_k16_k05k16, pODEProblem[32] = a_k08k12_k05k08k12, pODEProblem[33] = a_k05_k05k16, pODEProblem[34] = a_k12_k08k12, pODEProblem[35] = a_k16_k12k16

	x_k05k16 = 0.0 
	x_k05k08k16 = 0.0 
	x_k05k12k16 = 0.0 
	x_k16 = 0.0 
	x_0ac = 1.0 
	x_k12 = 0.0 
	x_k12k16 = 0.0 
	x_k05k12 = 0.0 
	x_k08 = 0.0 
	x_k05k08 = 0.0 
	x_k08k16 = 0.0 
	x_k08k12 = 0.0 
	x_k05k08k12 = 0.0 
	x_k05 = 0.0 
	x_4ac = 0.0 
	x_k08k12k16 = 0.0 

	 return [x_k05k16, x_k05k08k16, x_k05k12k16, x_k16, x_0ac, x_k12, x_k12k16, x_k05k12, x_k08, x_k05k08, x_k08k16, x_k08k12, x_k05k08k12, x_k05, x_4ac, x_k08k12k16]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                   parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :observable_x_0ac 
		return parameterInfo.nominalValue[35] 
	end

	if observableId === :observable_x_4ac 
		return parameterInfo.nominalValue[35] 
	end

	if observableId === :observable_x_k12 
		return parameterInfo.nominalValue[35] 
	end

	if observableId === :observable_x_k12k16 
		return parameterInfo.nominalValue[35] 
	end

	if observableId === :observable_x_k16 
		return parameterInfo.nominalValue[35] 
	end

	if observableId === :observable_x_k05 
		return parameterInfo.nominalValue[35] 
	end

	if observableId === :observable_x_k05k12 
		return parameterInfo.nominalValue[35] 
	end

	if observableId === :observable_x_k05k12k16 
		return parameterInfo.nominalValue[35] 
	end

	if observableId === :observable_x_k05k08 
		return parameterInfo.nominalValue[35] 
	end

	if observableId === :observable_x_k05k08k12 
		return parameterInfo.nominalValue[35] 
	end

	if observableId === :observable_x_k05k08k16 
		return parameterInfo.nominalValue[35] 
	end

	if observableId === :observable_x_k08 
		return parameterInfo.nominalValue[35] 
	end

	if observableId === :observable_x_k08k12 
		return parameterInfo.nominalValue[35] 
	end

	if observableId === :observable_x_k08k12k16 
		return parameterInfo.nominalValue[35] 
	end

	if observableId === :observable_x_k08k16 
		return parameterInfo.nominalValue[35] 
	end

end