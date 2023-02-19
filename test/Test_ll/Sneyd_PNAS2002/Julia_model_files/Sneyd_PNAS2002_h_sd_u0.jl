#u[1] = IPR_S, u[2] = IPR_I2, u[3] = IPR_R, u[4] = IPR_O, u[5] = IPR_I1, u[6] = IPR_A
#pODEProblemNames[1] = l_4, pODEProblemNames[2] = k_4, pODEProblemNames[3] = IP3, pODEProblemNames[4] = k4, pODEProblemNames[5] = k_2, pODEProblemNames[6] = l2, pODEProblemNames[7] = l_2, pODEProblemNames[8] = default, pODEProblemNames[9] = l_6, pODEProblemNames[10] = k1, pODEProblemNames[11] = k_3, pODEProblemNames[12] = l6, pODEProblemNames[13] = membrane, pODEProblemNames[14] = k3, pODEProblemNames[15] = l4, pODEProblemNames[16] = Ca, pODEProblemNames[17] = k2, pODEProblemNames[18] = k_1
#

function compute_h(u::AbstractVector, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                   θ_nonDynamic::AbstractVector, parameterInfo::ParametersInfo, observableId::Symbol,
                      parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :open_probability 
		return ( 0.9 * u[6] + 0.1 * u[4] ) ^ 4 
	end

end

function compute_u0!(u0::AbstractVector, pODEProblem::AbstractVector) 

	#pODEProblem[1] = l_4, pODEProblem[2] = k_4, pODEProblem[3] = IP3, pODEProblem[4] = k4, pODEProblem[5] = k_2, pODEProblem[6] = l2, pODEProblem[7] = l_2, pODEProblem[8] = default, pODEProblem[9] = l_6, pODEProblem[10] = k1, pODEProblem[11] = k_3, pODEProblem[12] = l6, pODEProblem[13] = membrane, pODEProblem[14] = k3, pODEProblem[15] = l4, pODEProblem[16] = Ca, pODEProblem[17] = k2, pODEProblem[18] = k_1

	IPR_S = 0.0 
	IPR_I2 = 0.0 
	IPR_R = 1.0 
	IPR_O = 0.0 
	IPR_I1 = 0.0 
	IPR_A = 0.0 

	u0 .= IPR_S, IPR_I2, IPR_R, IPR_O, IPR_I1, IPR_A
end

function compute_u0(pODEProblem::AbstractVector)::AbstractVector 

	#pODEProblem[1] = l_4, pODEProblem[2] = k_4, pODEProblem[3] = IP3, pODEProblem[4] = k4, pODEProblem[5] = k_2, pODEProblem[6] = l2, pODEProblem[7] = l_2, pODEProblem[8] = default, pODEProblem[9] = l_6, pODEProblem[10] = k1, pODEProblem[11] = k_3, pODEProblem[12] = l6, pODEProblem[13] = membrane, pODEProblem[14] = k3, pODEProblem[15] = l4, pODEProblem[16] = Ca, pODEProblem[17] = k2, pODEProblem[18] = k_1

	IPR_S = 0.0 
	IPR_I2 = 0.0 
	IPR_R = 1.0 
	IPR_O = 0.0 
	IPR_I1 = 0.0 
	IPR_A = 0.0 

	 return [IPR_S, IPR_I2, IPR_R, IPR_O, IPR_I1, IPR_A]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                   parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :open_probability 
		noiseParameter1_open_probability = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_open_probability 
	end

end