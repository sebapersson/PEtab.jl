#u[1] = IPR_S, u[2] = IPR_I2, u[3] = IPR_R, u[4] = IPR_O, u[5] = IPR_I1, u[6] = IPR_A
#pODEProblemNames[1] = k_1, pODEProblemNames[2] = l_4, pODEProblemNames[3] = k_4, pODEProblemNames[4] = IP3, pODEProblemNames[5] = k4, pODEProblemNames[6] = k_2, pODEProblemNames[7] = l2, pODEProblemNames[8] = l_2, pODEProblemNames[9] = default, pODEProblemNames[10] = l_6, pODEProblemNames[11] = k1, pODEProblemNames[12] = k_3, pODEProblemNames[13] = l6, pODEProblemNames[14] = membrane, pODEProblemNames[15] = k3, pODEProblemNames[16] = l4, pODEProblemNames[17] = Ca, pODEProblemNames[18] = k2
#

function compute_h(u::AbstractVector, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                   θ_nonDynamic::AbstractVector, parameterInfo::ParametersInfo, observableId::Symbol,
                      parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :open_probability 
		return ( 0.9 * u[6] + 0.1 * u[4] ) ^ 4.0 
	end

end

function compute_u0!(u0::AbstractVector, pODEProblem::AbstractVector) 

	#pODEProblem[1] = k_1, pODEProblem[2] = l_4, pODEProblem[3] = k_4, pODEProblem[4] = IP3, pODEProblem[5] = k4, pODEProblem[6] = k_2, pODEProblem[7] = l2, pODEProblem[8] = l_2, pODEProblem[9] = default, pODEProblem[10] = l_6, pODEProblem[11] = k1, pODEProblem[12] = k_3, pODEProblem[13] = l6, pODEProblem[14] = membrane, pODEProblem[15] = k3, pODEProblem[16] = l4, pODEProblem[17] = Ca, pODEProblem[18] = k2

	IPR_S = 0.0 
	IPR_I2 = 0.0 
	IPR_R = 1.0 
	IPR_O = 0.0 
	IPR_I1 = 0.0 
	IPR_A = 0.0 

	u0 .= IPR_S, IPR_I2, IPR_R, IPR_O, IPR_I1, IPR_A
end

function compute_u0(pODEProblem::AbstractVector)::AbstractVector 

	#pODEProblem[1] = k_1, pODEProblem[2] = l_4, pODEProblem[3] = k_4, pODEProblem[4] = IP3, pODEProblem[5] = k4, pODEProblem[6] = k_2, pODEProblem[7] = l2, pODEProblem[8] = l_2, pODEProblem[9] = default, pODEProblem[10] = l_6, pODEProblem[11] = k1, pODEProblem[12] = k_3, pODEProblem[13] = l6, pODEProblem[14] = membrane, pODEProblem[15] = k3, pODEProblem[16] = l4, pODEProblem[17] = Ca, pODEProblem[18] = k2

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