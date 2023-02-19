#u[1] = Naive, u[2] = Pathogen, u[3] = LateEffector, u[4] = EarlyEffector, u[5] = Memory
#pODEProblemNames[1] = mu_LL, pODEProblemNames[2] = delta_NE, pODEProblemNames[3] = mu_PE, pODEProblemNames[4] = mu_P, pODEProblemNames[5] = mu_PL, pODEProblemNames[6] = delta_EL, pODEProblemNames[7] = mu_EE, pODEProblemNames[8] = default, pODEProblemNames[9] = mu_N, pODEProblemNames[10] = rho_E, pODEProblemNames[11] = delta_LM, pODEProblemNames[12] = rho_P, pODEProblemNames[13] = mu_LE
#

function compute_h(u::AbstractVector, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                   θ_nonDynamic::AbstractVector, parameterInfo::ParametersInfo, observableId::Symbol,
                      parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :observable_EarlyEffector 
		return u[4] 
	end

	if observableId === :observable_LateEffector 
		return u[3] 
	end

	if observableId === :observable_Memory 
		return u[5] 
	end

	if observableId === :observable_Naive 
		return u[1] 
	end

end

function compute_u0!(u0::AbstractVector, pODEProblem::AbstractVector) 

	#pODEProblem[1] = mu_LL, pODEProblem[2] = delta_NE, pODEProblem[3] = mu_PE, pODEProblem[4] = mu_P, pODEProblem[5] = mu_PL, pODEProblem[6] = delta_EL, pODEProblem[7] = mu_EE, pODEProblem[8] = default, pODEProblem[9] = mu_N, pODEProblem[10] = rho_E, pODEProblem[11] = delta_LM, pODEProblem[12] = rho_P, pODEProblem[13] = mu_LE

	Naive = 8090.0 
	Pathogen = 1.0 
	LateEffector = 0.0 
	EarlyEffector = 0.0 
	Memory = 0.0 

	u0 .= Naive, Pathogen, LateEffector, EarlyEffector, Memory
end

function compute_u0(pODEProblem::AbstractVector)::AbstractVector 

	#pODEProblem[1] = mu_LL, pODEProblem[2] = delta_NE, pODEProblem[3] = mu_PE, pODEProblem[4] = mu_P, pODEProblem[5] = mu_PL, pODEProblem[6] = delta_EL, pODEProblem[7] = mu_EE, pODEProblem[8] = default, pODEProblem[9] = mu_N, pODEProblem[10] = rho_E, pODEProblem[11] = delta_LM, pODEProblem[12] = rho_P, pODEProblem[13] = mu_LE

	Naive = 8090.0 
	Pathogen = 1.0 
	LateEffector = 0.0 
	EarlyEffector = 0.0 
	Memory = 0.0 

	 return [Naive, Pathogen, LateEffector, EarlyEffector, Memory]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                   parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :observable_EarlyEffector 
		noiseParameter1_observable_EarlyEffector = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_EarlyEffector 
	end

	if observableId === :observable_LateEffector 
		noiseParameter1_observable_LateEffector = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_LateEffector 
	end

	if observableId === :observable_Memory 
		noiseParameter1_observable_Memory = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_Memory 
	end

	if observableId === :observable_Naive 
		noiseParameter1_observable_Naive = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_Naive 
	end

end