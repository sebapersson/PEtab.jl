#u[1] = STAT5A, u[2] = pApA, u[3] = nucpApB, u[4] = nucpBpB, u[5] = STAT5B, u[6] = pApB, u[7] = nucpApA, u[8] = pBpB
#pODEProblemNames[1] = ratio, pODEProblemNames[2] = k_imp_homo, pODEProblemNames[3] = k_exp_hetero, pODEProblemNames[4] = cyt, pODEProblemNames[5] = k_phos, pODEProblemNames[6] = specC17, pODEProblemNames[7] = Epo_degradation_BaF3, pODEProblemNames[8] = k_exp_homo, pODEProblemNames[9] = nuc, pODEProblemNames[10] = k_imp_hetero
##parameterInfo.nominalValue[7] = ratio_C 
#parameterInfo.nominalValue[11] = specC17_C 


function compute_h(u::AbstractVector, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                   θ_nonDynamic::AbstractVector, parameterInfo::ParametersInfo, observableId::Symbol,
                      parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :pSTAT5A_rel 
		return ( 100 * u[6] + 200 * u[2] * pODEProblem[6] ) / ( u[6] + u[1] * pODEProblem[6] + 2 * u[2] * pODEProblem[6] ) 
	end

	if observableId === :pSTAT5B_rel 
		return - ( 100 * u[6] - 200 * u[8] * ( pODEProblem[6] - 1 ) ) / ( ( u[5] * ( pODEProblem[6] - 1 ) - u[6] ) + 2 * u[8] * ( pODEProblem[6] - 1 ) ) 
	end

	if observableId === :rSTAT5A_rel 
		return ( 100 * u[6] + 100 * u[1] * pODEProblem[6] + 200 * u[2] * pODEProblem[6] ) / ( 2 * u[6] + u[1] * pODEProblem[6] + 2 * u[2] * pODEProblem[6] - u[5] * ( pODEProblem[6] - 1 ) - 2 * u[8] * ( pODEProblem[6] - 1 ) ) 
	end

end

function compute_u0!(u0::AbstractVector, pODEProblem::AbstractVector) 

	#pODEProblem[1] = ratio, pODEProblem[2] = k_imp_homo, pODEProblem[3] = k_exp_hetero, pODEProblem[4] = cyt, pODEProblem[5] = k_phos, pODEProblem[6] = specC17, pODEProblem[7] = Epo_degradation_BaF3, pODEProblem[8] = k_exp_homo, pODEProblem[9] = nuc, pODEProblem[10] = k_imp_hetero

	STAT5A = 207.6 * pODEProblem[1] 
	pApA = 0.0 
	nucpApB = 0.0 
	nucpBpB = 0.0 
	STAT5B = 207.6 - 207.6 * pODEProblem[1] 
	pApB = 0.0 
	nucpApA = 0.0 
	pBpB = 0.0 

	u0 .= STAT5A, pApA, nucpApB, nucpBpB, STAT5B, pApB, nucpApA, pBpB
end

function compute_u0(pODEProblem::AbstractVector)::AbstractVector 

	#pODEProblem[1] = ratio, pODEProblem[2] = k_imp_homo, pODEProblem[3] = k_exp_hetero, pODEProblem[4] = cyt, pODEProblem[5] = k_phos, pODEProblem[6] = specC17, pODEProblem[7] = Epo_degradation_BaF3, pODEProblem[8] = k_exp_homo, pODEProblem[9] = nuc, pODEProblem[10] = k_imp_hetero

	STAT5A = 207.6 * pODEProblem[1] 
	pApA = 0.0 
	nucpApB = 0.0 
	nucpBpB = 0.0 
	STAT5B = 207.6 - 207.6 * pODEProblem[1] 
	pApB = 0.0 
	nucpApA = 0.0 
	pBpB = 0.0 

	 return [STAT5A, pApA, nucpApB, nucpBpB, STAT5B, pApB, nucpApA, pBpB]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                   parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :pSTAT5A_rel 
		noiseParameter1_pSTAT5A_rel = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_pSTAT5A_rel 
	end

	if observableId === :pSTAT5B_rel 
		noiseParameter1_pSTAT5B_rel = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_pSTAT5B_rel 
	end

	if observableId === :rSTAT5A_rel 
		noiseParameter1_rSTAT5A_rel = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_rSTAT5A_rel 
	end

end