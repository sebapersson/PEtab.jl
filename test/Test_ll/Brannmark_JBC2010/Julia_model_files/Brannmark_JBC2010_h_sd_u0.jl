#u[1] = IRp, u[2] = IR, u[3] = IRins, u[4] = IRiP, u[5] = IRS, u[6] = X, u[7] = IRi, u[8] = IRSiP, u[9] = Xp
#pODEProblemNames[1] = k1c, pODEProblemNames[2] = k21, pODEProblemNames[3] = insulin_bool1, pODEProblemNames[4] = k1g, pODEProblemNames[5] = insulin_dose_2, pODEProblemNames[6] = k1a, pODEProblemNames[7] = insulin_dose_1, pODEProblemNames[8] = k1aBasic, pODEProblemNames[9] = k1d, pODEProblemNames[10] = insulin_time_1, pODEProblemNames[11] = insulin_time_2, pODEProblemNames[12] = cyt, pODEProblemNames[13] = k22, pODEProblemNames[14] = insulin_bool2, pODEProblemNames[15] = default, pODEProblemNames[16] = k1r, pODEProblemNames[17] = k1f, pODEProblemNames[18] = k1b, pODEProblemNames[19] = k3, pODEProblemNames[20] = km2, pODEProblemNames[21] = k1e, pODEProblemNames[22] = k_IRSiP_DosR, pODEProblemNames[23] = km3
#

function compute_h(u::AbstractVector, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                   θ_nonDynamic::AbstractVector, parameterInfo::ParametersInfo, observableId::Symbol,
                      parameterMap::θObsOrSdParameterMap)::Real 
	if observableId == :IR1_P 
		observableParameter1_IR1_P = getObsOrSdParam(θ_observable, parameterMap)
		return observableParameter1_IR1_P * ( u[1] + u[4] ) 
	end

	if observableId == :IRS1_P 
		observableParameter1_IRS1_P = getObsOrSdParam(θ_observable, parameterMap)
		return observableParameter1_IRS1_P * u[8] 
	end

	if observableId == :IRS1_P_DosR 
		observableParameter1_IRS1_P_DosR = getObsOrSdParam(θ_observable, parameterMap)
		return observableParameter1_IRS1_P_DosR * u[8] 
	end

end

function compute_u0!(u0::AbstractVector, pODEProblem::AbstractVector) 

	#pODEProblem[1] = k1c, pODEProblem[2] = k21, pODEProblem[3] = insulin_bool1, pODEProblem[4] = k1g, pODEProblem[5] = insulin_dose_2, pODEProblem[6] = k1a, pODEProblem[7] = insulin_dose_1, pODEProblem[8] = k1aBasic, pODEProblem[9] = k1d, pODEProblem[10] = insulin_time_1, pODEProblem[11] = insulin_time_2, pODEProblem[12] = cyt, pODEProblem[13] = k22, pODEProblem[14] = insulin_bool2, pODEProblem[15] = default, pODEProblem[16] = k1r, pODEProblem[17] = k1f, pODEProblem[18] = k1b, pODEProblem[19] = k3, pODEProblem[20] = km2, pODEProblem[21] = k1e, pODEProblem[22] = k_IRSiP_DosR, pODEProblem[23] = km3

	IRp = 1.7629010620181e-9 
	IR = 9.94957642787569 
	IRins = 0.0173972221725393 
	IRiP = 1.11590026152296e-5 
	IRS = 9.86699348701367 
	X = 9.99984199487351 
	IRi = 0.0330151891862681 
	IRSiP = 0.133006512986336 
	Xp = 0.000158005126497888 

	u0 .= IRp, IR, IRins, IRiP, IRS, X, IRi, IRSiP, Xp
end

function compute_u0(pODEProblem::AbstractVector)::AbstractVector 

	#pODEProblem[1] = k1c, pODEProblem[2] = k21, pODEProblem[3] = insulin_bool1, pODEProblem[4] = k1g, pODEProblem[5] = insulin_dose_2, pODEProblem[6] = k1a, pODEProblem[7] = insulin_dose_1, pODEProblem[8] = k1aBasic, pODEProblem[9] = k1d, pODEProblem[10] = insulin_time_1, pODEProblem[11] = insulin_time_2, pODEProblem[12] = cyt, pODEProblem[13] = k22, pODEProblem[14] = insulin_bool2, pODEProblem[15] = default, pODEProblem[16] = k1r, pODEProblem[17] = k1f, pODEProblem[18] = k1b, pODEProblem[19] = k3, pODEProblem[20] = km2, pODEProblem[21] = k1e, pODEProblem[22] = k_IRSiP_DosR, pODEProblem[23] = km3

	IRp = 1.7629010620181e-9 
	IR = 9.94957642787569 
	IRins = 0.0173972221725393 
	IRiP = 1.11590026152296e-5 
	IRS = 9.86699348701367 
	X = 9.99984199487351 
	IRi = 0.0330151891862681 
	IRSiP = 0.133006512986336 
	Xp = 0.000158005126497888 

	 return [IRp, IR, IRins, IRiP, IRS, X, IRi, IRSiP, Xp]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                   parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap)::Real 
	if observableId == :IR1_P 
		noiseParameter1_IR1_P = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_IR1_P 
	end

	if observableId == :IRS1_P 
		noiseParameter1_IRS1_P = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_IRS1_P 
	end

	if observableId == :IRS1_P_DosR 
		noiseParameter1_IRS1_P_DosR = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_IRS1_P_DosR 
	end

end