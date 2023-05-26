#u[1] = IRp, u[2] = IR, u[3] = IRins, u[4] = IRiP, u[5] = IRS, u[6] = X, u[7] = IRi, u[8] = IRSiP, u[9] = Xp
#pODEProblem[1] = k1c, pODEProblem[2] = k21, pODEProblem[3] = insulin_bool1, pODEProblem[4] = k1g, pODEProblem[5] = insulin_dose_2, pODEProblem[6] = k1a, pODEProblem[7] = insulin_dose_1, pODEProblem[8] = k1aBasic, pODEProblem[9] = insulin_time_1, pODEProblem[10] = insulin_time_2, pODEProblem[11] = k1d, pODEProblem[12] = cyt, pODEProblem[13] = k22, pODEProblem[14] = insulin_bool2, pODEProblem[15] = default, pODEProblem[16] = k1r, pODEProblem[17] = k1f, pODEProblem[18] = k1b, pODEProblem[19] = k3, pODEProblem[20] = km2, pODEProblem[21] = k1e, pODEProblem[22] = k_IRSiP_DosR, pODEProblem[23] = km3
#
function compute_∂h∂u!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,                    
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :IR1_P 
		observableParameter1_IR1_P = getObsOrSdParam(θ_observable, parameterMap)
		out[1] = observableParameter1_IR1_P
		out[4] = observableParameter1_IR1_P
		return nothing
	end

	if observableId == :IRS1_P 
		observableParameter1_IRS1_P = getObsOrSdParam(θ_observable, parameterMap)
		out[8] = observableParameter1_IRS1_P
		return nothing
	end

	if observableId == :IRS1_P_DosR 
		observableParameter1_IRS1_P_DosR = getObsOrSdParam(θ_observable, parameterMap)
		out[8] = observableParameter1_IRS1_P_DosR
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :IR1_P 
		return nothing
	end

	if observableId == :IRS1_P 
		return nothing
	end

	if observableId == :IRS1_P_DosR 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector, 
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :IR1_P 
		return nothing
	end

	if observableId == :IRS1_P 
		return nothing
	end

	if observableId == :IRS1_P_DosR 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector, 
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :IR1_P 
		return nothing
	end

	if observableId == :IRS1_P 
		return nothing
	end

	if observableId == :IRS1_P_DosR 
		return nothing
	end

end

