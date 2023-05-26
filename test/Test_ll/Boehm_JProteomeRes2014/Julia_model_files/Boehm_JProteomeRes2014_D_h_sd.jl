#u[1] = STAT5A, u[2] = pApA, u[3] = nucpApB, u[4] = nucpBpB, u[5] = STAT5B, u[6] = pApB, u[7] = nucpApA, u[8] = pBpB
#pODEProblem[1] = ratio, pODEProblem[2] = k_imp_homo, pODEProblem[3] = k_exp_hetero, pODEProblem[4] = cyt, pODEProblem[5] = k_phos, pODEProblem[6] = specC17, pODEProblem[7] = Epo_degradation_BaF3, pODEProblem[8] = k_exp_homo, pODEProblem[9] = nuc, pODEProblem[10] = k_imp_hetero
#
function compute_∂h∂u!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,                    
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :pSTAT5A_rel 
		out[1] = (pODEProblem[6]*(-100.0u[6] - 200.0u[2]*pODEProblem[6])) / ((u[6] + u[1]*pODEProblem[6] + 2.0u[2]*pODEProblem[6])^2)
		out[2] = (200.0u[1]*(pODEProblem[6]^2)) / ((u[6] + u[1]*pODEProblem[6] + 2.0u[2]*pODEProblem[6])^2)
		out[6] = (100.0u[1]*pODEProblem[6]) / ((u[6] + u[1]*pODEProblem[6] + 2.0u[2]*pODEProblem[6])^2)
		return nothing
	end

	if observableId == :pSTAT5B_rel 
		out[5] = ((1.0 - pODEProblem[6])*(200.0u[8]*pODEProblem[6] - 100.0u[6] - 200.0u[8])) / ((u[5]*pODEProblem[6] + 2.0u[8]*pODEProblem[6] - u[5] - u[6] - 2.0u[8])^2)
		out[6] = (100.0u[5] - 100.0u[5]*pODEProblem[6]) / ((u[5]*pODEProblem[6] + 2.0u[8]*pODEProblem[6] - u[5] - u[6] - 2.0u[8])^2)
		out[8] = (200.0u[5] + 200.0u[5]*(pODEProblem[6]^2) - 400.0u[5]*pODEProblem[6]) / ((u[5]*pODEProblem[6] + 2.0u[8]*pODEProblem[6] - u[5] - u[6] - 2.0u[8])^2)
		return nothing
	end

	if observableId == :rSTAT5A_rel 
		out[1] = (100.0u[5]*pODEProblem[6] + 100.0u[6]*pODEProblem[6] + 200.0u[8]*pODEProblem[6] - 100.0u[5]*(pODEProblem[6]^2) - 200.0u[8]*(pODEProblem[6]^2)) / ((u[5] + 2.0u[6] + 2.0u[8] + u[1]*pODEProblem[6] + 2.0u[2]*pODEProblem[6] - u[5]*pODEProblem[6] - 2.0u[8]*pODEProblem[6])^2)
		out[2] = (200.0u[5]*pODEProblem[6] + 200.0u[6]*pODEProblem[6] + 400.0u[8]*pODEProblem[6] - 200.0u[5]*(pODEProblem[6]^2) - 400.0u[8]*(pODEProblem[6]^2)) / ((u[5] + 2.0u[6] + 2.0u[8] + u[1]*pODEProblem[6] + 2.0u[2]*pODEProblem[6] - u[5]*pODEProblem[6] - 2.0u[8]*pODEProblem[6])^2)
		out[5] = ((1.0 - pODEProblem[6])*(-100.0u[6] - 100.0u[1]*pODEProblem[6] - 200.0u[2]*pODEProblem[6])) / ((u[5] + 2.0u[6] + 2.0u[8] + u[1]*pODEProblem[6] + 2.0u[2]*pODEProblem[6] - u[5]*pODEProblem[6] - 2.0u[8]*pODEProblem[6])^2)
		out[6] = (100.0u[5] + 200.0u[8] - 100.0u[1]*pODEProblem[6] - 100.0u[5]*pODEProblem[6] - 200.0u[2]*pODEProblem[6] - 200.0u[8]*pODEProblem[6]) / ((u[5] + 2.0u[6] + 2.0u[8] + u[1]*pODEProblem[6] + 2.0u[2]*pODEProblem[6] - u[5]*pODEProblem[6] - 2.0u[8]*pODEProblem[6])^2)
		out[8] = (2.0(pODEProblem[6] - 1.0)*(100.0u[6] + 100.0u[1]*pODEProblem[6] + 200.0u[2]*pODEProblem[6])) / ((u[5] + 2.0u[6] + 2.0u[8] + u[1]*pODEProblem[6] + 2.0u[2]*pODEProblem[6] - u[5]*pODEProblem[6] - 2.0u[8]*pODEProblem[6])^2)
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :pSTAT5A_rel 
		out[6] = (-100.0u[1]*u[6]) / ((u[6] + u[1]*pODEProblem[6] + 2.0u[2]*pODEProblem[6])^2)
		return nothing
	end

	if observableId == :pSTAT5B_rel 
		out[6] = (100.0u[5]*u[6]) / ((u[5]*pODEProblem[6] + 2.0u[8]*pODEProblem[6] - u[5] - u[6] - 2.0u[8])^2)
		return nothing
	end

	if observableId == :rSTAT5A_rel 
		out[6] = (100.0u[1]*u[5] + 200.0u[5]*u[2] + 100.0u[1]*u[6] + 100.0u[5]*u[6] + 200.0u[2]*u[6] + 200.0u[1]*u[8] + 200.0u[6]*u[8] + 400.0u[2]*u[8]) / ((u[5] + 2.0u[6] + 2.0u[8] + u[1]*pODEProblem[6] + 2.0u[2]*pODEProblem[6] - u[5]*pODEProblem[6] - 2.0u[8]*pODEProblem[6])^2)
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector, 
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :pSTAT5A_rel 
		return nothing
	end

	if observableId == :pSTAT5B_rel 
		return nothing
	end

	if observableId == :rSTAT5A_rel 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector, 
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :pSTAT5A_rel 
		return nothing
	end

	if observableId == :pSTAT5B_rel 
		return nothing
	end

	if observableId == :rSTAT5A_rel 
		return nothing
	end

end

