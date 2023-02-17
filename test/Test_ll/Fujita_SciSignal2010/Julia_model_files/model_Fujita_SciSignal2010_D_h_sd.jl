#u[1] = pAkt_S6, u[2] = pAkt, u[3] = pS6, u[4] = EGFR, u[5] = pEGFR_Akt, u[6] = pEGFR, u[7] = Akt, u[8] = S6, u[9] = EGF_EGFR
#pODEProblem[1] = EGF_end, pODEProblem[2] = reaction_5_k1, pODEProblem[3] = reaction_2_k2, pODEProblem[4] = init_AKT, pODEProblem[5] = init_EGFR, pODEProblem[6] = EGF_bool1, pODEProblem[7] = EGF_rate, pODEProblem[8] = EGFR_turnover, pODEProblem[9] = reaction_1_k1, pODEProblem[10] = reaction_1_k2, pODEProblem[11] = reaction_8_k1, pODEProblem[12] = reaction_4_k1, pODEProblem[13] = reaction_6_k1, pODEProblem[14] = reaction_2_k1, pODEProblem[15] = init_S6, pODEProblem[16] = reaction_7_k1, pODEProblem[17] = reaction_9_k1, pODEProblem[18] = reaction_3_k1, pODEProblem[19] = reaction_5_k2, pODEProblem[20] = Cell, pODEProblem[21] = EGF_0
#
function compute_∂h∂u!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,                    
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :pAkt_tot 
		observableParameter1_pAkt_tot = getObsOrSdParam(θ_observable, parameterMap)
		out[1] = observableParameter1_pAkt_tot
		out[2] = observableParameter1_pAkt_tot
		return nothing
	end

	if observableId == :pEGFR_tot 
		observableParameter1_pEGFR_tot = getObsOrSdParam(θ_observable, parameterMap)
		out[5] = observableParameter1_pEGFR_tot
		out[6] = observableParameter1_pEGFR_tot
		return nothing
	end

	if observableId == :pS6_tot 
		observableParameter1_pS6_tot = getObsOrSdParam(θ_observable, parameterMap)
		out[3] = observableParameter1_pS6_tot
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :pAkt_tot 
		return nothing
	end

	if observableId == :pEGFR_tot 
		return nothing
	end

	if observableId == :pS6_tot 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector, 
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :pAkt_tot 
		return nothing
	end

	if observableId == :pEGFR_tot 
		return nothing
	end

	if observableId == :pS6_tot 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector, 
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :pAkt_tot 
		return nothing
	end

	if observableId == :pEGFR_tot 
		return nothing
	end

	if observableId == :pS6_tot 
		return nothing
	end

end

