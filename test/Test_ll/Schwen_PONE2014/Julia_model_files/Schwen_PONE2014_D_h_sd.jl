#u[1] = IR2, u[2] = IR2in, u[3] = Rec2, u[4] = IR1in, u[5] = Uptake1, u[6] = Uptake2, u[7] = InsulinFragments, u[8] = IR1, u[9] = Rec1, u[10] = Ins, u[11] = BoundUnspec
#pODEProblem[1] = ka1, pODEProblem[2] = ini_R2fold, pODEProblem[3] = kout, pODEProblem[4] = ini_R1, pODEProblem[5] = kout_frag, pODEProblem[6] = koff_unspec, pODEProblem[7] = kin, pODEProblem[8] = ka2fold, pODEProblem[9] = kin2, pODEProblem[10] = kd1, pODEProblem[11] = kon_unspec, pODEProblem[12] = init_Ins, pODEProblem[13] = kd2fold, pODEProblem[14] = ExtracellularMedium, pODEProblem[15] = kout2
#
function compute_∂h∂u!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId === :observable_IR1 
		observableParameter1_observable_IR1, observableParameter2_observable_IR1 = getObsOrSdParam(θ_observable, parameterMap)
		out[4] = observableParameter2_observable_IR1
		out[8] = observableParameter2_observable_IR1
		return nothing
	end

	if observableId === :observable_IR2 
		observableParameter1_observable_IR2, observableParameter2_observable_IR2 = getObsOrSdParam(θ_observable, parameterMap)
		out[1] = observableParameter2_observable_IR2
		out[2] = observableParameter2_observable_IR2
		return nothing
	end

	if observableId === :observable_IRsum 
		observableParameter1_observable_IRsum, observableParameter2_observable_IRsum = getObsOrSdParam(θ_observable, parameterMap)
		out[1] = 0.395observableParameter2_observable_IRsum
		out[2] = 0.395observableParameter2_observable_IRsum
		out[4] = 0.605observableParameter2_observable_IRsum
		out[8] = 0.605observableParameter2_observable_IRsum
		return nothing
	end

	if observableId === :observable_Insulin 
		observableParameter1_observable_Insulin, observableParameter2_observable_Insulin, observableParameter3_observable_Insulin, observableParameter4_observable_Insulin = getObsOrSdParam(θ_observable, parameterMap)
		out[7] = (observableParameter2_observable_Insulin*observableParameter3_observable_Insulin*(observableParameter4_observable_Insulin^2)) / ((u[10] + observableParameter4_observable_Insulin + u[7]*observableParameter3_observable_Insulin)^2)
		out[10] = (observableParameter2_observable_Insulin*(observableParameter4_observable_Insulin^2)) / ((u[10] + observableParameter4_observable_Insulin + u[7]*observableParameter3_observable_Insulin)^2)
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId === :observable_IR1 
		return nothing
	end

	if observableId === :observable_IR2 
		return nothing
	end

	if observableId === :observable_IRsum 
		return nothing
	end

	if observableId === :observable_Insulin 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId === :observable_IR1 
		return nothing
	end

	if observableId === :observable_IR2 
		return nothing
	end

	if observableId === :observable_IRsum 
		return nothing
	end

	if observableId === :observable_Insulin 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId === :observable_IR1 
		return nothing
	end

	if observableId === :observable_IR2 
		return nothing
	end

	if observableId === :observable_IRsum 
		return nothing
	end

	if observableId === :observable_Insulin 
		return nothing
	end

end

