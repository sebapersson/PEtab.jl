#u[1] = IR2, u[2] = IR2in, u[3] = Rec2, u[4] = IR1in, u[5] = Uptake1, u[6] = Uptake2, u[7] = InsulinFragments, u[8] = IR1, u[9] = Rec1, u[10] = Ins, u[11] = BoundUnspec
#pODEProblemNames[1] = ka1, pODEProblemNames[2] = ini_R2fold, pODEProblemNames[3] = kout, pODEProblemNames[4] = ini_R1, pODEProblemNames[5] = kout_frag, pODEProblemNames[6] = koff_unspec, pODEProblemNames[7] = kin, pODEProblemNames[8] = ka2fold, pODEProblemNames[9] = kin2, pODEProblemNames[10] = kd1, pODEProblemNames[11] = kon_unspec, pODEProblemNames[12] = init_Ins, pODEProblemNames[13] = kd2fold, pODEProblemNames[14] = ExtracellularMedium, pODEProblemNames[15] = kout2
#

function compute_h(u::AbstractVector, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                   θ_nonDynamic::AbstractVector, parameterInfo::ParametersInfo, observableId::Symbol, 
                      parameterMap::θObsOrSdParameterMap)::Real 
	if observableId == :observable_IR1 
		observableParameter1_observable_IR1, observableParameter2_observable_IR1 = getObsOrSdParam(θ_observable, parameterMap)
		return observableParameter2_observable_IR1 * ( u[8] + u[4] + observableParameter1_observable_IR1 ) 
	end

	if observableId == :observable_IR2 
		observableParameter1_observable_IR2, observableParameter2_observable_IR2 = getObsOrSdParam(θ_observable, parameterMap)
		return observableParameter2_observable_IR2 * ( u[1] + u[2] + observableParameter1_observable_IR2 ) 
	end

	if observableId == :observable_IRsum 
		observableParameter1_observable_IRsum, observableParameter2_observable_IRsum = getObsOrSdParam(θ_observable, parameterMap)
		return observableParameter2_observable_IRsum * ( 0.605 * u[8] + 0.395 * u[1] + 0.605 * u[4] + 0.395 * u[2] + observableParameter1_observable_IRsum ) 
	end

	if observableId == :observable_Insulin 
		observableParameter1_observable_Insulin, observableParameter2_observable_Insulin, observableParameter3_observable_Insulin, observableParameter4_observable_Insulin = getObsOrSdParam(θ_observable, parameterMap)
		return observableParameter1_observable_Insulin + observableParameter2_observable_Insulin * ( u[10] + u[7] * observableParameter3_observable_Insulin ) / ( ( u[10] + u[7] * observableParameter3_observable_Insulin ) / observableParameter4_observable_Insulin + 1 ) 
	end

end

function compute_u0!(u0::AbstractVector, pODEProblem::AbstractVector) 

	#pODEProblem[1] = ka1, pODEProblem[2] = ini_R2fold, pODEProblem[3] = kout, pODEProblem[4] = ini_R1, pODEProblem[5] = kout_frag, pODEProblem[6] = koff_unspec, pODEProblem[7] = kin, pODEProblem[8] = ka2fold, pODEProblem[9] = kin2, pODEProblem[10] = kd1, pODEProblem[11] = kon_unspec, pODEProblem[12] = init_Ins, pODEProblem[13] = kd2fold, pODEProblem[14] = ExtracellularMedium, pODEProblem[15] = kout2

	IR2 = 0.0 
	IR2in = 0.0 
	Rec2 = pODEProblem[4] * pODEProblem[2] 
	IR1in = 0.0 
	Uptake1 = 0.0 
	Uptake2 = 0.0 
	InsulinFragments = 0.0 
	IR1 = 0.0 
	Rec1 = pODEProblem[4] 
	Ins = pODEProblem[12] 
	BoundUnspec = 0.0 

	u0 .= IR2, IR2in, Rec2, IR1in, Uptake1, Uptake2, InsulinFragments, IR1, Rec1, Ins, BoundUnspec
end

function compute_u0(pODEProblem::AbstractVector)::AbstractVector 

	#pODEProblem[1] = ka1, pODEProblem[2] = ini_R2fold, pODEProblem[3] = kout, pODEProblem[4] = ini_R1, pODEProblem[5] = kout_frag, pODEProblem[6] = koff_unspec, pODEProblem[7] = kin, pODEProblem[8] = ka2fold, pODEProblem[9] = kin2, pODEProblem[10] = kd1, pODEProblem[11] = kon_unspec, pODEProblem[12] = init_Ins, pODEProblem[13] = kd2fold, pODEProblem[14] = ExtracellularMedium, pODEProblem[15] = kout2

	IR2 = 0.0 
	IR2in = 0.0 
	Rec2 = pODEProblem[4] * pODEProblem[2] 
	IR1in = 0.0 
	Uptake1 = 0.0 
	Uptake2 = 0.0 
	InsulinFragments = 0.0 
	IR1 = 0.0 
	Rec1 = pODEProblem[4] 
	Ins = pODEProblem[12] 
	BoundUnspec = 0.0 

	 return [IR2, IR2in, Rec2, IR1in, Uptake1, Uptake2, InsulinFragments, IR1, Rec1, Ins, BoundUnspec]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector, 
                   parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap)::Real 
	if observableId == :observable_IR1 
		noiseParameter1_observable_IR1 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_IR1 
	end

	if observableId == :observable_IR2 
		noiseParameter1_observable_IR2 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_IR2 
	end

	if observableId == :observable_IRsum 
		noiseParameter1_observable_IRsum = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_IRsum 
	end

	if observableId == :observable_Insulin 
		noiseParameter1_observable_Insulin = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_observable_Insulin 
	end

end