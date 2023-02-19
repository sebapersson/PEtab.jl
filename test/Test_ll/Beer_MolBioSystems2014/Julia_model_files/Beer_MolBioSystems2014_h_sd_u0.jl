#u[1] = Glu, u[2] = cGlu, u[3] = Ind, u[4] = Bac
#pODEProblemNames[1] = lag_bool1, pODEProblemNames[2] = kdegi, pODEProblemNames[3] = medium, pODEProblemNames[4] = Bacmax, pODEProblemNames[5] = ksyn, pODEProblemNames[6] = kdim, pODEProblemNames[7] = tau, pODEProblemNames[8] = init_Bac, pODEProblemNames[9] = beta
#

function compute_h(u::AbstractVector, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                   θ_nonDynamic::AbstractVector, parameterInfo::ParametersInfo, observableId::Symbol,
                      parameterMap::θObsOrSdParameterMap)::Real 
	if observableId == :Bacnorm 
		return u[4] 
	end

	if observableId == :IndconcNormRange 
		return u[3] 
	end

end

function compute_u0!(u0::AbstractVector, pODEProblem::AbstractVector) 

	#pODEProblem[1] = lag_bool1, pODEProblem[2] = kdegi, pODEProblem[3] = medium, pODEProblem[4] = Bacmax, pODEProblem[5] = ksyn, pODEProblem[6] = kdim, pODEProblem[7] = tau, pODEProblem[8] = init_Bac, pODEProblem[9] = beta

	Glu = 10.0 
	cGlu = 0.0 
	Ind = 0.0 
	Bac = pODEProblem[8] 

	u0 .= Glu, cGlu, Ind, Bac
end

function compute_u0(pODEProblem::AbstractVector)::AbstractVector 

	#pODEProblem[1] = lag_bool1, pODEProblem[2] = kdegi, pODEProblem[3] = medium, pODEProblem[4] = Bacmax, pODEProblem[5] = ksyn, pODEProblem[6] = kdim, pODEProblem[7] = tau, pODEProblem[8] = init_Bac, pODEProblem[9] = beta

	Glu = 10.0 
	cGlu = 0.0 
	Ind = 0.0 
	Bac = pODEProblem[8] 

	 return [Glu, cGlu, Ind, Bac]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                   parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap)::Real 
	if observableId == :Bacnorm 
		noiseParameter1_Bacnorm = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_Bacnorm 
	end

	if observableId == :IndconcNormRange 
		noiseParameter1_IndconcNormRange = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_IndconcNormRange 
	end

end