#u[1] = sebastian, u[2] = damiano
#pODEProblemNames[1] = default, pODEProblemNames[2] = alpha, pODEProblemNames[3] = beta
#

function compute_h(u::AbstractVector, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                   θ_nonDynamic::AbstractVector, parameterInfo::ParametersInfo, observableId::Symbol,
                      parameterMap::θObsOrSdParameterMap)::Real 
	if observableId == :sebastian_measurement 
		return u[1] 
	end

	if observableId == :damiano_measurement 
		return u[2] 
	end

end

function compute_u0!(u0::AbstractVector, pODEProblem::AbstractVector) 

	#pODEProblem[1] = default, pODEProblem[2] = alpha, pODEProblem[3] = beta

	sebastian = 8.0 
	damiano = 4.0 

	u0 .= sebastian, damiano
end

function compute_u0(pODEProblem::AbstractVector)::AbstractVector 

	#pODEProblem[1] = default, pODEProblem[2] = alpha, pODEProblem[3] = beta

	sebastian = 8.0 
	damiano = 4.0 

	 return [sebastian, damiano]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                   parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap)::Real 
	if observableId == :sebastian_measurement 
		noiseParameter1_sebastian_measurement = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_sebastian_measurement 
	end

	if observableId == :damiano_measurement 
		noiseParameter1_damiano_measurement = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_damiano_measurement 
	end

end