#u[1] = x, u[2] = y
#pODEProblemNames[1] = c, pODEProblemNames[2] = default, pODEProblemNames[3] = b, pODEProblemNames[4] = a_scale, pODEProblemNames[5] = a, pODEProblemNames[6] = d
##parameterInfo.nominalValue[5] = a_scale_C 


function compute_h(u::AbstractVector, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                   θ_nonDynamic::AbstractVector, parameterInfo::ParametersInfo, observableId::Symbol, 
                      parameterMap::θObsOrSdParameterMap)::Real 
	if observableId == :x1 
		return u[1] 
	end

	if observableId == :x2 
		return u[2] 
	end

end

function compute_u0!(u0::AbstractVector, pODEProblem::AbstractVector) 

	#pODEProblem[1] = c, pODEProblem[2] = default, pODEProblem[3] = b, pODEProblem[4] = a_scale, pODEProblem[5] = a, pODEProblem[6] = d

	x = 0.0 
	y = 0.0 

	u0 .= x, y
end

function compute_u0(pODEProblem::AbstractVector)::AbstractVector 

	#pODEProblem[1] = c, pODEProblem[2] = default, pODEProblem[3] = b, pODEProblem[4] = a_scale, pODEProblem[5] = a, pODEProblem[6] = d

	x = 0.0 
	y = 0.0 

	 return [x, y]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector, 
                   parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap)::Real 
	if observableId == :x1 
		noiseParameter1_x1 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_x1 
	end

	if observableId == :x2 
		noiseParameter1_x2 = getObsOrSdParam(θ_sd, parameterMap)
		return noiseParameter1_x2 
	end

end