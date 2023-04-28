#u[1] = x1, u[2] = x2
#pODEProblemNames[1] = default, pODEProblemNames[2] = observable_x2, pODEProblemNames[3] = k3, pODEProblemNames[4] = k1, pODEProblemNames[5] = k2
#

function compute_h(u::AbstractVector, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                   θ_nonDynamic::AbstractVector, parameterInfo::ParametersInfo, observableId::Symbol,
                      parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :obs_x2 
		return u[2] 
	end

end

function compute_u0!(u0::AbstractVector, pODEProblem::AbstractVector) 

	#pODEProblem[1] = default, pODEProblem[2] = observable_x2, pODEProblem[3] = k3, pODEProblem[4] = k1, pODEProblem[5] = k2

	x1 = 0.0 
	x2 = 0.0 

	u0 .= x1, x2
end

function compute_u0(pODEProblem::AbstractVector)::AbstractVector 

	#pODEProblem[1] = default, pODEProblem[2] = observable_x2, pODEProblem[3] = k3, pODEProblem[4] = k1, pODEProblem[5] = k2

	x1 = 0.0 
	x2 = 0.0 

	 return [x1, x2]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                   parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :obs_x2 
		return 1 
	end

end