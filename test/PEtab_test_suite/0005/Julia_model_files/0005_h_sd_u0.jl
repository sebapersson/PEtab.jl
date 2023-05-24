#u[1] = B, u[2] = A
#pODEProblemNames[1] = compartment, pODEProblemNames[2] = b0, pODEProblemNames[3] = offset_A, pODEProblemNames[4] = k1, pODEProblemNames[5] = a0, pODEProblemNames[6] = k2
#

function compute_h(u::AbstractVector, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                   θ_nonDynamic::AbstractVector, parameterInfo::ParametersInfo, observableId::Symbol,
                      parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :obs_a 
		return u[2] + pODEProblem[3] 
	end

end

function compute_u0!(u0::AbstractVector, pODEProblem::AbstractVector) 

	#pODEProblem[1] = compartment, pODEProblem[2] = b0, pODEProblem[3] = offset_A, pODEProblem[4] = k1, pODEProblem[5] = a0, pODEProblem[6] = k2

	B = pODEProblem[2] 
	A = pODEProblem[5] 

	u0 .= B, A
end

function compute_u0(pODEProblem::AbstractVector)::AbstractVector 

	#pODEProblem[1] = compartment, pODEProblem[2] = b0, pODEProblem[3] = offset_A, pODEProblem[4] = k1, pODEProblem[5] = a0, pODEProblem[6] = k2

	B = pODEProblem[2] 
	A = pODEProblem[5] 

	 return [B, A]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                   parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :obs_a 
		return 1.0 
	end

end