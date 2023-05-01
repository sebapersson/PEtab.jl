#u[1] = B, u[2] = A
#pODEProblemNames[1] = compartment, pODEProblemNames[2] = b0, pODEProblemNames[3] = k1, pODEProblemNames[4] = a0, pODEProblemNames[5] = k2
#θ_nonDynamic[1] = scaling_A, θ_nonDynamic[2] = offset_A


function compute_h(u::AbstractVector, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                   θ_nonDynamic::AbstractVector, parameterInfo::ParametersInfo, observableId::Symbol,
                      parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :obs_a 
		return θ_nonDynamic[1] * u[2] + θ_nonDynamic[2] 
	end

end

function compute_u0!(u0::AbstractVector, pODEProblem::AbstractVector) 

	#pODEProblem[1] = compartment, pODEProblem[2] = b0, pODEProblem[3] = k1, pODEProblem[4] = a0, pODEProblem[5] = k2

	B = pODEProblem[2] 
	A = pODEProblem[4] 

	u0 .= B, A
end

function compute_u0(pODEProblem::AbstractVector)::AbstractVector 

	#pODEProblem[1] = compartment, pODEProblem[2] = b0, pODEProblem[3] = k1, pODEProblem[4] = a0, pODEProblem[5] = k2

	B = pODEProblem[2] 
	A = pODEProblem[4] 

	 return [B, A]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                   parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :obs_a 
		return 1.0 
	end

end