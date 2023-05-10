#u[1] = B, u[2] = A
#pODEProblemNames[1] = compartment, pODEProblemNames[2] = c1, pODEProblemNames[3] = b0, pODEProblemNames[4] = k1, pODEProblemNames[5] = a0, pODEProblemNames[6] = k2, pODEProblemNames[7] = __init__B__, pODEProblemNames[8] = __init__A__
#

function compute_h(u::AbstractVector, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                   θ_nonDynamic::AbstractVector, parameterInfo::ParametersInfo, observableId::Symbol,
                      parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :obs_a 
		return u[2] 
	end

	if observableId === :obs_b 
		return u[1] 
	end

end

function compute_u0!(u0::AbstractVector, pODEProblem::AbstractVector) 

	#pODEProblem[1] = compartment, pODEProblem[2] = c1, pODEProblem[3] = b0, pODEProblem[4] = k1, pODEProblem[5] = a0, pODEProblem[6] = k2, pODEProblem[7] = __init__B__, pODEProblem[8] = __init__A__

	B = pODEProblem[7] 
	A = pODEProblem[8] 

	u0 .= B, A
end

function compute_u0(pODEProblem::AbstractVector)::AbstractVector 

	#pODEProblem[1] = compartment, pODEProblem[2] = c1, pODEProblem[3] = b0, pODEProblem[4] = k1, pODEProblem[5] = a0, pODEProblem[6] = k2, pODEProblem[7] = __init__B__, pODEProblem[8] = __init__A__

	B = pODEProblem[7] 
	A = pODEProblem[8] 

	 return [B, A]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                   parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap)::Real 
	if observableId === :obs_a 
		return 0.5 
	end

	if observableId === :obs_b 
		return 0.2 
	end

end