#u[1] = B, u[2] = A
#pODEProblem[1] = compartment, pODEProblem[2] = c1, pODEProblem[3] = b0, pODEProblem[4] = k1, pODEProblem[5] = a0, pODEProblem[6] = k2, pODEProblem[7] = __init__B__, pODEProblem[8] = __init__A__
#
function compute_∂h∂u!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :obs_a 
		out[2] = 1
		return nothing
	end

	if observableId == :obs_b 
		out[1] = 1
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :obs_a 
		return nothing
	end

	if observableId == :obs_b 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :obs_a 
		return nothing
	end

	if observableId == :obs_b 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :obs_a 
		return nothing
	end

	if observableId == :obs_b 
		return nothing
	end

end

