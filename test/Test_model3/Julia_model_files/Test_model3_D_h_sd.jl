#u[1] = x, u[2] = y
#pODEProblem[1] = c, pODEProblem[2] = default, pODEProblem[3] = b, pODEProblem[4] = a_scale, pODEProblem[5] = a, pODEProblem[6] = d
#
function compute_∂h∂u!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId === :x1 
		out[1] = 1
		return nothing
	end

	if observableId === :x2 
		out[2] = 1
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId === :x1 
		return nothing
	end

	if observableId === :x2 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId === :x1 
		return nothing
	end

	if observableId === :x2 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId === :x1 
		return nothing
	end

	if observableId === :x2 
		return nothing
	end

end

