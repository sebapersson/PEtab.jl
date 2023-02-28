#u[1] = sebastian, u[2] = damiano
#pODEProblem[1] = default, pODEProblem[2] = alpha, pODEProblem[3] = beta
#
function compute_∂h∂u!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,                    
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :sebastian_measurement 
		out[1] = 1
		return nothing
	end

	if observableId == :damiano_measurement 
		out[2] = 1
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :sebastian_measurement 
		return nothing
	end

	if observableId == :damiano_measurement 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector, 
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :sebastian_measurement 
		return nothing
	end

	if observableId == :damiano_measurement 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector, 
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :sebastian_measurement 
		return nothing
	end

	if observableId == :damiano_measurement 
		return nothing
	end

end

