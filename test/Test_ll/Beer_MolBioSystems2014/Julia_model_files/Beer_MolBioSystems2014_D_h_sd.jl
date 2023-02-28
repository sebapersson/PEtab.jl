#u[1] = Glu, u[2] = cGlu, u[3] = Ind, u[4] = Bac
#pODEProblem[1] = lag_bool1, pODEProblem[2] = kdegi, pODEProblem[3] = medium, pODEProblem[4] = Bacmax, pODEProblem[5] = ksyn, pODEProblem[6] = kdim, pODEProblem[7] = tau, pODEProblem[8] = init_Bac, pODEProblem[9] = beta
#
function compute_∂h∂u!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,                    
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :Bacnorm 
		out[4] = 1
		return nothing
	end

	if observableId == :IndconcNormRange 
		out[3] = 1
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :Bacnorm 
		return nothing
	end

	if observableId == :IndconcNormRange 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector, 
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :Bacnorm 
		return nothing
	end

	if observableId == :IndconcNormRange 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector, 
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :Bacnorm 
		return nothing
	end

	if observableId == :IndconcNormRange 
		return nothing
	end

end

