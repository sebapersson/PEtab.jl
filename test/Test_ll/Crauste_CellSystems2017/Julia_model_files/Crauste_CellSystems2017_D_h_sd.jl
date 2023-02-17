#u[1] = Naive, u[2] = Pathogen, u[3] = LateEffector, u[4] = EarlyEffector, u[5] = Memory
#pODEProblem[1] = mu_LL, pODEProblem[2] = delta_NE, pODEProblem[3] = mu_PE, pODEProblem[4] = mu_P, pODEProblem[5] = mu_PL, pODEProblem[6] = delta_EL, pODEProblem[7] = mu_EE, pODEProblem[8] = default, pODEProblem[9] = mu_N, pODEProblem[10] = rho_E, pODEProblem[11] = delta_LM, pODEProblem[12] = rho_P, pODEProblem[13] = mu_LE
#
function compute_∂h∂u!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :observable_EarlyEffector 
		out[4] = 1
		return nothing
	end

	if observableId == :observable_LateEffector 
		out[3] = 1
		return nothing
	end

	if observableId == :observable_Memory 
		out[5] = 1
		return nothing
	end

	if observableId == :observable_Naive 
		out[1] = 1
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :observable_EarlyEffector 
		return nothing
	end

	if observableId == :observable_LateEffector 
		return nothing
	end

	if observableId == :observable_Memory 
		return nothing
	end

	if observableId == :observable_Naive 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :observable_EarlyEffector 
		return nothing
	end

	if observableId == :observable_LateEffector 
		return nothing
	end

	if observableId == :observable_Memory 
		return nothing
	end

	if observableId == :observable_Naive 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :observable_EarlyEffector 
		return nothing
	end

	if observableId == :observable_LateEffector 
		return nothing
	end

	if observableId == :observable_Memory 
		return nothing
	end

	if observableId == :observable_Naive 
		return nothing
	end

end

