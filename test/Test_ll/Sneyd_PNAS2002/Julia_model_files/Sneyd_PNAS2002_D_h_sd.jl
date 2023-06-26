#u[1] = IPR_S, u[2] = IPR_I2, u[3] = IPR_R, u[4] = IPR_O, u[5] = IPR_I1, u[6] = IPR_A
#pODEProblem[1] = k_1, pODEProblem[2] = l_4, pODEProblem[3] = k_4, pODEProblem[4] = IP3, pODEProblem[5] = k4, pODEProblem[6] = k_2, pODEProblem[7] = l2, pODEProblem[8] = l_2, pODEProblem[9] = default, pODEProblem[10] = l_6, pODEProblem[11] = k1, pODEProblem[12] = k_3, pODEProblem[13] = l6, pODEProblem[14] = membrane, pODEProblem[15] = k3, pODEProblem[16] = l4, pODEProblem[17] = Ca, pODEProblem[18] = k2
#
function compute_∂h∂u!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :open_probability 
		out[4] = 0.4((0.1u[4] + 0.9u[6])^3.0)
		out[6] = 3.6((0.1u[4] + 0.9u[6])^3.0)
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :open_probability 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :open_probability 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :open_probability 
		return nothing
	end

end

