#u[1] = b10, u[2] = bio, u[3] = ohbio, u[4] = zea, u[5] = bcry, u[6] = ohb10, u[7] = bcar
#pODEProblem[1] = kc2_multiplier, pODEProblem[2] = init_zea, pODEProblem[3] = kc4_multiplier, pODEProblem[4] = cyt, pODEProblem[5] = k5_multiplier, pODEProblem[6] = kc1_multiplier, pODEProblem[7] = init_b10, pODEProblem[8] = init_bcry, pODEProblem[9] = kb1_multiplier, pODEProblem[10] = kb2_multiplier, pODEProblem[11] = kc1, pODEProblem[12] = kc4, pODEProblem[13] = init_ohb10, pODEProblem[14] = init_bcar, pODEProblem[15] = kc2, pODEProblem[16] = kb2, pODEProblem[17] = k5, pODEProblem[18] = kb1
#
function compute_∂h∂u!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :ob10 
		out[1] = 1
		return nothing
	end

	if observableId == :obcar 
		out[7] = 1
		return nothing
	end

	if observableId == :obcry 
		out[5] = 1
		return nothing
	end

	if observableId == :obio 
		out[2] = 1
		return nothing
	end

	if observableId == :oohb10 
		out[6] = 1
		return nothing
	end

	if observableId == :ozea 
		out[4] = 1
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :ob10 
		return nothing
	end

	if observableId == :obcar 
		return nothing
	end

	if observableId == :obcry 
		return nothing
	end

	if observableId == :obio 
		return nothing
	end

	if observableId == :oohb10 
		return nothing
	end

	if observableId == :ozea 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :ob10 
		return nothing
	end

	if observableId == :obcar 
		return nothing
	end

	if observableId == :obcry 
		return nothing
	end

	if observableId == :obio 
		return nothing
	end

	if observableId == :oohb10 
		return nothing
	end

	if observableId == :ozea 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector,
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :ob10 
		return nothing
	end

	if observableId == :obcar 
		return nothing
	end

	if observableId == :obcry 
		return nothing
	end

	if observableId == :obio 
		return nothing
	end

	if observableId == :oohb10 
		return nothing
	end

	if observableId == :ozea 
		return nothing
	end

end

