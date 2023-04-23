#u[1] = x_k05k16, u[2] = x_k05k08k16, u[3] = x_k05k12k16, u[4] = x_k16, u[5] = x_0ac, u[6] = x_k12, u[7] = x_k12k16, u[8] = x_k05k12, u[9] = x_k08, u[10] = x_k05k08, u[11] = x_k08k16, u[12] = x_k08k12, u[13] = x_k05k08k12, u[14] = x_k05, u[15] = x_4ac, u[16] = x_k08k12k16
#pODEProblem[1] = a_k05_k05k12, pODEProblem[2] = a_0ac_k16, pODEProblem[3] = a_k08_k08k12, pODEProblem[4] = a_k16_k08k16, pODEProblem[5] = a_k05k08k16_4ac, pODEProblem[6] = a_k05k16_k05k08k16, pODEProblem[7] = a_k05k16_k05k12k16, pODEProblem[8] = a_k05k08_k05k08k16, pODEProblem[9] = a_k08k12_k08k12k16, pODEProblem[10] = a_k12k16_k05k12k16, pODEProblem[11] = a_k12_k12k16, pODEProblem[12] = a_0ac_k12, pODEProblem[13] = a_b, pODEProblem[14] = a_k05_k05k08, pODEProblem[15] = a_k08k16_k05k08k16, pODEProblem[16] = a_k08k12k16_4ac, pODEProblem[17] = a_k05k08k12_4ac, pODEProblem[18] = a_k05k08_k05k08k12, pODEProblem[19] = a_k08k16_k08k12k16, pODEProblem[20] = a_k05k12k16_4ac, pODEProblem[21] = a_k05k12_k05k12k16, pODEProblem[22] = a_k08_k05k08, pODEProblem[23] = a_k08_k08k16, pODEProblem[24] = a_k05k12_k05k08k12, pODEProblem[25] = a_k12k16_k08k12k16, pODEProblem[26] = a_k12_k05k12, pODEProblem[27] = da_b, pODEProblem[28] = compartment, pODEProblem[29] = a_0ac_k08, pODEProblem[30] = a_0ac_k05, pODEProblem[31] = a_k16_k05k16, pODEProblem[32] = a_k08k12_k05k08k12, pODEProblem[33] = a_k05_k05k16, pODEProblem[34] = a_k12_k08k12, pODEProblem[35] = a_k16_k12k16
#
function compute_∂h∂u!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,                    
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :observable_x_0ac 
		out[5] = 1
		return nothing
	end

	if observableId == :observable_x_4ac 
		out[15] = 1
		return nothing
	end

	if observableId == :observable_x_k12 
		out[6] = 1
		return nothing
	end

	if observableId == :observable_x_k12k16 
		out[7] = 1
		return nothing
	end

	if observableId == :observable_x_k16 
		out[4] = 1
		return nothing
	end

	if observableId == :observable_x_k05 
		out[14] = 1
		return nothing
	end

	if observableId == :observable_x_k05k12 
		out[8] = 1
		return nothing
	end

	if observableId == :observable_x_k05k12k16 
		out[3] = 1
		return nothing
	end

	if observableId == :observable_x_k05k08 
		out[10] = 1
		return nothing
	end

	if observableId == :observable_x_k05k08k12 
		out[13] = 1
		return nothing
	end

	if observableId == :observable_x_k05k08k16 
		out[2] = 1
		return nothing
	end

	if observableId == :observable_x_k08 
		out[9] = 1
		return nothing
	end

	if observableId == :observable_x_k08k12 
		out[12] = 1
		return nothing
	end

	if observableId == :observable_x_k08k12k16 
		out[16] = 1
		return nothing
	end

	if observableId == :observable_x_k08k16 
		out[11] = 1
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, pODEProblem::AbstractVector, θ_observable::AbstractVector,
                       θ_nonDynamic::AbstractVector, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :observable_x_0ac 
		return nothing
	end

	if observableId == :observable_x_4ac 
		return nothing
	end

	if observableId == :observable_x_k12 
		return nothing
	end

	if observableId == :observable_x_k12k16 
		return nothing
	end

	if observableId == :observable_x_k16 
		return nothing
	end

	if observableId == :observable_x_k05 
		return nothing
	end

	if observableId == :observable_x_k05k12 
		return nothing
	end

	if observableId == :observable_x_k05k12k16 
		return nothing
	end

	if observableId == :observable_x_k05k08 
		return nothing
	end

	if observableId == :observable_x_k05k08k12 
		return nothing
	end

	if observableId == :observable_x_k05k08k16 
		return nothing
	end

	if observableId == :observable_x_k08 
		return nothing
	end

	if observableId == :observable_x_k08k12 
		return nothing
	end

	if observableId == :observable_x_k08k12k16 
		return nothing
	end

	if observableId == :observable_x_k08k16 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector, 
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :observable_x_0ac 
		return nothing
	end

	if observableId == :observable_x_4ac 
		return nothing
	end

	if observableId == :observable_x_k12 
		return nothing
	end

	if observableId == :observable_x_k12k16 
		return nothing
	end

	if observableId == :observable_x_k16 
		return nothing
	end

	if observableId == :observable_x_k05 
		return nothing
	end

	if observableId == :observable_x_k05k12 
		return nothing
	end

	if observableId == :observable_x_k05k12k16 
		return nothing
	end

	if observableId == :observable_x_k05k08 
		return nothing
	end

	if observableId == :observable_x_k05k08k12 
		return nothing
	end

	if observableId == :observable_x_k05k08k16 
		return nothing
	end

	if observableId == :observable_x_k08 
		return nothing
	end

	if observableId == :observable_x_k08k12 
		return nothing
	end

	if observableId == :observable_x_k08k12k16 
		return nothing
	end

	if observableId == :observable_x_k08k16 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, pODEProblem::AbstractVector, θ_nonDynamic::AbstractVector, 
                        parameterInfo::ParametersInfo, observableId::Symbol, parameterMap::θObsOrSdParameterMap, out) 
	if observableId == :observable_x_0ac 
		return nothing
	end

	if observableId == :observable_x_4ac 
		return nothing
	end

	if observableId == :observable_x_k12 
		return nothing
	end

	if observableId == :observable_x_k12k16 
		return nothing
	end

	if observableId == :observable_x_k16 
		return nothing
	end

	if observableId == :observable_x_k05 
		return nothing
	end

	if observableId == :observable_x_k05k12 
		return nothing
	end

	if observableId == :observable_x_k05k12k16 
		return nothing
	end

	if observableId == :observable_x_k05k08 
		return nothing
	end

	if observableId == :observable_x_k05k08k12 
		return nothing
	end

	if observableId == :observable_x_k05k08k16 
		return nothing
	end

	if observableId == :observable_x_k08 
		return nothing
	end

	if observableId == :observable_x_k08k12 
		return nothing
	end

	if observableId == :observable_x_k08k12k16 
		return nothing
	end

	if observableId == :observable_x_k08k16 
		return nothing
	end

end

