#u[1] = STAT5A, u[2] = pApA, u[3] = nucpApB, u[4] = nucpBpB, u[5] = STAT5B, u[6] = pApB, u[7] = nucpApA, u[8] = pBpB
#p_ode_problem[1] = ratio, p_ode_problem[2] = k_imp_homo, p_ode_problem[3] = k_exp_hetero, p_ode_problem[4] = cyt, p_ode_problem[5] = k_phos, p_ode_problem[6] = specC17, p_ode_problem[7] = Epo_degradation_BaF3, p_ode_problem[8] = k_exp_homo, p_ode_problem[9] = nuc, p_ode_problem[10] = k_imp_hetero
#
function compute_∂h∂u!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,                    
                       θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :pSTAT5A_rel 
		out[1] = (p_ode_problem[6]*(-100u[6] - 200u[2]*p_ode_problem[6])) / ((u[6] + u[1]*p_ode_problem[6] + 2u[2]*p_ode_problem[6])^2)
		out[2] = (200.0u[1]*(p_ode_problem[6]^2)) / ((u[6] + u[1]*p_ode_problem[6] + 2u[2]*p_ode_problem[6])^2)
		out[6] = (100.0u[1]*p_ode_problem[6]) / ((u[6] + u[1]*p_ode_problem[6] + 2u[2]*p_ode_problem[6])^2)
		return nothing
	end

	if observableId == :pSTAT5B_rel 
		out[5] = ((p_ode_problem[6] - 1)*(100u[6] + 200u[8] - 200u[8]*p_ode_problem[6])) / ((u[5]*p_ode_problem[6] + 2u[8]*p_ode_problem[6] - u[5] - u[6] - 2u[8])^2)
		out[6] = (100.0u[5] - 100.0u[5]*p_ode_problem[6]) / ((u[5]*p_ode_problem[6] + 2u[8]*p_ode_problem[6] - u[5] - u[6] - 2u[8])^2)
		out[8] = (200.0u[5] + 200.0u[5]*(p_ode_problem[6]^2) - 400.0u[5]*p_ode_problem[6]) / ((u[5]*p_ode_problem[6] + 2u[8]*p_ode_problem[6] - u[5] - u[6] - 2u[8])^2)
		return nothing
	end

	if observableId == :rSTAT5A_rel 
		out[1] = (100.0u[5]*p_ode_problem[6] + 100.0u[6]*p_ode_problem[6] + 200.0u[8]*p_ode_problem[6] - 100.0u[5]*(p_ode_problem[6]^2) - 200.0u[8]*(p_ode_problem[6]^2)) / ((u[5] + 2u[6] + 2u[8] + u[1]*p_ode_problem[6] + 2u[2]*p_ode_problem[6] - u[5]*p_ode_problem[6] - 2u[8]*p_ode_problem[6])^2)
		out[2] = (200.0u[5]*p_ode_problem[6] + 200.0u[6]*p_ode_problem[6] + 400.0u[8]*p_ode_problem[6] - 200.0u[5]*(p_ode_problem[6]^2) - 400.0u[8]*(p_ode_problem[6]^2)) / ((u[5] + 2u[6] + 2u[8] + u[1]*p_ode_problem[6] + 2u[2]*p_ode_problem[6] - u[5]*p_ode_problem[6] - 2u[8]*p_ode_problem[6])^2)
		out[5] = ((1 - p_ode_problem[6])*(-100u[6] - 100u[1]*p_ode_problem[6] - 200u[2]*p_ode_problem[6])) / ((u[5] + 2u[6] + 2u[8] + u[1]*p_ode_problem[6] + 2u[2]*p_ode_problem[6] - u[5]*p_ode_problem[6] - 2u[8]*p_ode_problem[6])^2)
		out[6] = (100.0u[5] + 200.0u[8] - 100.0u[1]*p_ode_problem[6] - 100.0u[5]*p_ode_problem[6] - 200.0u[2]*p_ode_problem[6] - 200.0u[8]*p_ode_problem[6]) / ((u[5] + 2u[6] + 2u[8] + u[1]*p_ode_problem[6] + 2u[2]*p_ode_problem[6] - u[5]*p_ode_problem[6] - 2u[8]*p_ode_problem[6])^2)
		out[8] = ((2p_ode_problem[6] - 2)*(100u[6] + 100u[1]*p_ode_problem[6] + 200u[2]*p_ode_problem[6])) / ((u[5] + 2u[6] + 2u[8] + u[1]*p_ode_problem[6] + 2u[2]*p_ode_problem[6] - u[5]*p_ode_problem[6] - 2u[8]*p_ode_problem[6])^2)
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                       θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :pSTAT5A_rel 
		out[6] = (-100.0u[1]*u[6]) / ((u[6] + u[1]*p_ode_problem[6] + 2u[2]*p_ode_problem[6])^2)
		return nothing
	end

	if observableId == :pSTAT5B_rel 
		out[6] = (100.0u[5]*u[6]) / ((u[5]*p_ode_problem[6] + 2u[8]*p_ode_problem[6] - u[5] - u[6] - 2u[8])^2)
		return nothing
	end

	if observableId == :rSTAT5A_rel 
		out[6] = (100.0u[1]*u[5] + 100.0u[1]*u[6] + 100.0u[5]*u[6] + 200.0u[1]*u[8] + 200.0u[5]*u[2] + 200.0u[2]*u[6] + 200.0u[6]*u[8] + 400.0u[2]*u[8]) / ((u[5] + 2u[6] + 2u[8] + u[1]*p_ode_problem[6] + 2u[2]*p_ode_problem[6] - u[5]*p_ode_problem[6] - 2u[8]*p_ode_problem[6])^2)
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector, 
                        parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :pSTAT5A_rel 
		return nothing
	end

	if observableId == :pSTAT5B_rel 
		return nothing
	end

	if observableId == :rSTAT5A_rel 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector, 
                        parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :pSTAT5A_rel 
		return nothing
	end

	if observableId == :pSTAT5B_rel 
		return nothing
	end

	if observableId == :rSTAT5A_rel 
		return nothing
	end

end

