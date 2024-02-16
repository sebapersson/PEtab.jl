#u[1] = IR2, u[2] = IR2in, u[3] = Rec2, u[4] = IR1in, u[5] = Uptake1, u[6] = Uptake2, u[7] = InsulinFragments, u[8] = IR1, u[9] = Rec1, u[10] = Ins, u[11] = BoundUnspec
#p_ode_problem[1] = ka1, p_ode_problem[2] = ini_R2fold, p_ode_problem[3] = kout, p_ode_problem[4] = ini_R1, p_ode_problem[5] = kout_frag, p_ode_problem[6] = koff_unspec, p_ode_problem[7] = kin, p_ode_problem[8] = ka2fold, p_ode_problem[9] = kin2, p_ode_problem[10] = kd1, p_ode_problem[11] = kon_unspec, p_ode_problem[12] = init_Ins, p_ode_problem[13] = kd2fold, p_ode_problem[14] = kout2, p_ode_problem[15] = ExtracellularMedium
#
function compute_∂h∂u!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                  θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :observable_IR1 
		observableParameter1_observable_IR1, observableParameter2_observable_IR1 = get_obs_sd_parameter(θ_observable, parameter_map)
		out[4] = observableParameter2_observable_IR1
		out[8] = observableParameter2_observable_IR1
		return nothing
	end

	if observableId == :observable_IR2 
		observableParameter1_observable_IR2, observableParameter2_observable_IR2 = get_obs_sd_parameter(θ_observable, parameter_map)
		out[1] = observableParameter2_observable_IR2
		out[2] = observableParameter2_observable_IR2
		return nothing
	end

	if observableId == :observable_IRsum 
		observableParameter1_observable_IRsum, observableParameter2_observable_IRsum = get_obs_sd_parameter(θ_observable, parameter_map)
		out[1] = 0.395observableParameter2_observable_IRsum
		out[2] = 0.395observableParameter2_observable_IRsum
		out[4] = 0.605observableParameter2_observable_IRsum
		out[8] = 0.605observableParameter2_observable_IRsum
		return nothing
	end

	if observableId == :observable_Insulin 
		observableParameter1_observable_Insulin, observableParameter2_observable_Insulin, observableParameter3_observable_Insulin, observableParameter4_observable_Insulin = get_obs_sd_parameter(θ_observable, parameter_map)
		out[7] = (observableParameter2_observable_Insulin*observableParameter3_observable_Insulin*(observableParameter4_observable_Insulin^2)) / ((u[10] + observableParameter4_observable_Insulin + u[7]*observableParameter3_observable_Insulin)^2)
		out[10] = (observableParameter2_observable_Insulin*(observableParameter4_observable_Insulin^2)) / ((u[10] + observableParameter4_observable_Insulin + u[7]*observableParameter3_observable_Insulin)^2)
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                  θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :observable_IR1 
		return nothing
	end

	if observableId == :observable_IR2 
		return nothing
	end

	if observableId == :observable_IRsum 
		return nothing
	end

	if observableId == :observable_Insulin 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                   parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :observable_IR1 
		return nothing
	end

	if observableId == :observable_IR2 
		return nothing
	end

	if observableId == :observable_IRsum 
		return nothing
	end

	if observableId == :observable_Insulin 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                   parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :observable_IR1 
		return nothing
	end

	if observableId == :observable_IR2 
		return nothing
	end

	if observableId == :observable_IRsum 
		return nothing
	end

	if observableId == :observable_Insulin 
		return nothing
	end

end

