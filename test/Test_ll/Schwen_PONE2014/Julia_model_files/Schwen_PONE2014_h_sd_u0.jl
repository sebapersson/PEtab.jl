#u[1] = IR2, u[2] = IR2in, u[3] = Rec2, u[4] = IR1in, u[5] = Uptake1, u[6] = Uptake2, u[7] = InsulinFragments, u[8] = IR1, u[9] = Rec1, u[10] = Ins, u[11] = BoundUnspec
#p_ode_problem_names[1] = ka1, p_ode_problem_names[2] = ini_R2fold, p_ode_problem_names[3] = kout, p_ode_problem_names[4] = ini_R1, p_ode_problem_names[5] = kout_frag, p_ode_problem_names[6] = koff_unspec, p_ode_problem_names[7] = kin, p_ode_problem_names[8] = ka2fold, p_ode_problem_names[9] = kin2, p_ode_problem_names[10] = kd1, p_ode_problem_names[11] = kon_unspec, p_ode_problem_names[12] = init_Ins, p_ode_problem_names[13] = kd2fold, p_ode_problem_names[14] = kout2, p_ode_problem_names[15] = ExtracellularMedium
#

function compute_h(u::AbstractVector, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
               θ_non_dynamic::AbstractVector, parameter_info::ParametersInfo, observableId::Symbol,
                  parameter_map::θObsOrSdParameterMap)::Real 
	if observableId === :observable_IR1 
		observableParameter1_observable_IR1, observableParameter2_observable_IR1 = get_obs_sd_parameter(θ_observable, parameter_map)
		return observableParameter2_observable_IR1 * ( u[8] + u[4] + observableParameter1_observable_IR1 ) 
	end

	if observableId === :observable_IR2 
		observableParameter1_observable_IR2, observableParameter2_observable_IR2 = get_obs_sd_parameter(θ_observable, parameter_map)
		return observableParameter2_observable_IR2 * ( u[1] + u[2] + observableParameter1_observable_IR2 ) 
	end

	if observableId === :observable_IRsum 
		observableParameter1_observable_IRsum, observableParameter2_observable_IRsum = get_obs_sd_parameter(θ_observable, parameter_map)
		return observableParameter2_observable_IRsum * ( 0.605 * u[8] + 0.395 * u[1] + 0.605 * u[4] + 0.395 * u[2] + observableParameter1_observable_IRsum ) 
	end

	if observableId === :observable_Insulin 
		observableParameter1_observable_Insulin, observableParameter2_observable_Insulin, observableParameter3_observable_Insulin, observableParameter4_observable_Insulin = get_obs_sd_parameter(θ_observable, parameter_map)
		return observableParameter1_observable_Insulin + observableParameter2_observable_Insulin * ( u[10] + u[7] * observableParameter3_observable_Insulin ) / ( ( u[10] + u[7] * observableParameter3_observable_Insulin ) / observableParameter4_observable_Insulin + 1.0 ) 
	end

end

function compute_u0!(u0::AbstractVector, p_ode_problem::AbstractVector) 

	#p_ode_problem[1] = ka1, p_ode_problem[2] = ini_R2fold, p_ode_problem[3] = kout, p_ode_problem[4] = ini_R1, p_ode_problem[5] = kout_frag, p_ode_problem[6] = koff_unspec, p_ode_problem[7] = kin, p_ode_problem[8] = ka2fold, p_ode_problem[9] = kin2, p_ode_problem[10] = kd1, p_ode_problem[11] = kon_unspec, p_ode_problem[12] = init_Ins, p_ode_problem[13] = kd2fold, p_ode_problem[14] = kout2, p_ode_problem[15] = ExtracellularMedium

	t = 0.0 # u at time zero

	IR2 = 0.0 
	IR2in = 0.0 
	Rec2 = p_ode_problem[4] * p_ode_problem[2] 
	IR1in = 0.0 
	Uptake1 = 0.0 
	Uptake2 = 0.0 
	InsulinFragments = 0.0 
	IR1 = 0.0 
	Rec1 = p_ode_problem[4] 
	Ins = p_ode_problem[12] 
	BoundUnspec = 0.0 

	u0 .= IR2, IR2in, Rec2, IR1in, Uptake1, Uptake2, InsulinFragments, IR1, Rec1, Ins, BoundUnspec
end

function compute_u0(p_ode_problem::AbstractVector)::AbstractVector 

	#p_ode_problem[1] = ka1, p_ode_problem[2] = ini_R2fold, p_ode_problem[3] = kout, p_ode_problem[4] = ini_R1, p_ode_problem[5] = kout_frag, p_ode_problem[6] = koff_unspec, p_ode_problem[7] = kin, p_ode_problem[8] = ka2fold, p_ode_problem[9] = kin2, p_ode_problem[10] = kd1, p_ode_problem[11] = kon_unspec, p_ode_problem[12] = init_Ins, p_ode_problem[13] = kd2fold, p_ode_problem[14] = kout2, p_ode_problem[15] = ExtracellularMedium

	t = 0.0 # u at time zero

	IR2 = 0.0 
	IR2in = 0.0 
	Rec2 = p_ode_problem[4] * p_ode_problem[2] 
	IR1in = 0.0 
	Uptake1 = 0.0 
	Uptake2 = 0.0 
	InsulinFragments = 0.0 
	IR1 = 0.0 
	Rec1 = p_ode_problem[4] 
	Ins = p_ode_problem[12] 
	BoundUnspec = 0.0 

	 return [IR2, IR2in, Rec2, IR1in, Uptake1, Uptake2, InsulinFragments, IR1, Rec1, Ins, BoundUnspec]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
               parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap)::Real 
	if observableId === :observable_IR1 
		noiseParameter1_observable_IR1 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_IR1 
	end

	if observableId === :observable_IR2 
		noiseParameter1_observable_IR2 = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_IR2 
	end

	if observableId === :observable_IRsum 
		noiseParameter1_observable_IRsum = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_IRsum 
	end

	if observableId === :observable_Insulin 
		noiseParameter1_observable_Insulin = get_obs_sd_parameter(θ_sd, parameter_map)
		return noiseParameter1_observable_Insulin 
	end


end

