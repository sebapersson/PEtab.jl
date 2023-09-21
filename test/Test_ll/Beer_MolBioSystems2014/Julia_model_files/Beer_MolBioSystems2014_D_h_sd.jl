#u[1] = Glu, u[2] = cGlu, u[3] = Ind, u[4] = Bac
#p_ode_problem[1] = lag_bool1, p_ode_problem[2] = kdegi, p_ode_problem[3] = medium, p_ode_problem[4] = Bacmax, p_ode_problem[5] = ksyn, p_ode_problem[6] = kdim, p_ode_problem[7] = tau, p_ode_problem[8] = init_Bac, p_ode_problem[9] = beta
#
function compute_∂h∂u!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                       θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :Bacnorm 
		out[4] = 1
		return nothing
	end

	if observableId == :IndconcNormRange 
		out[3] = 1
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, p_ode_problem::AbstractVector, θ_observable::AbstractVector,
                       θ_non_dynamic::AbstractVector, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :Bacnorm 
		return nothing
	end

	if observableId == :IndconcNormRange 
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                        parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :Bacnorm 
		return nothing
	end

	if observableId == :IndconcNormRange 
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,  θ_non_dynamic::AbstractVector,
                        parameter_info::ParametersInfo, observableId::Symbol, parameter_map::θObsOrSdParameterMap, out) 
	if observableId == :Bacnorm 
		return nothing
	end

	if observableId == :IndconcNormRange 
		return nothing
	end

end

