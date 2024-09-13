#u[1] = x1, u[2] = x2, u[3] = observable_x2
#p_ode_problem[1] = k3, p_ode_problem[2] = k1, p_ode_problem[3] = k2, p_ode_problem[4] = default
#
function compute_∂h∂u!(u, t::Real, p_ode_problem::AbstractVector, xobservable::AbstractVector,
                  xnondynamic::AbstractVector, observableId::Symbol, parametermap::ObservableNoiseMap, out)
	if observableId == :obs_x2
		out[2] = 1
		return nothing
	end

end

function compute_∂h∂p!(u, t::Real, p_ode_problem::AbstractVector, xobservable::AbstractVector,
                  xnondynamic::AbstractVector, observableId::Symbol, parametermap::ObservableNoiseMap, out)
	if observableId == :obs_x2
		return nothing
	end

end

function compute_∂σ∂σu!(u, t::Real, xnoise::AbstractVector, p_ode_problem::AbstractVector,  xnondynamic::AbstractVector,
                   petab_parameters::PEtabParameters, observableId::Symbol, parametermap::ObservableNoiseMap, out)
	if observableId == :obs_x2
		return nothing
	end

end

function compute_∂σ∂σp!(u, t::Real, xnoise::AbstractVector, p_ode_problem::AbstractVector,  xnondynamic::AbstractVector,
                   petab_parameters::PEtabParameters, observableId::Symbol, parametermap::ObservableNoiseMap, out)
	if observableId == :obs_x2
		return nothing
	end

end
