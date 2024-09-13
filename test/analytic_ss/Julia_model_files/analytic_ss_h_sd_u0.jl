function compute_h(u::AbstractVector, t::Real, p::AbstractVector, xobservable::AbstractVector, xnondynamic::AbstractVector, nominal_values::Vector{Float64}, obsid::Symbol, map::ObservableNoiseMap)::Real
	if obsid == :x1
		return u[1] 
	end
	if obsid == :x2
		return u[2] 
	end
end

function compute_u0(p::AbstractVector)::AbstractVector
	x = 0.0
	y = 0.0
	return [x, y]
end

function compute_u0!(u0::AbstractVector, p::AbstractVector)
	x = 0.0
	y = 0.0
	u0 .= x, y
end

function compute_σ(u::AbstractVector, t::Real, p::AbstractVector, xnoise::AbstractVector, xnondynamic::AbstractVector, nominal_values::Vector{Float64}, obsid::Symbol, map::ObservableNoiseMap)::Real
	if obsid == :x1
		noiseParameter1_x1 = get_obs_sd_parameter(xnoise, map)[1]
		return noiseParameter1_x1 
	end
	if obsid == :x2
		noiseParameter1_x2 = get_obs_sd_parameter(xnoise, map)[1]
		return noiseParameter1_x2 
	end
end

