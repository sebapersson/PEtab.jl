function compute_h(u::AbstractVector, t::Real, p::AbstractVector, xobservable::AbstractVector, xnondynamic::AbstractVector, nominal_values::Vector{Float64}, obsid::Symbol, map::ObservableNoiseMap)::Real
	if obsid == :IR1_P
		observableParameter1_IR1_P = get_obs_sd_parameter(xobservable, map)[1]
		return observableParameter1_IR1_P*(u[1]+u[4]) 
	end
	if obsid == :IRS1_P
		observableParameter1_IRS1_P = get_obs_sd_parameter(xobservable, map)[1]
		return observableParameter1_IRS1_P*u[8] 
	end
	if obsid == :IRS1_P_DosR
		observableParameter1_IRS1_P_DosR = get_obs_sd_parameter(xobservable, map)[1]
		return observableParameter1_IRS1_P_DosR*u[8] 
	end
end

function compute_u0(p::AbstractVector)::AbstractVector
	IRp = 1.7629010620181*1e-9
	IR = 9.94957642787569
	IRins = 0.0173972221725393
	IRiP = 1.11590026152296*1e-5
	IRS = 9.86699348701367
	X = 9.99984199487351
	IRi = 0.0330151891862681
	IRSiP = 0.133006512986336
	Xp = 0.000158005126497888
	return [IRp, IR, IRins, IRiP, IRS, X, IRi, IRSiP, Xp]
end

function compute_u0!(u0::AbstractVector, p::AbstractVector)
	IRp = 1.7629010620181*1e-9
	IR = 9.94957642787569
	IRins = 0.0173972221725393
	IRiP = 1.11590026152296*1e-5
	IRS = 9.86699348701367
	X = 9.99984199487351
	IRi = 0.0330151891862681
	IRSiP = 0.133006512986336
	Xp = 0.000158005126497888
	u0 .= IRp, IR, IRins, IRiP, IRS, X, IRi, IRSiP, Xp
end

function compute_σ(u::AbstractVector, t::Real, p::AbstractVector, xnoise::AbstractVector, xnondynamic::AbstractVector, nominal_values::Vector{Float64}, obsid::Symbol, map::ObservableNoiseMap)::Real
	if obsid == :IR1_P
		noiseParameter1_IR1_P = get_obs_sd_parameter(xnoise, map)[1]
		return noiseParameter1_IR1_P 
	end
	if obsid == :IRS1_P
		noiseParameter1_IRS1_P = get_obs_sd_parameter(xnoise, map)[1]
		return noiseParameter1_IRS1_P 
	end
	if obsid == :IRS1_P_DosR
		noiseParameter1_IRS1_P_DosR = get_obs_sd_parameter(xnoise, map)[1]
		return noiseParameter1_IRS1_P_DosR 
	end
end

