function compute_h(__u_model::AbstractVector, t::Real, __p_model::AbstractVector, xobservable::AbstractVector, xnondynamic_mech::AbstractVector, x_ml_models, x_ml_models_constant, nominal_values::Vector{Float64}, obsid::Symbol, map::ObservableNoiseMap, __sys_observables, ml_models)::Real
	if obsid == :IR1_P
		observableParameter1_IR1_P = get_obs_sd_parameter(xobservable, map)[1]
		return observableParameter1_IR1_P*(__u_model[7]+__u_model[5]) 
	end
	if obsid == :IRS1_P
		observableParameter1_IRS1_P = get_obs_sd_parameter(xobservable, map)[1]
		return observableParameter1_IRS1_P*__u_model[3] 
	end
	if obsid == :IRS1_P_DosR
		observableParameter1_IRS1_P_DosR = get_obs_sd_parameter(xobservable, map)[1]
		return observableParameter1_IRS1_P_DosR*__u_model[3] 
	end
end

function compute_u0!(__u0_model::AbstractVector, __p_model::AbstractVector, __post_eq)
	IR = 9.94957642787569
	IRS = 9.86699348701367
	IRSiP = 0.133006512986336
	IRi = 0.0330151891862681
	IRiP = 1.11590026152296*1e-5
	IRins = 0.0173972221725393
	IRp = 1.7629010620181*1e-9
	X = 9.99984199487351
	Xp = 0.000158005126497888
	__u0_model .= IR, IRS, IRSiP, IRi, IRiP, IRins, IRp, X, Xp
end

function compute_u0(__p_model::AbstractVector, __post_eq)::AbstractVector
	IR = 9.94957642787569
	IRS = 9.86699348701367
	IRSiP = 0.133006512986336
	IRi = 0.0330151891862681
	IRiP = 1.11590026152296*1e-5
	IRins = 0.0173972221725393
	IRp = 1.7629010620181*1e-9
	X = 9.99984199487351
	Xp = 0.000158005126497888
	return [IR, IRS, IRSiP, IRi, IRiP, IRins, IRp, X, Xp]
end

function compute_σ(__u_model::AbstractVector, t::Real, __p_model::AbstractVector, xnoise::AbstractVector, xnondynamic_mech::AbstractVector, x_ml_models, x_ml_models_constant, nominal_values::Vector{Float64}, obsid::Symbol, map::ObservableNoiseMap, __sys_observables, nn)::Real
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