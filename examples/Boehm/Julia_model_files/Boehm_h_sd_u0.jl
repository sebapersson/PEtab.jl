#u[1] = STAT5A, u[2] = pApA, u[3] = nucpApB, u[4] = nucpBpB, u[5] = STAT5B, u[6] = pApB, u[7] = nucpApA, u[8] = pBpB
#p_ode_problem_names[1] = ratio, p_ode_problem_names[2] = k_imp_homo, p_ode_problem_names[3] = k_exp_hetero, p_ode_problem_names[4] = cyt, p_ode_problem_names[5] = k_phos, p_ode_problem_names[6] = specC17, p_ode_problem_names[7] = Epo_degradation_BaF3, p_ode_problem_names[8] = k_exp_homo, p_ode_problem_names[9] = nuc, p_ode_problem_names[10] = k_imp_hetero
##parameter_info.nominalValue[7] = ratio_C 
#parameter_info.nominalValue[11] = specC17_C 

function compute_h(u::AbstractVector, t::Real, p_ode_problem::AbstractVector,
                   θ_observable::AbstractVector,
                   θ_non_dynamic::AbstractVector, parameter_info::ParametersInfo,
                   observableId::Symbol,
                   parameter_map::θObsOrSdParameterMap)::Real
    if observableId === :pSTAT5A_rel
        return (100 * u[6] + 200 * u[2] * p_ode_problem[6]) /
               (u[6] + u[1] * p_ode_problem[6] + 2 * u[2] * p_ode_problem[6])
    end

    if observableId === :pSTAT5B_rel
        return -(100 * u[6] - 200 * u[8] * (p_ode_problem[6] - 1)) /
               ((u[5] * (p_ode_problem[6] - 1) - u[6]) + 2 * u[8] * (p_ode_problem[6] - 1))
    end

    if observableId === :rSTAT5A_rel
        return (100 * u[6] + 100 * u[1] * p_ode_problem[6] + 200 * u[2] * p_ode_problem[6]) /
               (2 * u[6] + u[1] * p_ode_problem[6] + 2 * u[2] * p_ode_problem[6] -
                u[5] * (p_ode_problem[6] - 1) - 2 * u[8] * (p_ode_problem[6] - 1))
    end
end

function compute_u0!(u0::AbstractVector, p_ode_problem::AbstractVector)

    #p_ode_problem[1] = ratio, p_ode_problem[2] = k_imp_homo, p_ode_problem[3] = k_exp_hetero, p_ode_problem[4] = cyt, p_ode_problem[5] = k_phos, p_ode_problem[6] = specC17, p_ode_problem[7] = Epo_degradation_BaF3, p_ode_problem[8] = k_exp_homo, p_ode_problem[9] = nuc, p_ode_problem[10] = k_imp_hetero

    STAT5A = 207.6 * p_ode_problem[1]
    pApA = 0.0
    nucpApB = 0.0
    nucpBpB = 0.0
    STAT5B = 207.6 - 207.6 * p_ode_problem[1]
    pApB = 0.0
    nucpApA = 0.0
    pBpB = 0.0

    u0 .= STAT5A, pApA, nucpApB, nucpBpB, STAT5B, pApB, nucpApA, pBpB
end

function compute_u0(p_ode_problem::AbstractVector)::AbstractVector

    #p_ode_problem[1] = ratio, p_ode_problem[2] = k_imp_homo, p_ode_problem[3] = k_exp_hetero, p_ode_problem[4] = cyt, p_ode_problem[5] = k_phos, p_ode_problem[6] = specC17, p_ode_problem[7] = Epo_degradation_BaF3, p_ode_problem[8] = k_exp_homo, p_ode_problem[9] = nuc, p_ode_problem[10] = k_imp_hetero

    STAT5A = 207.6 * p_ode_problem[1]
    pApA = 0.0
    nucpApB = 0.0
    nucpBpB = 0.0
    STAT5B = 207.6 - 207.6 * p_ode_problem[1]
    pApB = 0.0
    nucpApA = 0.0
    pBpB = 0.0

    return [STAT5A, pApA, nucpApB, nucpBpB, STAT5B, pApB, nucpApA, pBpB]
end

function compute_σ(u::AbstractVector, t::Real, θ_sd::AbstractVector,
                   p_ode_problem::AbstractVector, θ_non_dynamic::AbstractVector,
                   parameter_info::ParametersInfo, observableId::Symbol,
                   parameter_map::θObsOrSdParameterMap)::Real
    if observableId === :pSTAT5A_rel
        noiseParameter1_pSTAT5A_rel = get_obs_sd_parameter(θ_sd, parameter_map)
        return noiseParameter1_pSTAT5A_rel
    end

    if observableId === :pSTAT5B_rel
        noiseParameter1_pSTAT5B_rel = get_obs_sd_parameter(θ_sd, parameter_map)
        return noiseParameter1_pSTAT5B_rel
    end

    if observableId === :rSTAT5A_rel
        noiseParameter1_rSTAT5A_rel = get_obs_sd_parameter(θ_sd, parameter_map)
        return noiseParameter1_rSTAT5A_rel
    end
end
