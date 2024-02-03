#u[1] = p1EpoRpJAK2, u[2] = pSTAT5, u[3] = EpoRJAK2_CIS, u[4] = SOCS3nRNA4, u[5] = SOCS3RNA, u[6] = SHP1, u[7] = STAT5, u[8] = EpoRJAK2, u[9] = CISnRNA1, u[10] = SOCS3nRNA1, u[11] = SOCS3nRNA2, u[12] = CISnRNA3, u[13] = CISnRNA4, u[14] = SOCS3, u[15] = CISnRNA5, u[16] = SOCS3nRNA5, u[17] = SOCS3nRNA3, u[18] = SHP1Act, u[19] = npSTAT5, u[20] = p12EpoRpJAK2, u[21] = p2EpoRpJAK2, u[22] = CIS, u[23] = EpoRpJAK2, u[24] = CISnRNA2, u[25] = CISRNA
#p_ode_problem[1] = STAT5Exp, p_ode_problem[2] = STAT5Imp, p_ode_problem[3] = init_SOCS3_multiplier, p_ode_problem[4] = EpoRCISRemove, p_ode_problem[5] = STAT5ActEpoR, p_ode_problem[6] = SHP1ActEpoR, p_ode_problem[7] = JAK2EpoRDeaSHP1, p_ode_problem[8] = CISTurn, p_ode_problem[9] = SOCS3Turn, p_ode_problem[10] = init_EpoRJAK2_CIS, p_ode_problem[11] = SOCS3Inh, p_ode_problem[12] = ActD, p_ode_problem[13] = init_CIS_multiplier, p_ode_problem[14] = cyt, p_ode_problem[15] = CISRNAEqc, p_ode_problem[16] = JAK2ActEpo, p_ode_problem[17] = Epo, p_ode_problem[18] = SOCS3oe, p_ode_problem[19] = CISInh, p_ode_problem[20] = SHP1Dea, p_ode_problem[21] = SOCS3EqcOE, p_ode_problem[22] = CISRNADelay, p_ode_problem[23] = init_SHP1, p_ode_problem[24] = CISEqcOE, p_ode_problem[25] = EpoRActJAK2, p_ode_problem[26] = SOCS3RNAEqc, p_ode_problem[27] = CISEqc, p_ode_problem[28] = SHP1ProOE, p_ode_problem[29] = SOCS3RNADelay, p_ode_problem[30] = init_STAT5, p_ode_problem[31] = CISoe, p_ode_problem[32] = CISRNATurn, p_ode_problem[33] = init_SHP1_multiplier, p_ode_problem[34] = init_EpoRJAK2, p_ode_problem[35] = nuc, p_ode_problem[36] = EpoRCISInh, p_ode_problem[37] = STAT5ActJAK2, p_ode_problem[38] = SOCS3RNATurn, p_ode_problem[39] = SOCS3Eqc
#
function compute_∂h∂u!(u, t::Real, p_ode_problem::AbstractVector,
                       θ_observable::AbstractVector,
                       θ_non_dynamic::AbstractVector, observableId::Symbol,
                       parameter_map::θObsOrSdParameterMap, out)
    if observableId == :observable_CISRNA_foldA
        observableParameter1_observable_CISRNA_foldA = get_obs_sd_parameter(θ_observable,
                                                                            parameter_map)
        out[25] = observableParameter1_observable_CISRNA_foldA / p_ode_problem[15]
        return nothing
    end

    if observableId == :observable_CISRNA_foldB
        observableParameter1_observable_CISRNA_foldB = get_obs_sd_parameter(θ_observable,
                                                                            parameter_map)
        out[25] = observableParameter1_observable_CISRNA_foldB / p_ode_problem[15]
        return nothing
    end

    if observableId == :observable_CISRNA_foldC
        observableParameter1_observable_CISRNA_foldC = get_obs_sd_parameter(θ_observable,
                                                                            parameter_map)
        out[25] = observableParameter1_observable_CISRNA_foldC / p_ode_problem[15]
        return nothing
    end

    if observableId == :observable_CIS_abs
        out[22] = 1
        return nothing
    end

    if observableId == :observable_CIS_au
        observableParameter1_observable_CIS_au, observableParameter2_observable_CIS_au = get_obs_sd_parameter(θ_observable,
                                                                                                              parameter_map)
        out[22] = observableParameter2_observable_CIS_au / p_ode_problem[27]
        return nothing
    end

    if observableId == :observable_CIS_au1
        observableParameter1_observable_CIS_au1 = get_obs_sd_parameter(θ_observable,
                                                                       parameter_map)
        out[22] = observableParameter1_observable_CIS_au1 / p_ode_problem[27]
        return nothing
    end

    if observableId == :observable_CIS_au2
        observableParameter1_observable_CIS_au2 = get_obs_sd_parameter(θ_observable,
                                                                       parameter_map)
        out[22] = observableParameter1_observable_CIS_au2 / p_ode_problem[27]
        return nothing
    end

    if observableId == :observable_SHP1_abs
        out[6] = 1
        out[18] = 1
        return nothing
    end

    if observableId == :observable_SOCS3RNA_foldA
        observableParameter1_observable_SOCS3RNA_foldA = get_obs_sd_parameter(θ_observable,
                                                                              parameter_map)
        out[5] = observableParameter1_observable_SOCS3RNA_foldA / p_ode_problem[26]
        return nothing
    end

    if observableId == :observable_SOCS3RNA_foldB
        observableParameter1_observable_SOCS3RNA_foldB = get_obs_sd_parameter(θ_observable,
                                                                              parameter_map)
        out[5] = observableParameter1_observable_SOCS3RNA_foldB / p_ode_problem[26]
        return nothing
    end

    if observableId == :observable_SOCS3RNA_foldC
        observableParameter1_observable_SOCS3RNA_foldC = get_obs_sd_parameter(θ_observable,
                                                                              parameter_map)
        out[5] = observableParameter1_observable_SOCS3RNA_foldC / p_ode_problem[26]
        return nothing
    end

    if observableId == :observable_SOCS3_abs
        out[14] = 1
        return nothing
    end

    if observableId == :observable_SOCS3_au
        observableParameter1_observable_SOCS3_au, observableParameter2_observable_SOCS3_au = get_obs_sd_parameter(θ_observable,
                                                                                                                  parameter_map)
        out[14] = observableParameter2_observable_SOCS3_au / p_ode_problem[39]
        return nothing
    end

    if observableId == :observable_STAT5_abs
        out[7] = 1
        return nothing
    end

    if observableId == :observable_pEpoR_au
        observableParameter1_observable_pEpoR_au, observableParameter2_observable_pEpoR_au = get_obs_sd_parameter(θ_observable,
                                                                                                                  parameter_map)
        out[1] = (16observableParameter2_observable_pEpoR_au) / p_ode_problem[34]
        out[20] = (16observableParameter2_observable_pEpoR_au) / p_ode_problem[34]
        out[21] = (16observableParameter2_observable_pEpoR_au) / p_ode_problem[34]
        return nothing
    end

    if observableId == :observable_pJAK2_au
        observableParameter1_observable_pJAK2_au, observableParameter2_observable_pJAK2_au = get_obs_sd_parameter(θ_observable,
                                                                                                                  parameter_map)
        out[1] = (2observableParameter2_observable_pJAK2_au) / p_ode_problem[34]
        out[20] = (2observableParameter2_observable_pJAK2_au) / p_ode_problem[34]
        out[21] = (2observableParameter2_observable_pJAK2_au) / p_ode_problem[34]
        out[23] = (2observableParameter2_observable_pJAK2_au) / p_ode_problem[34]
        return nothing
    end

    if observableId == :observable_pSTAT5B_rel
        observableParameter1_observable_pSTAT5B_rel = get_obs_sd_parameter(θ_observable,
                                                                           parameter_map)
        out[2] = (100.0u[7]) / ((u[7] + u[2])^2)
        out[7] = (-100u[2]) / ((u[7] + u[2])^2)
        return nothing
    end

    if observableId == :observable_pSTAT5_au
        observableParameter1_observable_pSTAT5_au, observableParameter2_observable_pSTAT5_au = get_obs_sd_parameter(θ_observable,
                                                                                                                    parameter_map)
        out[2] = observableParameter2_observable_pSTAT5_au / p_ode_problem[30]
        return nothing
    end

    if observableId == :observable_tSHP1_au
        observableParameter1_observable_tSHP1_au = get_obs_sd_parameter(θ_observable,
                                                                        parameter_map)
        out[6] = observableParameter1_observable_tSHP1_au / p_ode_problem[23]
        out[18] = observableParameter1_observable_tSHP1_au / p_ode_problem[23]
        return nothing
    end

    if observableId == :observable_tSTAT5_au
        observableParameter1_observable_tSTAT5_au = get_obs_sd_parameter(θ_observable,
                                                                         parameter_map)
        out[2] = observableParameter1_observable_tSTAT5_au / p_ode_problem[30]
        out[7] = observableParameter1_observable_tSTAT5_au / p_ode_problem[30]
        return nothing
    end
end

function compute_∂h∂p!(u, t::Real, p_ode_problem::AbstractVector,
                       θ_observable::AbstractVector,
                       θ_non_dynamic::AbstractVector, observableId::Symbol,
                       parameter_map::θObsOrSdParameterMap, out)
    if observableId == :observable_CISRNA_foldA
        observableParameter1_observable_CISRNA_foldA = get_obs_sd_parameter(θ_observable,
                                                                            parameter_map)
        out[15] = (-u[25] * observableParameter1_observable_CISRNA_foldA) /
                  (p_ode_problem[15]^2)
        return nothing
    end

    if observableId == :observable_CISRNA_foldB
        observableParameter1_observable_CISRNA_foldB = get_obs_sd_parameter(θ_observable,
                                                                            parameter_map)
        out[15] = (-u[25] * observableParameter1_observable_CISRNA_foldB) /
                  (p_ode_problem[15]^2)
        return nothing
    end

    if observableId == :observable_CISRNA_foldC
        observableParameter1_observable_CISRNA_foldC = get_obs_sd_parameter(θ_observable,
                                                                            parameter_map)
        out[15] = (-u[25] * observableParameter1_observable_CISRNA_foldC) /
                  (p_ode_problem[15]^2)
        return nothing
    end

    if observableId == :observable_CIS_abs
        return nothing
    end

    if observableId == :observable_CIS_au
        observableParameter1_observable_CIS_au, observableParameter2_observable_CIS_au = get_obs_sd_parameter(θ_observable,
                                                                                                              parameter_map)
        out[27] = (-u[22] * observableParameter2_observable_CIS_au) / (p_ode_problem[27]^2)
        return nothing
    end

    if observableId == :observable_CIS_au1
        observableParameter1_observable_CIS_au1 = get_obs_sd_parameter(θ_observable,
                                                                       parameter_map)
        out[27] = (-u[22] * observableParameter1_observable_CIS_au1) / (p_ode_problem[27]^2)
        return nothing
    end

    if observableId == :observable_CIS_au2
        observableParameter1_observable_CIS_au2 = get_obs_sd_parameter(θ_observable,
                                                                       parameter_map)
        out[27] = (-u[22] * observableParameter1_observable_CIS_au2) / (p_ode_problem[27]^2)
        return nothing
    end

    if observableId == :observable_SHP1_abs
        return nothing
    end

    if observableId == :observable_SOCS3RNA_foldA
        observableParameter1_observable_SOCS3RNA_foldA = get_obs_sd_parameter(θ_observable,
                                                                              parameter_map)
        out[26] = (-u[5] * observableParameter1_observable_SOCS3RNA_foldA) /
                  (p_ode_problem[26]^2)
        return nothing
    end

    if observableId == :observable_SOCS3RNA_foldB
        observableParameter1_observable_SOCS3RNA_foldB = get_obs_sd_parameter(θ_observable,
                                                                              parameter_map)
        out[26] = (-u[5] * observableParameter1_observable_SOCS3RNA_foldB) /
                  (p_ode_problem[26]^2)
        return nothing
    end

    if observableId == :observable_SOCS3RNA_foldC
        observableParameter1_observable_SOCS3RNA_foldC = get_obs_sd_parameter(θ_observable,
                                                                              parameter_map)
        out[26] = (-u[5] * observableParameter1_observable_SOCS3RNA_foldC) /
                  (p_ode_problem[26]^2)
        return nothing
    end

    if observableId == :observable_SOCS3_abs
        return nothing
    end

    if observableId == :observable_SOCS3_au
        observableParameter1_observable_SOCS3_au, observableParameter2_observable_SOCS3_au = get_obs_sd_parameter(θ_observable,
                                                                                                                  parameter_map)
        out[39] = (-u[14] * observableParameter2_observable_SOCS3_au) /
                  (p_ode_problem[39]^2)
        return nothing
    end

    if observableId == :observable_STAT5_abs
        return nothing
    end

    if observableId == :observable_pEpoR_au
        observableParameter1_observable_pEpoR_au, observableParameter2_observable_pEpoR_au = get_obs_sd_parameter(θ_observable,
                                                                                                                  parameter_map)
        out[34] = (observableParameter2_observable_pEpoR_au * (-16u[20] - 16u[1] - 16u[21])) /
                  (p_ode_problem[34]^2)
        return nothing
    end

    if observableId == :observable_pJAK2_au
        observableParameter1_observable_pJAK2_au, observableParameter2_observable_pJAK2_au = get_obs_sd_parameter(θ_observable,
                                                                                                                  parameter_map)
        out[34] = (-observableParameter2_observable_pJAK2_au *
                   (2u[23] + 2u[20] + 2u[1] + 2u[21])) / (p_ode_problem[34]^2)
        return nothing
    end

    if observableId == :observable_pSTAT5B_rel
        return nothing
    end

    if observableId == :observable_pSTAT5_au
        observableParameter1_observable_pSTAT5_au, observableParameter2_observable_pSTAT5_au = get_obs_sd_parameter(θ_observable,
                                                                                                                    parameter_map)
        out[30] = (-observableParameter2_observable_pSTAT5_au * u[2]) /
                  (p_ode_problem[30]^2)
        return nothing
    end

    if observableId == :observable_tSHP1_au
        observableParameter1_observable_tSHP1_au = get_obs_sd_parameter(θ_observable,
                                                                        parameter_map)
        out[23] = (-observableParameter1_observable_tSHP1_au * (u[6] + u[18])) /
                  (p_ode_problem[23]^2)
        return nothing
    end

    if observableId == :observable_tSTAT5_au
        observableParameter1_observable_tSTAT5_au = get_obs_sd_parameter(θ_observable,
                                                                         parameter_map)
        out[30] = (observableParameter1_observable_tSTAT5_au * (-u[7] - u[2])) /
                  (p_ode_problem[30]^2)
        return nothing
    end
end

function compute_∂σ∂σu!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,
                        θ_non_dynamic::AbstractVector,
                        parameter_info::ParametersInfo, observableId::Symbol,
                        parameter_map::θObsOrSdParameterMap, out)
    if observableId == :observable_CISRNA_foldA
        return nothing
    end

    if observableId == :observable_CISRNA_foldB
        return nothing
    end

    if observableId == :observable_CISRNA_foldC
        return nothing
    end

    if observableId == :observable_CIS_abs
        return nothing
    end

    if observableId == :observable_CIS_au
        return nothing
    end

    if observableId == :observable_CIS_au1
        return nothing
    end

    if observableId == :observable_CIS_au2
        return nothing
    end

    if observableId == :observable_SHP1_abs
        return nothing
    end

    if observableId == :observable_SOCS3RNA_foldA
        return nothing
    end

    if observableId == :observable_SOCS3RNA_foldB
        return nothing
    end

    if observableId == :observable_SOCS3RNA_foldC
        return nothing
    end

    if observableId == :observable_SOCS3_abs
        return nothing
    end

    if observableId == :observable_SOCS3_au
        return nothing
    end

    if observableId == :observable_STAT5_abs
        return nothing
    end

    if observableId == :observable_pEpoR_au
        return nothing
    end

    if observableId == :observable_pJAK2_au
        return nothing
    end

    if observableId == :observable_pSTAT5B_rel
        return nothing
    end

    if observableId == :observable_pSTAT5_au
        return nothing
    end

    if observableId == :observable_tSHP1_au
        return nothing
    end

    if observableId == :observable_tSTAT5_au
        return nothing
    end
end

function compute_∂σ∂σp!(u, t::Real, θ_sd::AbstractVector, p_ode_problem::AbstractVector,
                        θ_non_dynamic::AbstractVector,
                        parameter_info::ParametersInfo, observableId::Symbol,
                        parameter_map::θObsOrSdParameterMap, out)
    if observableId == :observable_CISRNA_foldA
        return nothing
    end

    if observableId == :observable_CISRNA_foldB
        return nothing
    end

    if observableId == :observable_CISRNA_foldC
        return nothing
    end

    if observableId == :observable_CIS_abs
        return nothing
    end

    if observableId == :observable_CIS_au
        return nothing
    end

    if observableId == :observable_CIS_au1
        return nothing
    end

    if observableId == :observable_CIS_au2
        return nothing
    end

    if observableId == :observable_SHP1_abs
        return nothing
    end

    if observableId == :observable_SOCS3RNA_foldA
        return nothing
    end

    if observableId == :observable_SOCS3RNA_foldB
        return nothing
    end

    if observableId == :observable_SOCS3RNA_foldC
        return nothing
    end

    if observableId == :observable_SOCS3_abs
        return nothing
    end

    if observableId == :observable_SOCS3_au
        return nothing
    end

    if observableId == :observable_STAT5_abs
        return nothing
    end

    if observableId == :observable_pEpoR_au
        return nothing
    end

    if observableId == :observable_pJAK2_au
        return nothing
    end

    if observableId == :observable_pSTAT5B_rel
        return nothing
    end

    if observableId == :observable_pSTAT5_au
        return nothing
    end

    if observableId == :observable_tSHP1_au
        return nothing
    end

    if observableId == :observable_tSTAT5_au
        return nothing
    end
end
