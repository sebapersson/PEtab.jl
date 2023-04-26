# Model name: Isensee
# Number of parameters: 69
# Number of species: 25
function getODEModel_Isensee()

    ### Define independent and dependent variables
    ModelingToolkit.@variables t pAC(t) Rp8_Br_cAMPS(t) Rp8_pCPT_cAMPS(t) PDE(t) Rp_cAMPS(t) RII_2(t) RIIp_Rp8_Br_cAMPS_C_2(t) cAMP(t) RIIp_Sp8_Br_cAMPS_C_2(t) IBMX(t) AC_Fsk(t) RIIp_C_2(t) Sp8_Br_cAMPS(t) RII_C_2(t) RIIp_Sp8_Br_cAMPS_2(t) Csub(t) Csub_H89(t) AC(t) RIIp_cAMP_2(t) pAC_Fsk(t) RIIp_Rp8_pCPT_cAMPS_C_2(t) pPDE(t) RIIp_Rp_cAMPS_C_2(t) RIIp_2(t) RIIp_cAMP_C_2(t)

    ### Store dependent variables in array for ODESystem command
    stateArray = [pAC, Rp8_Br_cAMPS, Rp8_pCPT_cAMPS, PDE, Rp_cAMPS, RII_2, RIIp_Rp8_Br_cAMPS_C_2, cAMP, RIIp_Sp8_Br_cAMPS_C_2, IBMX, AC_Fsk, RIIp_C_2, Sp8_Br_cAMPS, RII_C_2, RIIp_Sp8_Br_cAMPS_2, Csub, Csub_H89, AC, RIIp_cAMP_2, pAC_Fsk, RIIp_Rp8_pCPT_cAMPS_C_2, pPDE, RIIp_Rp_cAMPS_C_2, RIIp_2, RIIp_cAMP_C_2]

    ### Define variable parameters

    ### Define potential algebraic variables
    ModelingToolkit.@variables Rp_cAMPS_pAB(t) H89(t) Fsk(t) IBMXex(t) Rp8_Br_cAMPS_pAB(t) Rp8_pCPT_cAMPS_pAB(t) fourABnOH(t) Sp8_Br_cAMPS_AM(t)

    ### Define parameters
    ModelingToolkit.@parameters ki_Rp8_pCPT_cAMPS_pAB xi_b_Rp_cAMPS RII2_total H89_level fourABnOH_incubation_time KD_Fsk kdeg_cAMP_free Rp8_Br_cAMPS_pAB_level xi_KD_Rp8_Br_cAMPS kf_PDE_Csub Sp8_Br_cAMPS_AM_level xi_kf_RII_C_2__RII_2 kf_RIIp_2__RII_2 kf_cAMP IBMX_time Rp_cAMPS_pAB_incubation_time kf_H89 kf_RII_C_2__RII_2 kf_RII_C_2__RIIp_C_2 xi_i_Rp8_pCPT_cAMPS_pAB xi_pAC fourABnOH_level ki_Sp8_Br_cAMPS_AM xi_b_Sp8_Br_cAMPS xi_b_Rp8_Br_cAMPS H89_time kdeg_cAMP xi_AC_cAMP_Fsk xi_b_Rp8_pCPT_cAMPS ki_Rp_cAMPS_pAB xi_i_Rp_cAMPS_pAB KD_PDE_Csub ki_IBMX Fsk_time Rp8_pCPT_cAMPS_pAB_incubation_time PDE_total ki_Rp8_Br_cAMPS_pAB xi_kf_RII_2__RII_C_2 default xi_i_Rp8_Br_cAMPS_pAB Sp8_Br_cAMPS_AM_time xi_KD_Rp8_pCPT_cAMPS kp_AC xi_pPDE xi_KD_Rp_cAMPS Rp8_Br_cAMPS_pAB_incubation_time AC_total kf_RIIp_C_2__RII_C_2 xi_KD_Sp8_Br_cAMPS nuc kf_RIIp_cAMP_C_2__RIIp_2 KD_cAMP KD_IBMX kf_Fsk xi_i_Sp8_Br_cAMPS_AM cyt KD_H89 ks_AC_cAMP Rp8_pCPT_cAMPS_pAB_level Rp_cAMPS_pAB_level kf_RII_2__RII_C_2 Fsk_level kdp_AC IBMX_level

    ### Store parameters in array for ODESystem command
    parameterArray = [ki_Rp8_pCPT_cAMPS_pAB, xi_b_Rp_cAMPS, RII2_total, H89_level, fourABnOH_incubation_time, KD_Fsk, kdeg_cAMP_free, Rp8_Br_cAMPS_pAB_level, xi_KD_Rp8_Br_cAMPS, kf_PDE_Csub, Sp8_Br_cAMPS_AM_level, xi_kf_RII_C_2__RII_2, kf_RIIp_2__RII_2, kf_cAMP, IBMX_time, Rp_cAMPS_pAB_incubation_time, kf_H89, kf_RII_C_2__RII_2, kf_RII_C_2__RIIp_C_2, xi_i_Rp8_pCPT_cAMPS_pAB, xi_pAC, fourABnOH_level, ki_Sp8_Br_cAMPS_AM, xi_b_Sp8_Br_cAMPS, xi_b_Rp8_Br_cAMPS, H89_time, kdeg_cAMP, xi_AC_cAMP_Fsk, xi_b_Rp8_pCPT_cAMPS, ki_Rp_cAMPS_pAB, xi_i_Rp_cAMPS_pAB, KD_PDE_Csub, ki_IBMX, Fsk_time, Rp8_pCPT_cAMPS_pAB_incubation_time, PDE_total, ki_Rp8_Br_cAMPS_pAB, xi_kf_RII_2__RII_C_2, default, xi_i_Rp8_Br_cAMPS_pAB, Sp8_Br_cAMPS_AM_time, xi_KD_Rp8_pCPT_cAMPS, kp_AC, xi_pPDE, xi_KD_Rp_cAMPS, Rp8_Br_cAMPS_pAB_incubation_time, AC_total, kf_RIIp_C_2__RII_C_2, xi_KD_Sp8_Br_cAMPS, nuc, kf_RIIp_cAMP_C_2__RIIp_2, KD_cAMP, KD_IBMX, kf_Fsk, xi_i_Sp8_Br_cAMPS_AM, cyt, KD_H89, ks_AC_cAMP, Rp8_pCPT_cAMPS_pAB_level, Rp_cAMPS_pAB_level, kf_RII_2__RII_C_2, Fsk_level, kdp_AC, IBMX_level]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Continious events ###

    ### Discrete events ###

    ### Derivatives ###
    eqs = [
    D(pAC) ~ -1.0 * ( 1 /cyt ) * (cyt * Fsk * kf_Fsk * pAC)+1.0 * ( 1 /cyt ) * (cyt * KD_Fsk * kf_Fsk * pAC_Fsk)+1.0 * ( 1 /cyt ) * (cyt * AC * Csub * kp_AC)-1.0 * ( 1 /cyt ) * (cyt * kdp_AC * pAC),
    D(Rp8_Br_cAMPS) ~ +1.0 * ( 1 /cyt ) * (cyt * -ki_Rp8_Br_cAMPS_pAB * (Rp8_Br_cAMPS - Rp8_Br_cAMPS_pAB * xi_i_Rp8_Br_cAMPS_pAB)),
    D(Rp8_pCPT_cAMPS) ~ +1.0 * ( 1 /cyt ) * (cyt * -ki_Rp8_pCPT_cAMPS_pAB * (Rp8_pCPT_cAMPS - Rp8_pCPT_cAMPS_pAB * xi_i_Rp8_pCPT_cAMPS_pAB)),
    D(PDE) ~ -1.0 * ( 1 /cyt ) * (cyt * Csub * PDE * kf_PDE_Csub)+1.0 * ( 1 /cyt ) * (cyt * KD_PDE_Csub * kf_PDE_Csub * pPDE),
    D(Rp_cAMPS) ~ +1.0 * ( 1 /cyt ) * (cyt * -ki_Rp_cAMPS_pAB * (Rp_cAMPS - Rp_cAMPS_pAB * xi_i_Rp_cAMPS_pAB)),
    D(RII_2) ~ +1.0 * ( 1 /cyt ) * (cyt * RII_C_2 * kf_RII_C_2__RII_2)-1.0 * ( 1 /cyt ) * (cyt * Csub * RII_2 * kf_RII_2__RII_C_2)+1.0 * ( 1 /cyt ) * (cyt * RIIp_2 * kf_RIIp_2__RII_2),
    D(RIIp_Rp8_Br_cAMPS_C_2) ~ +1.0 * ( 1 /cyt ) * (cyt * RIIp_C_2 * Rp8_Br_cAMPS * kf_cAMP * xi_b_Rp8_Br_cAMPS)-1.0 * ( 1 /cyt ) * (cyt * KD_cAMP * RIIp_Rp8_Br_cAMPS_C_2 * kf_cAMP * xi_b_Rp8_Br_cAMPS * xi_KD_Rp8_Br_cAMPS),
    D(cAMP) ~ +1.0 * ( 1 /cyt ) * (cyt * ks_AC_cAMP * (AC + pAC * xi_pAC))+1.0 * ( 1 /cyt ) * (cyt * ks_AC_cAMP * xi_AC_cAMP_Fsk * (AC_Fsk + pAC_Fsk * xi_pAC))-1.0 * ( 1 /cyt ) * (cyt * (KD_IBMX * cAMP * kdeg_cAMP_free * (PDE + pPDE * xi_pPDE) / (IBMX + KD_IBMX))),
    D(RIIp_Sp8_Br_cAMPS_C_2) ~ +1.0 * ( 1 /cyt ) * (cyt * RIIp_C_2 * Sp8_Br_cAMPS * kf_cAMP * xi_b_Sp8_Br_cAMPS)-1.0 * ( 1 /cyt ) * (cyt * KD_cAMP * RIIp_Sp8_Br_cAMPS_C_2 * kf_cAMP * xi_b_Sp8_Br_cAMPS * xi_KD_Sp8_Br_cAMPS)-1.0 * ( 1 /cyt ) * (cyt * RIIp_Sp8_Br_cAMPS_C_2 * kf_RIIp_cAMP_C_2__RIIp_2),
    D(IBMX) ~ +1.0 * ( 1 /cyt ) * (cyt * -ki_IBMX * (IBMX - IBMXex)),
    D(AC_Fsk) ~ +1.0 * ( 1 /cyt ) * (cyt * AC * Fsk * kf_Fsk)-1.0 * ( 1 /cyt ) * (cyt * AC_Fsk * KD_Fsk * kf_Fsk)-1.0 * ( 1 /cyt ) * (cyt * AC_Fsk * Csub * kp_AC)+1.0 * ( 1 /cyt ) * (cyt * kdp_AC * pAC_Fsk),
    D(RIIp_C_2) ~ -1.0 * ( 1 /cyt ) * (cyt * RIIp_C_2 * kf_RII_C_2__RII_2 * xi_kf_RII_C_2__RII_2)+1.0 * ( 1 /cyt ) * (cyt * Csub * RIIp_2 * kf_RII_2__RII_C_2 * xi_kf_RII_2__RII_C_2)+1.0 * ( 1 /cyt ) * (cyt * RII_C_2 * kf_RII_C_2__RIIp_C_2)-1.0 * ( 1 /cyt ) * (cyt * RIIp_C_2 * kf_RIIp_C_2__RII_C_2)-1.0 * ( 1 /cyt ) * (cyt * RIIp_C_2 * cAMP * kf_cAMP)+1.0 * ( 1 /cyt ) * (cyt * KD_cAMP * RIIp_cAMP_C_2 * kf_cAMP)+1.0 * ( 1 /cyt ) * (cyt * (KD_IBMX * RIIp_cAMP_C_2 * kdeg_cAMP * (PDE + pPDE * xi_pPDE) / (IBMX + KD_IBMX)))-1.0 * ( 1 /cyt ) * (cyt * RIIp_C_2 * Rp8_Br_cAMPS * kf_cAMP * xi_b_Rp8_Br_cAMPS)+1.0 * ( 1 /cyt ) * (cyt * KD_cAMP * RIIp_Rp8_Br_cAMPS_C_2 * kf_cAMP * xi_b_Rp8_Br_cAMPS * xi_KD_Rp8_Br_cAMPS)-1.0 * ( 1 /cyt ) * (cyt * RIIp_C_2 * Rp8_pCPT_cAMPS * kf_cAMP * xi_b_Rp8_pCPT_cAMPS)+1.0 * ( 1 /cyt ) * (cyt * KD_cAMP * RIIp_Rp8_pCPT_cAMPS_C_2 * kf_cAMP * xi_b_Rp8_pCPT_cAMPS * xi_KD_Rp8_pCPT_cAMPS)-1.0 * ( 1 /cyt ) * (cyt * RIIp_C_2 * Rp_cAMPS * kf_cAMP * xi_b_Rp_cAMPS)+1.0 * ( 1 /cyt ) * (cyt * KD_cAMP * RIIp_Rp_cAMPS_C_2 * kf_cAMP * xi_b_Rp_cAMPS * xi_KD_Rp_cAMPS)-1.0 * ( 1 /cyt ) * (cyt * RIIp_C_2 * Sp8_Br_cAMPS * kf_cAMP * xi_b_Sp8_Br_cAMPS)+1.0 * ( 1 /cyt ) * (cyt * KD_cAMP * RIIp_Sp8_Br_cAMPS_C_2 * kf_cAMP * xi_b_Sp8_Br_cAMPS * xi_KD_Sp8_Br_cAMPS),
    D(Sp8_Br_cAMPS) ~ +1.0 * ( 1 /cyt ) * (cyt * -ki_Sp8_Br_cAMPS_AM * (Sp8_Br_cAMPS - Sp8_Br_cAMPS_AM * xi_i_Sp8_Br_cAMPS_AM)),
    D(RII_C_2) ~ -1.0 * ( 1 /cyt ) * (cyt * RII_C_2 * kf_RII_C_2__RII_2)+1.0 * ( 1 /cyt ) * (cyt * Csub * RII_2 * kf_RII_2__RII_C_2)-1.0 * ( 1 /cyt ) * (cyt * RII_C_2 * kf_RII_C_2__RIIp_C_2)+1.0 * ( 1 /cyt ) * (cyt * RIIp_C_2 * kf_RIIp_C_2__RII_C_2),
    D(RIIp_Sp8_Br_cAMPS_2) ~ +1.0 * ( 1 /cyt ) * (cyt * RIIp_Sp8_Br_cAMPS_C_2 * kf_RIIp_cAMP_C_2__RIIp_2)-1.0 * ( 1 /cyt ) * (cyt * KD_cAMP * RIIp_Sp8_Br_cAMPS_2 * kf_cAMP * xi_b_Sp8_Br_cAMPS * xi_KD_Sp8_Br_cAMPS),
    D(Csub) ~ +1.0 * ( 1 /cyt ) * (cyt * RII_C_2 * kf_RII_C_2__RII_2)-1.0 * ( 1 /cyt ) * (cyt * Csub * RII_2 * kf_RII_2__RII_C_2)+1.0 * ( 1 /cyt ) * (cyt * RIIp_C_2 * kf_RII_C_2__RII_2 * xi_kf_RII_C_2__RII_2)-1.0 * ( 1 /cyt ) * (cyt * Csub * RIIp_2 * kf_RII_2__RII_C_2 * xi_kf_RII_2__RII_C_2)+1.0 * ( 1 /cyt ) * (cyt * RIIp_cAMP_C_2 * kf_RIIp_cAMP_C_2__RIIp_2)+1.0 * ( 1 /cyt ) * (cyt * RIIp_Sp8_Br_cAMPS_C_2 * kf_RIIp_cAMP_C_2__RIIp_2)-1.0 * ( 1 /cyt ) * (cyt * Csub * H89 * kf_H89)+1.0 * ( 1 /cyt ) * (cyt * Csub_H89 * KD_H89 * kf_H89),
    D(Csub_H89) ~ +1.0 * ( 1 /cyt ) * (cyt * Csub * H89 * kf_H89)-1.0 * ( 1 /cyt ) * (cyt * Csub_H89 * KD_H89 * kf_H89),
    D(AC) ~ -1.0 * ( 1 /cyt ) * (cyt * AC * Fsk * kf_Fsk)+1.0 * ( 1 /cyt ) * (cyt * AC_Fsk * KD_Fsk * kf_Fsk)-1.0 * ( 1 /cyt ) * (cyt * AC * Csub * kp_AC)+1.0 * ( 1 /cyt ) * (cyt * kdp_AC * pAC),
    D(RIIp_cAMP_2) ~ +1.0 * ( 1 /cyt ) * (cyt * RIIp_cAMP_C_2 * kf_RIIp_cAMP_C_2__RIIp_2)-1.0 * ( 1 /cyt ) * (cyt * KD_cAMP * RIIp_cAMP_2 * kf_cAMP)-1.0 * ( 1 /cyt ) * (cyt * (KD_IBMX * RIIp_cAMP_2 * kdeg_cAMP * (PDE + pPDE * xi_pPDE) / (IBMX + KD_IBMX))),
    D(pAC_Fsk) ~ +1.0 * ( 1 /cyt ) * (cyt * Fsk * kf_Fsk * pAC)-1.0 * ( 1 /cyt ) * (cyt * KD_Fsk * kf_Fsk * pAC_Fsk)+1.0 * ( 1 /cyt ) * (cyt * AC_Fsk * Csub * kp_AC)-1.0 * ( 1 /cyt ) * (cyt * kdp_AC * pAC_Fsk),
    D(RIIp_Rp8_pCPT_cAMPS_C_2) ~ +1.0 * ( 1 /cyt ) * (cyt * RIIp_C_2 * Rp8_pCPT_cAMPS * kf_cAMP * xi_b_Rp8_pCPT_cAMPS)-1.0 * ( 1 /cyt ) * (cyt * KD_cAMP * RIIp_Rp8_pCPT_cAMPS_C_2 * kf_cAMP * xi_b_Rp8_pCPT_cAMPS * xi_KD_Rp8_pCPT_cAMPS),
    D(pPDE) ~ +1.0 * ( 1 /cyt ) * (cyt * Csub * PDE * kf_PDE_Csub)-1.0 * ( 1 /cyt ) * (cyt * KD_PDE_Csub * kf_PDE_Csub * pPDE),
    D(RIIp_Rp_cAMPS_C_2) ~ +1.0 * ( 1 /cyt ) * (cyt * RIIp_C_2 * Rp_cAMPS * kf_cAMP * xi_b_Rp_cAMPS)-1.0 * ( 1 /cyt ) * (cyt * KD_cAMP * RIIp_Rp_cAMPS_C_2 * kf_cAMP * xi_b_Rp_cAMPS * xi_KD_Rp_cAMPS),
    D(RIIp_2) ~ +1.0 * ( 1 /cyt ) * (cyt * RIIp_C_2 * kf_RII_C_2__RII_2 * xi_kf_RII_C_2__RII_2)-1.0 * ( 1 /cyt ) * (cyt * Csub * RIIp_2 * kf_RII_2__RII_C_2 * xi_kf_RII_2__RII_C_2)-1.0 * ( 1 /cyt ) * (cyt * RIIp_2 * kf_RIIp_2__RII_2)+1.0 * ( 1 /cyt ) * (cyt * KD_cAMP * RIIp_cAMP_2 * kf_cAMP)+1.0 * ( 1 /cyt ) * (cyt * (KD_IBMX * RIIp_cAMP_2 * kdeg_cAMP * (PDE + pPDE * xi_pPDE) / (IBMX + KD_IBMX)))+1.0 * ( 1 /cyt ) * (cyt * KD_cAMP * RIIp_Sp8_Br_cAMPS_2 * kf_cAMP * xi_b_Sp8_Br_cAMPS * xi_KD_Sp8_Br_cAMPS),
    D(RIIp_cAMP_C_2) ~ +1.0 * ( 1 /cyt ) * (cyt * RIIp_C_2 * cAMP * kf_cAMP)-1.0 * ( 1 /cyt ) * (cyt * KD_cAMP * RIIp_cAMP_C_2 * kf_cAMP)-1.0 * ( 1 /cyt ) * (cyt * (KD_IBMX * RIIp_cAMP_C_2 * kdeg_cAMP * (PDE + pPDE * xi_pPDE) / (IBMX + KD_IBMX)))-1.0 * ( 1 /cyt ) * (cyt * RIIp_cAMP_C_2 * kf_RIIp_cAMP_C_2__RIIp_2),
    Rp_cAMPS_pAB ~ Rp_cAMPS_pAB_level * ifelse(t - Rp_cAMPS_pAB_incubation_time < 0, 0, 1),
    H89 ~ H89_level * ifelse(t - H89_time < 0, 0, 1),
    Fsk ~ Fsk_level * ifelse(t - Fsk_time < 0, 0, 1),
    IBMXex ~ IBMX_level * ifelse(t - IBMX_time < 0, 0, 1),
    Rp8_Br_cAMPS_pAB ~ Rp8_Br_cAMPS_pAB_level * ifelse(t - Rp8_Br_cAMPS_pAB_incubation_time < 0, 0, 1),
    Rp8_pCPT_cAMPS_pAB ~ Rp8_pCPT_cAMPS_pAB_level * ifelse(t - Rp8_pCPT_cAMPS_pAB_incubation_time < 0, 0, 1),
    fourABnOH ~ fourABnOH_level * ifelse(t - fourABnOH_incubation_time < 0, 0, 1),
    Sp8_Br_cAMPS_AM ~ Sp8_Br_cAMPS_AM_level * ifelse(t - Sp8_Br_cAMPS_AM_time < 0, 0, 1)
    ]

    @named sys = ODESystem(eqs, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    initialSpeciesValues = [
    pAC => 0.0,
    Rp8_Br_cAMPS => 0.0,
    Rp8_pCPT_cAMPS => 0.0,
    PDE => 1.0,
    Rp_cAMPS => 0.0,
    RII_2 => 0.057671482854616,
    RIIp_Rp8_Br_cAMPS_C_2 => 0.0,
    cAMP => 0.0575218758977949,
    RIIp_Sp8_Br_cAMPS_C_2 => 0.0,
    IBMX => 0.0,
    AC_Fsk => 0.0,
    RIIp_C_2 => 0.287974352643203,
    Sp8_Br_cAMPS => 0.0,
    RII_C_2 => 0.514562213223039,
    RIIp_Sp8_Br_cAMPS_2 => 0.0,
    Csub => 0.192051405433538,
    Csub_H89 => 0.0,
    AC => 1.0,
    RIIp_cAMP_2 => 0.000400416634006059,
    pAC_Fsk => 0.0,
    RIIp_Rp8_pCPT_cAMPS_C_2 => 0.0,
    pPDE => 0.0,
    RIIp_Rp_cAMPS_C_2 => 0.0,
    RIIp_2 => 0.133979505944916,
    RIIp_cAMP_C_2 => 0.00541202870022029
    ]

    ### SBML file parameter values ###
    trueParameterValues = [
    ki_Rp8_pCPT_cAMPS_pAB => 9.24294887100985,
    xi_b_Rp_cAMPS => 0.607960299288444,
    RII2_total => 1.0,
    H89_level => 0.0,
    fourABnOH_incubation_time => 0.0,
    KD_Fsk => 14.2656743301721,
    kdeg_cAMP_free => 7.35892283451732,
    Rp8_Br_cAMPS_pAB_level => 0.0,
    xi_KD_Rp8_Br_cAMPS => 0.040150612333366,
    kf_PDE_Csub => 0.0,
    Sp8_Br_cAMPS_AM_level => 0.0,
    xi_kf_RII_C_2__RII_2 => 0.811223214879921,
    kf_RIIp_2__RII_2 => 0.0323066621451248,
    kf_cAMP => 0.49762030092122,
    IBMX_time => 0.0,
    Rp_cAMPS_pAB_incubation_time => 0.0,
    kf_H89 => 0.000741493422675392,
    kf_RII_C_2__RII_2 => 0.0187262534054505,
    kf_RII_C_2__RIIp_C_2 => 0.0227841500487899,
    xi_i_Rp8_pCPT_cAMPS_pAB => 1.0,
    xi_pAC => 1.0,
    fourABnOH_level => 0.0,
    ki_Sp8_Br_cAMPS_AM => 0.131083036712545,
    xi_b_Sp8_Br_cAMPS => 21.7551956731725,
    xi_b_Rp8_Br_cAMPS => 0.0408695544562406,
    H89_time => 0.0,
    kdeg_cAMP => 1.00032978441703e-5,
    xi_AC_cAMP_Fsk => 640.551657640036,
    xi_b_Rp8_pCPT_cAMPS => 0.0268897774883765,
    ki_Rp_cAMPS_pAB => 0.0163472838984393,
    xi_i_Rp_cAMPS_pAB => 1.0,
    KD_PDE_Csub => 0.0,
    ki_IBMX => 999.99999996515,
    Fsk_time => 0.0,
    Rp8_pCPT_cAMPS_pAB_incubation_time => 0.0,
    PDE_total => 1.0,
    ki_Rp8_Br_cAMPS_pAB => 6.39991617125844,
    xi_kf_RII_2__RII_C_2 => 0.0189295042137343,
    default => 1.0,
    xi_i_Rp8_Br_cAMPS_pAB => 1.0,
    Sp8_Br_cAMPS_AM_time => 0.0,
    xi_KD_Rp8_pCPT_cAMPS => 0.194630402579553,
    kp_AC => 0.0,
    xi_pPDE => 1.0,
    xi_KD_Rp_cAMPS => 0.163369645426461,
    Rp8_Br_cAMPS_pAB_incubation_time => 0.0,
    AC_total => 1.0,
    kf_RIIp_C_2__RII_C_2 => 0.0256808704479994,
    xi_KD_Sp8_Br_cAMPS => 0.222276461349335,
    nuc => 1.0,
    kf_RIIp_cAMP_C_2__RIIp_2 => 0.104924782853171,
    KD_cAMP => 2.84986906175302,
    KD_IBMX => 11.9768989891142,
    kf_Fsk => 999.991987450139,
    xi_i_Sp8_Br_cAMPS_AM => 1.0,
    cyt => 1.0,
    KD_H89 => 0.0480736009761681,
    ks_AC_cAMP => 0.423299046028554,
    Rp8_pCPT_cAMPS_pAB_level => 0.0,
    Rp_cAMPS_pAB_level => 0.0,
    kf_RII_2__RII_C_2 => 1.26077939273477,
    Fsk_level => 0.0,
    kdp_AC => 0.0,
    IBMX_level => 0.0
    ]

    return sys, initialSpeciesValues, trueParameterValues

end
