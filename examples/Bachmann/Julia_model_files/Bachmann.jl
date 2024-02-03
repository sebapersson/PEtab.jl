# Model name: Bachmann
# Number of parameters: 37
# Number of species: 25
function getODEModel_Bachmann()

    ### Define independent and dependent variables
    ModelingToolkit.@variables t p1EpoRpJAK2(t) pSTAT5(t) EpoRJAK2_CIS(t) SOCS3nRNA4(t) SOCS3RNA(t) SHP1(t) STAT5(t) EpoRJAK2(t) CISnRNA1(t) SOCS3nRNA1(t) SOCS3nRNA2(t) CISnRNA3(t) CISnRNA4(t) SOCS3(t) CISnRNA5(t) SOCS3nRNA5(t) SOCS3nRNA3(t) SHP1Act(t) npSTAT5(t) p12EpoRpJAK2(t) p2EpoRpJAK2(t) CIS(t) EpoRpJAK2(t) CISnRNA2(t) CISRNA(t)

    ### Store dependent variables in array for ODESystem command
    stateArray = [
        p1EpoRpJAK2,
        pSTAT5,
        EpoRJAK2_CIS,
        SOCS3nRNA4,
        SOCS3RNA,
        SHP1,
        STAT5,
        EpoRJAK2,
        CISnRNA1,
        SOCS3nRNA1,
        SOCS3nRNA2,
        CISnRNA3,
        CISnRNA4,
        SOCS3,
        CISnRNA5,
        SOCS3nRNA5,
        SOCS3nRNA3,
        SHP1Act,
        npSTAT5,
        p12EpoRpJAK2,
        p2EpoRpJAK2,
        CIS,
        EpoRpJAK2,
        CISnRNA2,
        CISRNA,
    ]

    ### Define variable parameters

    ### Define potential algebraic variables

    ### Define parameters
    ModelingToolkit.@parameters STAT5Exp STAT5Imp init_SOCS3_multiplier EpoRCISRemove STAT5ActEpoR SHP1ActEpoR JAK2EpoRDeaSHP1 CISTurn SOCS3Turn init_EpoRJAK2_CIS SOCS3Inh ActD init_CIS_multiplier cyt CISRNAEqc JAK2ActEpo Epo SOCS3oe CISInh SHP1Dea SOCS3EqcOE CISRNADelay init_SHP1 CISEqcOE EpoRActJAK2 SOCS3RNAEqc CISEqc SHP1ProOE SOCS3RNADelay init_STAT5 CISoe CISRNATurn init_SHP1_multiplier init_EpoRJAK2 nuc EpoRCISInh STAT5ActJAK2 SOCS3RNATurn SOCS3Eqc

    ### Store parameters in array for ODESystem command
    parameterArray = [
        STAT5Exp,
        STAT5Imp,
        init_SOCS3_multiplier,
        EpoRCISRemove,
        STAT5ActEpoR,
        SHP1ActEpoR,
        JAK2EpoRDeaSHP1,
        CISTurn,
        SOCS3Turn,
        init_EpoRJAK2_CIS,
        SOCS3Inh,
        ActD,
        init_CIS_multiplier,
        cyt,
        CISRNAEqc,
        JAK2ActEpo,
        Epo,
        SOCS3oe,
        CISInh,
        SHP1Dea,
        SOCS3EqcOE,
        CISRNADelay,
        init_SHP1,
        CISEqcOE,
        EpoRActJAK2,
        SOCS3RNAEqc,
        CISEqc,
        SHP1ProOE,
        SOCS3RNADelay,
        init_STAT5,
        CISoe,
        CISRNATurn,
        init_SHP1_multiplier,
        init_EpoRJAK2,
        nuc,
        EpoRCISInh,
        STAT5ActJAK2,
        SOCS3RNATurn,
        SOCS3Eqc,
    ]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Continious events ###

    ### Discrete events ###

    ### Derivatives ###
    eqs = [
        D(p1EpoRpJAK2) ~ +1.0 * (1 / cyt) *
                         (cyt *
                          (EpoRpJAK2 * EpoRActJAK2 / (SOCS3 * SOCS3Inh / SOCS3Eqc + 1))) -
                         1.0 * (1 / cyt) *
                         (cyt * (3 * EpoRActJAK2 * p1EpoRpJAK2 /
                           ((SOCS3 * SOCS3Inh / SOCS3Eqc + 1) *
                            (EpoRCISInh * EpoRJAK2_CIS + 1)))) -
                         1.0 * (1 / cyt) *
                         (cyt * (JAK2EpoRDeaSHP1 * SHP1Act * p1EpoRpJAK2 / init_SHP1)),
        D(pSTAT5) ~ +1.0 * (1 / cyt) *
                    (cyt * (STAT5 * STAT5ActJAK2 *
                      (EpoRpJAK2 + p12EpoRpJAK2 + p1EpoRpJAK2 + p2EpoRpJAK2) /
                      (init_EpoRJAK2 * (SOCS3 * SOCS3Inh / SOCS3Eqc + 1)))) +
                    1.0 * (1 / cyt) *
                    (cyt * (STAT5 * STAT5ActEpoR * (p12EpoRpJAK2 + p1EpoRpJAK2)^(2) /
                      ((init_EpoRJAK2)^(2) * (SOCS3 * SOCS3Inh / SOCS3Eqc + 1) *
                       (CIS * CISInh / CISEqc + 1)))) -
                    1.0 * (1 / cyt) * (cyt * STAT5Imp * pSTAT5),
        D(EpoRJAK2_CIS) ~ -1.0 * (1 / cyt) *
                          (cyt *
                           (EpoRJAK2_CIS * EpoRCISRemove * (p12EpoRpJAK2 + p1EpoRpJAK2) /
                            init_EpoRJAK2)),
        D(SOCS3nRNA4) ~ +1.0 * (1 / nuc) * (nuc * SOCS3nRNA3 * SOCS3RNADelay) -
                        1.0 * (1 / nuc) * (nuc * SOCS3nRNA4 * SOCS3RNADelay),
        D(SOCS3RNA) ~ +1.0 * (1 / cyt) * (nuc * SOCS3nRNA5 * SOCS3RNADelay) -
                      1.0 * (1 / cyt) * (cyt * SOCS3RNA * SOCS3RNATurn),
        D(SHP1) ~ -1.0 * (1 / cyt) *
                  (cyt * (SHP1 * SHP1ActEpoR *
                    (EpoRpJAK2 + p12EpoRpJAK2 + p1EpoRpJAK2 + p2EpoRpJAK2) / init_EpoRJAK2)) +
                  1.0 * (1 / cyt) * (cyt * SHP1Dea * SHP1Act),
        D(STAT5) ~ -1.0 * (1 / cyt) *
                   (cyt * (STAT5 * STAT5ActJAK2 *
                     (EpoRpJAK2 + p12EpoRpJAK2 + p1EpoRpJAK2 + p2EpoRpJAK2) /
                     (init_EpoRJAK2 * (SOCS3 * SOCS3Inh / SOCS3Eqc + 1)))) -
                   1.0 * (1 / cyt) *
                   (cyt * (STAT5 * STAT5ActEpoR * (p12EpoRpJAK2 + p1EpoRpJAK2)^(2) /
                     ((init_EpoRJAK2)^(2) * (SOCS3 * SOCS3Inh / SOCS3Eqc + 1) *
                      (CIS * CISInh / CISEqc + 1)))) +
                   1.0 * (1 / cyt) * (nuc * STAT5Exp * npSTAT5),
        D(EpoRJAK2) ~ -1.0 * (1 / cyt) *
                      (cyt *
                       (Epo * EpoRJAK2 * JAK2ActEpo / (SOCS3 * SOCS3Inh / SOCS3Eqc + 1))) +
                      1.0 * (1 / cyt) *
                      (cyt * (EpoRpJAK2 * JAK2EpoRDeaSHP1 * SHP1Act / init_SHP1)) +
                      1.0 * (1 / cyt) *
                      (cyt * (JAK2EpoRDeaSHP1 * SHP1Act * p1EpoRpJAK2 / init_SHP1)) +
                      1.0 * (1 / cyt) *
                      (cyt * (JAK2EpoRDeaSHP1 * SHP1Act * p2EpoRpJAK2 / init_SHP1)) +
                      1.0 * (1 / cyt) *
                      (cyt * (JAK2EpoRDeaSHP1 * SHP1Act * p12EpoRpJAK2 / init_SHP1)),
        D(CISnRNA1) ~ +1.0 * (1 / nuc) *
                      (nuc * (CISRNAEqc * CISRNATurn * npSTAT5 * ActD / init_STAT5)) -
                      1.0 * (1 / nuc) * (nuc * CISnRNA1 * CISRNADelay),
        D(SOCS3nRNA1) ~ +1.0 * (1 / nuc) *
                        (nuc * (SOCS3RNAEqc * SOCS3RNATurn * npSTAT5 * ActD / init_STAT5)) -
                        1.0 * (1 / nuc) * (nuc * SOCS3nRNA1 * SOCS3RNADelay),
        D(SOCS3nRNA2) ~ +1.0 * (1 / nuc) * (nuc * SOCS3nRNA1 * SOCS3RNADelay) -
                        1.0 * (1 / nuc) * (nuc * SOCS3nRNA2 * SOCS3RNADelay),
        D(CISnRNA3) ~ +1.0 * (1 / nuc) * (nuc * CISnRNA2 * CISRNADelay) -
                      1.0 * (1 / nuc) * (nuc * CISnRNA3 * CISRNADelay),
        D(CISnRNA4) ~ +1.0 * (1 / nuc) * (nuc * CISnRNA3 * CISRNADelay) -
                      1.0 * (1 / nuc) * (nuc * CISnRNA4 * CISRNADelay),
        D(SOCS3) ~ +1.0 * (1 / cyt) *
                   (cyt * (SOCS3RNA * SOCS3Eqc * SOCS3Turn / SOCS3RNAEqc)) -
                   1.0 * (1 / cyt) * (cyt * SOCS3 * SOCS3Turn) +
                   1.0 * (1 / cyt) * (cyt * SOCS3oe * SOCS3Eqc * SOCS3Turn * SOCS3EqcOE),
        D(CISnRNA5) ~ +1.0 * (1 / nuc) * (nuc * CISnRNA4 * CISRNADelay) -
                      1.0 * (1 / nuc) * (nuc * CISnRNA5 * CISRNADelay),
        D(SOCS3nRNA5) ~ +1.0 * (1 / nuc) * (nuc * SOCS3nRNA4 * SOCS3RNADelay) -
                        1.0 * (1 / nuc) * (nuc * SOCS3nRNA5 * SOCS3RNADelay),
        D(SOCS3nRNA3) ~ +1.0 * (1 / nuc) * (nuc * SOCS3nRNA2 * SOCS3RNADelay) -
                        1.0 * (1 / nuc) * (nuc * SOCS3nRNA3 * SOCS3RNADelay),
        D(SHP1Act) ~ +1.0 * (1 / cyt) *
                     (cyt * (SHP1 * SHP1ActEpoR *
                       (EpoRpJAK2 + p12EpoRpJAK2 + p1EpoRpJAK2 + p2EpoRpJAK2) /
                       init_EpoRJAK2)) - 1.0 * (1 / cyt) * (cyt * SHP1Dea * SHP1Act),
        D(npSTAT5) ~ +1.0 * (1 / nuc) * (cyt * STAT5Imp * pSTAT5) -
                     1.0 * (1 / nuc) * (nuc * STAT5Exp * npSTAT5),
        D(p12EpoRpJAK2) ~ +1.0 * (1 / cyt) *
                          (cyt * (3 * EpoRActJAK2 * p1EpoRpJAK2 /
                            ((SOCS3 * SOCS3Inh / SOCS3Eqc + 1) *
                             (EpoRCISInh * EpoRJAK2_CIS + 1)))) +
                          1.0 * (1 / cyt) *
                          (cyt *
                           (EpoRActJAK2 * p2EpoRpJAK2 / (SOCS3 * SOCS3Inh / SOCS3Eqc + 1))) -
                          1.0 * (1 / cyt) *
                          (cyt * (JAK2EpoRDeaSHP1 * SHP1Act * p12EpoRpJAK2 / init_SHP1)),
        D(p2EpoRpJAK2) ~ +1.0 * (1 / cyt) *
                         (cyt * (3 * EpoRpJAK2 * EpoRActJAK2 /
                           ((SOCS3 * SOCS3Inh / SOCS3Eqc + 1) *
                            (EpoRCISInh * EpoRJAK2_CIS + 1)))) -
                         1.0 * (1 / cyt) *
                         (cyt *
                          (EpoRActJAK2 * p2EpoRpJAK2 / (SOCS3 * SOCS3Inh / SOCS3Eqc + 1))) -
                         1.0 * (1 / cyt) *
                         (cyt * (JAK2EpoRDeaSHP1 * SHP1Act * p2EpoRpJAK2 / init_SHP1)),
        D(CIS) ~ +1.0 * (1 / cyt) * (cyt * (CISRNA * CISEqc * CISTurn / CISRNAEqc)) -
                 1.0 * (1 / cyt) * (cyt * CIS * CISTurn) +
                 1.0 * (1 / cyt) * (cyt * CISEqc * CISTurn * CISEqcOE * CISoe),
        D(EpoRpJAK2) ~ +1.0 * (1 / cyt) *
                       (cyt *
                        (Epo * EpoRJAK2 * JAK2ActEpo / (SOCS3 * SOCS3Inh / SOCS3Eqc + 1))) -
                       1.0 * (1 / cyt) *
                       (cyt * (EpoRpJAK2 * JAK2EpoRDeaSHP1 * SHP1Act / init_SHP1)) -
                       1.0 * (1 / cyt) *
                       (cyt * (EpoRpJAK2 * EpoRActJAK2 / (SOCS3 * SOCS3Inh / SOCS3Eqc + 1))) -
                       1.0 * (1 / cyt) *
                       (cyt *
                        (3 * EpoRpJAK2 * EpoRActJAK2 / ((SOCS3 * SOCS3Inh / SOCS3Eqc + 1) *
                          (EpoRCISInh * EpoRJAK2_CIS + 1)))),
        D(CISnRNA2) ~ +1.0 * (1 / nuc) * (nuc * CISnRNA1 * CISRNADelay) -
                      1.0 * (1 / nuc) * (nuc * CISnRNA2 * CISRNADelay),
        D(CISRNA) ~ +1.0 * (1 / cyt) * (nuc * CISnRNA5 * CISRNADelay) -
                    1.0 * (1 / cyt) * (cyt * CISRNA * CISRNATurn),
    ]

    @named sys = ODESystem(eqs, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    initialSpeciesValues = [
        p1EpoRpJAK2 => 0.0,
        pSTAT5 => 0.0,
        EpoRJAK2_CIS => init_EpoRJAK2_CIS,
        SOCS3nRNA4 => 0.0,
        SOCS3RNA => 0.0,
        SHP1 => init_SHP1 * (init_SHP1_multiplier * SHP1ProOE + 1),
        STAT5 => init_STAT5,
        EpoRJAK2 => init_EpoRJAK2,
        CISnRNA1 => 0.0,
        SOCS3nRNA1 => 0.0,
        SOCS3nRNA2 => 0.0,
        CISnRNA3 => 0.0,
        CISnRNA4 => 0.0,
        SOCS3 => init_SOCS3_multiplier * SOCS3EqcOE * SOCS3Eqc,
        CISnRNA5 => 0.0,
        SOCS3nRNA5 => 0.0,
        SOCS3nRNA3 => 0.0,
        SHP1Act => 0.0,
        npSTAT5 => 0.0,
        p12EpoRpJAK2 => 0.0,
        p2EpoRpJAK2 => 0.0,
        CIS => init_CIS_multiplier * CISEqc * CISEqcOE,
        EpoRpJAK2 => 0.0,
        CISnRNA2 => 0.0,
        CISRNA => 0.0,
    ]

    ### SBML file parameter values ###
    trueParameterValues = [
        STAT5Exp => 0.0745150819016423,
        STAT5Imp => 0.0268865083829685,
        init_SOCS3_multiplier => 0.0,
        EpoRCISRemove => 5.42980693903448,
        STAT5ActEpoR => 38.9957991073948,
        SHP1ActEpoR => 0.00100000000000006,
        JAK2EpoRDeaSHP1 => 142.72332309738,
        CISTurn => 0.0083988695167017,
        SOCS3Turn => 9999.99999999912,
        init_EpoRJAK2_CIS => 0.0,
        SOCS3Inh => 10.4078649133666,
        ActD => 1.25e-7,
        init_CIS_multiplier => 0.0,
        cyt => 0.4,
        CISRNAEqc => 1.0,
        JAK2ActEpo => 633167.430600806,
        Epo => 1.25e-7,
        SOCS3oe => 1.25e-7,
        CISInh => 7.85269991450496e8,
        SHP1Dea => 0.00816220490950374,
        SOCS3EqcOE => 0.679165515556864,
        CISRNADelay => 0.14477775532111,
        init_SHP1 => 26.7251164277109,
        CISEqcOE => 0.530264447119609,
        EpoRActJAK2 => 0.267304849333058,
        SOCS3RNAEqc => 1.0,
        CISEqc => 432.860413434913,
        SHP1ProOE => 2.82568153411555,
        SOCS3RNADelay => 1.06458446742251,
        init_STAT5 => 79.75363993771,
        CISoe => 1.25e-7,
        CISRNATurn => 999.999999999946,
        init_SHP1_multiplier => 1.0,
        init_EpoRJAK2 => 3.97622369384192,
        nuc => 0.275,
        EpoRCISInh => 999999.999999912,
        STAT5ActJAK2 => 0.0781068855795467,
        SOCS3RNATurn => 0.00830917643120369,
        SOCS3Eqc => 173.64470023136,
    ]

    return sys, initialSpeciesValues, trueParameterValues
end
