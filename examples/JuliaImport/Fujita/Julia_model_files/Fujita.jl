# Model name: Fujita
# Number of parameters: 20
# Number of species: 9
function getODEModel_Fujita()

    ### Define independent and dependent variables
    ModelingToolkit.@variables t pAkt_S6(t) pAkt(t) pS6(t) EGFR(t) pEGFR_Akt(t) pEGFR(t) Akt(t) S6(t) EGF_EGFR(t)

    ### Store dependent variables in array for ODESystem command
    stateArray = [pAkt_S6, pAkt, pS6, EGFR, pEGFR_Akt, pEGFR, Akt, S6, EGF_EGFR]

    ### Define variable parameters

    ### Define potential algebraic variables
    ModelingToolkit.@variables EGF(t)

    ### Define parameters
    ModelingToolkit.@parameters EGF_end reaction_5_k1 reaction_2_k2 init_AKT init_EGFR EGF_rate EGFR_turnover reaction_1_k1 reaction_1_k2 reaction_8_k1 reaction_4_k1 reaction_6_k1 reaction_2_k1 init_S6 reaction_7_k1 reaction_9_k1 reaction_3_k1 reaction_5_k2 Cell EGF_0

    ### Store parameters in array for ODESystem command
    parameterArray = [EGF_end, reaction_5_k1, reaction_2_k2, init_AKT, init_EGFR, EGF_rate, EGFR_turnover, reaction_1_k1, reaction_1_k2, reaction_8_k1, reaction_4_k1, reaction_6_k1, reaction_2_k1, init_S6, reaction_7_k1, reaction_9_k1, reaction_3_k1, reaction_5_k2, Cell, EGF_0]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Continious events ###

    ### Discrete events ###

    ### Derivatives ###
    Eqn = [
    D(pAkt_S6) ~ +1.0 * ( 1 /Cell ) * (Cell * (S6 * pAkt * reaction_5_k1 - pAkt_S6 * reaction_5_k2))-1.0 * ( 1 /Cell ) * (Cell * pAkt_S6 * reaction_6_k1),
    D(pAkt) ~ +1.0 * ( 1 /Cell ) * (Cell * pEGFR_Akt * reaction_3_k1)-1.0 * ( 1 /Cell ) * (Cell * (S6 * pAkt * reaction_5_k1 - pAkt_S6 * reaction_5_k2))+1.0 * ( 1 /Cell ) * (Cell * pAkt_S6 * reaction_6_k1)-1.0 * ( 1 /Cell ) * (Cell * pAkt * reaction_7_k1),
    D(pS6) ~ +1.0 * ( 1 /Cell ) * (Cell * pAkt_S6 * reaction_6_k1)-1.0 * ( 1 /Cell ) * (Cell * pS6 * reaction_8_k1),
    D(EGFR) ~ -1.0 * ( 1 /Cell ) * (Cell * (EGF * EGFR * reaction_1_k1 - EGF_EGFR * reaction_1_k2))-1.0 * ( 1 /Cell ) * (Cell * EGFR * EGFR_turnover)+1.0 * ( 1 /Cell ) * (Cell * 68190 * EGFR_turnover),
    D(pEGFR_Akt) ~ +1.0 * ( 1 /Cell ) * (Cell * (Akt * pEGFR * reaction_2_k1 - pEGFR_Akt * reaction_2_k2))-1.0 * ( 1 /Cell ) * (Cell * pEGFR_Akt * reaction_3_k1),
    D(pEGFR) ~ -1.0 * ( 1 /Cell ) * (Cell * (Akt * pEGFR * reaction_2_k1 - pEGFR_Akt * reaction_2_k2))+1.0 * ( 1 /Cell ) * (Cell * pEGFR_Akt * reaction_3_k1)-1.0 * ( 1 /Cell ) * (Cell * pEGFR * reaction_4_k1)+1.0 * ( 1 /Cell ) * (Cell * EGF_EGFR * reaction_9_k1),
    D(Akt) ~ -1.0 * ( 1 /Cell ) * (Cell * (Akt * pEGFR * reaction_2_k1 - pEGFR_Akt * reaction_2_k2))+1.0 * ( 1 /Cell ) * (Cell * pAkt * reaction_7_k1),
    D(S6) ~ -1.0 * ( 1 /Cell ) * (Cell * (S6 * pAkt * reaction_5_k1 - pAkt_S6 * reaction_5_k2))+1.0 * ( 1 /Cell ) * (Cell * pS6 * reaction_8_k1),
    D(EGF_EGFR) ~ +1.0 * ( 1 /Cell ) * (Cell * (EGF * EGFR * reaction_1_k1 - EGF_EGFR * reaction_1_k2))-1.0 * ( 1 /Cell ) * (Cell * EGF_EGFR * reaction_9_k1),
    EGF ~ ifelse(t <= EGF_end, EGF_rate * t + EGF_0, 0)
    ]

    @named odeSYSTEMS = ODESystem(Eqn, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    initSV = [
    pAkt_S6 => 0.0,
    pAkt => 0.0,
    pS6 => 0.0,
    EGFR => init_EGFR,
    pEGFR_Akt => 0.0,
    pEGFR => 0.0,
    Akt => init_AKT,
    S6 => init_S6,
    EGF_EGFR => 0.0
    ]

    ### SBML file parameter values ###
    truePV = [
    EGF_end => 0.0,
    reaction_5_k1 => 2.9643709900602e-6,
    reaction_2_k2 => 41469.6914053245,
    init_AKT => 0.00332683237159935,
    init_EGFR => 2.26508055977911e7,
    EGF_rate => 0.0,
    EGFR_turnover => 0.001449799125736,
    reaction_1_k1 => 0.00372345533395159,
    reaction_1_k2 => 0.00262709856442467,
    reaction_8_k1 => 0.000941161525754959,
    reaction_4_k1 => 0.0308146966905863,
    reaction_6_k1 => 9.20585474645043e-6,
    reaction_2_k1 => 0.00103236148008131,
    init_S6 => 205.86301335244,
    reaction_7_k1 => 0.0119329694583145,
    reaction_9_k1 => 0.0273281571867514,
    reaction_3_k1 => 0.454840577578597,
    reaction_5_k2 => 0.000404055756190126,
    Cell => 1.0,
    EGF_0 => 0.0
    ]

    return odeSYSTEMS, initSV, truePV

end
