function getODEModel_Fujita()
	# Model name: Fujita
	# Number of parameters: 20
	# Number of species: 9

    ### Define independent and dependent variables
    ModelingToolkit.@variables t pEGFR_Akt(t) EGF_EGFR(t) Akt(t) pS6(t) pAkt(t) EGFR(t) pEGFR(t) S6(t) pAkt_S6(t) EGF(t) 

    ### Store dependent variables in array for ODESystem command
    stateArray = [pEGFR_Akt, EGF_EGFR, Akt, pS6, pAkt, EGFR, pEGFR, S6, pAkt_S6]

    ### Define variable parameters

    ### Define potential algebraic variables

    ### Define parameters
    ModelingToolkit.@parameters EGF_end reaction_5_k1 reaction_2_k2 init_AKT init_EGFR EGF_bool1 EGF_rate EGFR_turnover reaction_1_k1 reaction_1_k2 reaction_8_k1 reaction_4_k1 reaction_6_k1 reaction_2_k1 init_S6 reaction_7_k1 reaction_9_k1 reaction_3_k1 reaction_5_k2 Cell EGF_0 

    ### Store parameters in array for ODESystem command
    parameterArray = [EGF_end, reaction_5_k1, reaction_2_k2, init_AKT, init_EGFR, EGF_bool1, EGF_rate, EGFR_turnover, reaction_1_k1, reaction_1_k2, reaction_8_k1, reaction_4_k1, reaction_6_k1, reaction_2_k1, init_S6, reaction_7_k1, reaction_9_k1, reaction_3_k1, reaction_5_k2, Cell, EGF_0]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Continious events ###

    ### Discrete events ###

    ### Derivatives ###
    eqs = [
    D(pAkt_S6) ~ reaction_5_k1*S6*pAkt - reaction_5_k2*pAkt_S6 - reaction_6_k1*pAkt_S6,
    D(pAkt) ~ reaction_3_k1*pEGFR_Akt + reaction_5_k2*pAkt_S6 + reaction_6_k1*pAkt_S6 - reaction_7_k1*pAkt - reaction_5_k1*S6*pAkt,
    D(pS6) ~ reaction_6_k1*pAkt_S6 - reaction_8_k1*pS6,
    D(EGFR) ~ 68190EGFR_turnover + reaction_1_k2*EGF_EGFR - EGFR_turnover*EGFR - reaction_1_k1*EGF*EGFR,
    D(pEGFR_Akt) ~ reaction_2_k1*Akt*pEGFR - reaction_2_k2*pEGFR_Akt - reaction_3_k1*pEGFR_Akt,
    D(pEGFR) ~ reaction_9_k1*EGF_EGFR + reaction_2_k2*pEGFR_Akt + reaction_3_k1*pEGFR_Akt - reaction_4_k1*pEGFR - reaction_2_k1*Akt*pEGFR,
    D(Akt) ~ reaction_7_k1*pAkt + reaction_2_k2*pEGFR_Akt - reaction_2_k1*Akt*pEGFR,
    D(S6) ~ reaction_5_k2*pAkt_S6 + reaction_8_k1*pS6 - reaction_5_k1*S6*pAkt,
    D(EGF_EGFR) ~ reaction_1_k1*EGF*EGFR - reaction_1_k2*EGF_EGFR - reaction_9_k1*EGF_EGFR,
    EGF~((1 - EGF_bool1)*( EGF_0 + EGF_rate*t) + EGF_bool1*( 0))
    ]

    @named sys = ODESystem(eqs, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    initialSpeciesValues = [
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
    trueParameterValues = [
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
        EGF_0 => 0.0, 
        EGF_bool1 => 0.0
    ]

    return sys, initialSpeciesValues, trueParameterValues

end
