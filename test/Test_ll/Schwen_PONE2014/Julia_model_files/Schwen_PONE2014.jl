# Model name: Schwen_PONE2014
# Number of parameters: 14
# Number of species: 11
function getODEModel_Schwen_PONE2014()

    ### Define independent and dependent variables
    ModelingToolkit.@variables t IR2(t) IR2in(t) Rec2(t) IR1in(t) Uptake1(t) Uptake2(t) InsulinFragments(t) IR1(t) Rec1(t) Ins(t) BoundUnspec(t)

    ### Store dependent variables in array for ODESystem command
    stateArray = [IR2, IR2in, Rec2, IR1in, Uptake1, Uptake2, InsulinFragments, IR1, Rec1, Ins, BoundUnspec]

    ### Define variable parameters

    ### Define potential algebraic variables

    ### Define parameters
    ModelingToolkit.@parameters ka1 ini_R2fold kout ini_R1 kout_frag koff_unspec kin ka2fold kin2 kd1 kon_unspec init_Ins kd2fold ExtracellularMedium kout2

    ### Store parameters in array for ODESystem command
    parameterArray = [ka1, ini_R2fold, kout, ini_R1, kout_frag, koff_unspec, kin, ka2fold, kin2, kd1, kon_unspec, init_Ins, kd2fold, ExtracellularMedium, kout2]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Continious events ###

    ### Discrete events ###

    ### Derivatives ###
    eqs = [
    D(IR2) ~ +1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * Ins * Rec2 * ka1 * ka2fold)-1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR2 * kd1 * kd2fold)-1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR2 * kin2)+1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR2in * kout2),
    D(IR2in) ~ +1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR2 * kin2)-1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR2in * kout2)-1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR2in * kout_frag),
    D(Rec2) ~ -1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * Ins * Rec2 * ka1 * ka2fold)+1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR2 * kd1 * kd2fold)+1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR2in * kout_frag),
    D(IR1in) ~ +1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR1 * kin)-1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR1in * kout)-1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR1in * kout_frag),
    D(Uptake1) ~ +1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * (Ins * Rec1 * ka1 - IR1 * kd1)),
    D(Uptake2) ~ +1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * (Ins * Rec2 * ka1 * ka2fold - IR2 * kd1 * kd2fold)),
    D(InsulinFragments) ~ +1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR1in * kout_frag)+1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR2in * kout_frag),
    D(IR1) ~ +1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * Ins * Rec1 * ka1)-1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR1 * kd1)-1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR1 * kin)+1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR1in * kout),
    D(Rec1) ~ -1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * Ins * Rec1 * ka1)+1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR1 * kd1)+1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR1in * kout_frag),
    D(Ins) ~ -1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * Ins * Rec1 * ka1)-1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * Ins * Rec2 * ka1 * ka2fold)-1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * Ins * kon_unspec)+1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * BoundUnspec * koff_unspec)+1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR1 * kd1)+1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * IR2 * kd1 * kd2fold),
    D(BoundUnspec) ~ +1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * Ins * kon_unspec)-1.0 * ( 1 /ExtracellularMedium ) * (ExtracellularMedium * BoundUnspec * koff_unspec)
    ]

    @named sys = ODESystem(eqs, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    initialSpeciesValues = [
    IR2 => 0.0,
    IR2in => 0.0,
    Rec2 => ini_R1 * ini_R2fold,
    IR1in => 0.0,
    Uptake1 => 0.0,
    Uptake2 => 0.0,
    InsulinFragments => 0.0,
    IR1 => 0.0,
    Rec1 => ini_R1,
    Ins => init_Ins,
    BoundUnspec => 0.0
    ]

    ### SBML file parameter values ###
    trueParameterValues = [
    ka1 => 0.00937980436663883,
    ini_R2fold => 16.457631927604,
    kout => 324.725838145278,
    ini_R1 => 47.2370172820096,
    kout_frag => 0.0100421669378689,
    koff_unspec => 9.6457882009227,
    kin => 3.75654890101743,
    ka2fold => 2.0907692381484,
    kin2 => 0.545304029714509,
    kd1 => 6.72269169161034,
    kon_unspec => 19.941427249128,
    init_Ins => 0.0,
    kd2fold => 9.61850107655493,
    ExtracellularMedium => 1.0,
    kout2 => 0.0529079560976487
    ]

    return sys, initialSpeciesValues, trueParameterValues

end
