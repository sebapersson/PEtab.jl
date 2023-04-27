# Model name: Brannmark
# Number of parameters: 20
# Number of species: 9
function getODEModel_Brannmark()

    ### Define independent and dependent variables
    ModelingToolkit.@variables t IRp(t) IR(t) IRins(t) IRiP(t) IRS(t) X(t) IRi(t) IRSiP(t) Xp(t)

    ### Store dependent variables in array for ODESystem command
    stateArray = [IRp, IR, IRins, IRiP, IRS, X, IRi, IRSiP, Xp]

    ### Define variable parameters

    ### Define potential algebraic variables
    ModelingToolkit.@variables insulin(t)

    ### Define parameters
    ModelingToolkit.@parameters k1c k21 k1g insulin_dose_2 k1a insulin_dose_1 k1aBasic k1d insulin_time_1 insulin_time_2 cyt k22 default k1r k1f k1b k3 km2 k1e k_IRSiP_DosR km3

    ### Store parameters in array for ODESystem command
    parameterArray = [k1c, k21, k1g, insulin_dose_2, k1a, insulin_dose_1, k1aBasic, k1d, insulin_time_1, insulin_time_2, cyt, k22, default, k1r, k1f, k1b, k3, km2, k1e, k_IRSiP_DosR, km3]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Continious events ###

    ### Discrete events ###

    ### Derivatives ###
    Eqns = [
    D(IRp) ~ +1.0 * ( 1 /cyt ) * (cyt * IRins * k1c)-1.0 * ( 1 /cyt ) * (cyt * IRp * k1d)-1.0 * ( 1 /cyt ) * (cyt * IRp * k1g),
    D(IR) ~ -1.0 * ( 1 /cyt ) * (cyt * (IR * k1aBasic + IR * insulin * k1a))+1.0 * ( 1 /cyt ) * (cyt * IRins * k1b)+1.0 * ( 1 /cyt ) * (cyt * IRp * k1g)+1.0 * ( 1 /cyt ) * (cyt * IRi * k1r),
    D(IRins) ~ +1.0 * ( 1 /cyt ) * (cyt * (IR * k1aBasic + IR * insulin * k1a))-1.0 * ( 1 /cyt ) * (cyt * IRins * k1b)-1.0 * ( 1 /cyt ) * (cyt * IRins * k1c),
    D(IRiP) ~ +1.0 * ( 1 /cyt ) * (cyt * IRp * k1d)-1.0 * ( 1 /cyt ) * (cyt * IRiP * (k1e + Xp * k1f / (Xp + 1))),
    D(IRS) ~ -1.0 * ( 1 /cyt ) * (cyt * IRS * k21 * (IRp + IRiP * k22))+1.0 * ( 1 /cyt ) * (cyt * IRSiP * km2),
    D(X) ~ -1.0 * ( 1 /cyt ) * (cyt * IRSiP * X * k3)+1.0 * ( 1 /cyt ) * (cyt * Xp * km3),
    D(IRi) ~ +1.0 * ( 1 /cyt ) * (cyt * IRiP * (k1e + Xp * k1f / (Xp + 1)))-1.0 * ( 1 /cyt ) * (cyt * IRi * k1r),
    D(IRSiP) ~ +1.0 * ( 1 /cyt ) * (cyt * IRS * k21 * (IRp + IRiP * k22))-1.0 * ( 1 /cyt ) * (cyt * IRSiP * km2),
    D(Xp) ~ +1.0 * ( 1 /cyt ) * (cyt * IRSiP * X * k3)-1.0 * ( 1 /cyt ) * (cyt * Xp * km3),
    insulin ~ insulin_dose_1 * ifelse(t - insulin_time_1 < 0, 0, 1) + insulin_dose_2 * ifelse(t - insulin_time_2 < 0, 0, 1)
    ]

    @named odeSYS = ODESystem(Eqns, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    iSV = [
    IRp => 1.7629010620181e-9,
    IR => 9.94957642787569,
    IRins => 0.0173972221725393,
    IRiP => 1.11590026152296e-5,
    IRS => 9.86699348701367,
    X => 9.99984199487351,
    IRi => 0.0330151891862681,
    IRSiP => 0.133006512986336,
    Xp => 0.000158005126497888
    ]

    ### SBML file parameter values ###
    tPVal = [
    k1c => 0.050861851404055,
    k21 => 2.13019897196189,
    k1g => 1931.1338834437,
    insulin_dose_2 => 0.0,
    k1a => 0.177252330941141,
    insulin_dose_1 => 0.3,
    k1aBasic => 0.000394105679186913,
    k1d => 499999.999999974,
    insulin_time_1 => 0.0,
    insulin_time_2 => 1000.0,
    cyt => 1.0,
    k22 => 658.762927786248,
    default => 1.0,
    k1r => 0.0266983879216281,
    k1f => 499999.990737798,
    k1b => 0.174529566448397,
    k3 => 4.94369803061052e-5,
    km2 => 1.16168060611079,
    k1e => 1.00000000000005e-6,
    k_IRSiP_DosR => 37.9636812744313,
    km3 => 0.416147033419453
    ]

    return odeSYS, iSV, tPVal

end
