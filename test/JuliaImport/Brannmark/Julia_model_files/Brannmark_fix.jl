# Model name: Brannmark
# Number of parameters: 21
# Number of species: 9
function getODEModel_Brannmark()

    ### Define independent and dependent variables
    ModelingToolkit.@variables t IRins(t) X(t) IRp(t) IRS(t) IR(t) IRi(t) IRiP(t) IRSiP(t) Xp(t) 

    ### Store dependent variables in array for ODESystem command
    stateArray = [IRins, X, IRp, IRS, IR, IRi, IRiP, IRSiP, Xp]

    ### Define variable parameters

    ### Define potential algebraic variables
    ModelingToolkit.@variables insulin(t) 

    ### Define parameters
    ModelingToolkit.@parameters k1c k21 insulin_bool1 k1g insulin_dose_2 k1a insulin_dose_1 k1aBasic insulin_time_1 insulin_time_2 k1d cyt k22 insulin_bool2 default k1r k1f k1b k3 km2 k1e k_IRSiP_DosR km3 

    ### Store parameters in array for ODESystem command
    parameterArray = [k1c, k21, insulin_bool1, k1g, insulin_dose_2, k1a, insulin_dose_1, k1aBasic, insulin_time_1, insulin_time_2, k1d, cyt, k22, insulin_bool2, default, k1r, k1f, k1b, k3, km2, k1e, k_IRSiP_DosR, km3]

    ### Define an operator for the differentiation w.r.t. time
    D = Differential(t)

    ### Continious events ###

    ### Discrete events ###

    ### Derivatives ###
    eqs = [
    D(IRp) ~ k1c*IRins - k1d*IRp - k1g*IRp,
    D(IR) ~ k1b*IRins + k1g*IRp + k1r*IRi - k1aBasic*IR - k1a*IR*insulin,
    D(IRins) ~ k1aBasic*IR + k1a*IR*insulin - k1b*IRins - k1c*IRins,
    D(IRiP) ~ k1d*IRp + ((-k1f*Xp) / (1 + Xp) - k1e)*IRiP,
    D(IRS) ~ km2*IRSiP - k21*(k22*IRiP + IRp)*IRS,
    D(X) ~ km3*Xp - k3*IRSiP*X,
    D(IRi) ~ (k1e + (k1f*Xp) / (1 + Xp))*IRiP - k1r*IRi,
    D(IRSiP) ~ k21*(k22*IRiP + IRp)*IRS - km2*IRSiP,
    D(Xp) ~ k3*IRSiP*X - km3*Xp,
    insulin~insulin_dose_1*((1 - insulin_bool1)*( 0) + insulin_bool1*( 1)) + insulin_dose_2*((1 - insulin_bool2)*( 0) + insulin_bool2*( 1))
    ]
    @named sys = ODESystem(eqs, t, stateArray, parameterArray)

    ### Initial species concentrations ###
    initialSpeciesValues = [
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
    trueParameterValues = [
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
        km3 => 0.416147033419453, 
        insulin_bool1 => 0.0, 
        insulin_bool2 => 0.0
    ]

    return sys, initialSpeciesValues, trueParameterValues

end
